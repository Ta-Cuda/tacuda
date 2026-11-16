/* TA-LIB Copyright (c) 1999-2026, Mario Fortier
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or
* without modification, are permitted provided that the following
* conditions are met:
*
* - Redistributions of source code must retain the above copyright
*   notice, this list of conditions and the following disclaimer.
*
* - Redistributions in binary form must reproduce the above copyright
*   notice, this list of conditions and the following disclaimer in
*   the documentation and/or other materials provided with the
*   distribution.
*
* - Neither name of author nor the names of its contributors
*   may be used to endorse or promote products derived from this
*   software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
* OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
* WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
* OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
* EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/* Pinned batch-only extraction; see PROVENANCE.json. */

TA_LIB_API TA_RetCode TA_MIDPRICE( int    startIdx,
                                   int    endIdx,
                                   const double inHigh[],
                                   const double inLow[],
                                   int optInTimePeriod,
                                   int          *outBegIdx,
                                   int          *outNBElement,
                                   double        outReal[] )
{
   double local_sufHighest[30];
   double *sufHighest = &local_sufHighest[0];
   double local_preHighest[30];
   double *preHighest = &local_preHighest[0];
   double local_sufLowest[30];
   double *sufLowest = &local_sufLowest[0];
   double local_preLowest[30];
   double *preLowest = &local_preLowest[0];
   double lowest;
   double highest;
   double tmpLow;
   double tmpHigh;
   int outIdx;
   int nbInitialElementNeeded;
   int trailingIdx;
   int today;
   int i;
   int blockStart;
   int nAvail;
   int m;
   int blockNext;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* MIDPRICE = (Highest High + Lowest Low)/2
    *
    * This function is equivalent to MEDPRICE when the
    * period is 1.
    */
   /* Identify the minimum number of price bar needed
    * to identify at least one output over the specified
    * period.
    */
   nbInitialElementNeeded = optInTimePeriod - 1;
   /* Move up the start index if there is not
    * enough initial data.
    */
   if( startIdx < nbInitialElementNeeded )
   {
      startIdx = nbInitialElementNeeded;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   /* Proceed with the calculation for the requested range.
    * Note that this algorithm allows the input and
    * output to be the same buffer.
    *
    * Van Herk / Gil-Werman block scan, block-batched form. The p outputs
    * belonging to one block boundary are produced together: one backward
    * pass builds the older block's suffix extrema, one forward pass builds
    * the newer block's prefix extrema, and a third pass combines them.
    * Both extrema travel in the same passes.
    * All the loops are straight-line with no data-dependent branching,
    * which is what lets a compiler vectorize them, and the work per bar is
    * a fixed number of comparisons regardless of period. Every scratch
    * array holds COPIES, so input and output may alias.
    *
    * Producing a whole block at a time is also why this cannot be turned
    * into a per-bar automaton, so the streaming tier runs midprice_ALT1
    * below. See issue #147.
    */
   outIdx = 0;
   today = startIdx;
   trailingIdx = startIdx - nbInitialElementNeeded;
   if( optInTimePeriod < 1 ) return TA_INTERNAL_ERROR(358);
   if( (int)optInTimePeriod > (int)(sizeof(local_sufHighest)/sizeof(double)) )
   {
      sufHighest = TA_Malloc( sizeof(double)*optInTimePeriod );
      if( !sufHighest )
      {
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      sufHighest = &local_sufHighest[0];
   }
   if( optInTimePeriod < 1 ) return TA_INTERNAL_ERROR(359);
   if( (int)optInTimePeriod > (int)(sizeof(local_preHighest)/sizeof(double)) )
   {
      preHighest = TA_Malloc( sizeof(double)*optInTimePeriod );
      if( !preHighest )
      {
         if( sufHighest != &local_sufHighest[0] ) TA_Free( sufHighest );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      preHighest = &local_preHighest[0];
   }
   if( optInTimePeriod < 1 ) return TA_INTERNAL_ERROR(360);
   if( (int)optInTimePeriod > (int)(sizeof(local_sufLowest)/sizeof(double)) )
   {
      sufLowest = TA_Malloc( sizeof(double)*optInTimePeriod );
      if( !sufLowest )
      {
         if( sufHighest != &local_sufHighest[0] ) TA_Free( sufHighest );
         if( preHighest != &local_preHighest[0] ) TA_Free( preHighest );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      sufLowest = &local_sufLowest[0];
   }
   if( optInTimePeriod < 1 ) return TA_INTERNAL_ERROR(361);
   if( (int)optInTimePeriod > (int)(sizeof(local_preLowest)/sizeof(double)) )
   {
      preLowest = TA_Malloc( sizeof(double)*optInTimePeriod );
      if( !preLowest )
      {
         if( sufHighest != &local_sufHighest[0] ) TA_Free( sufHighest );
         if( preHighest != &local_preHighest[0] ) TA_Free( preHighest );
         if( sufLowest != &local_sufLowest[0] ) TA_Free( sufLowest );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      preLowest = &local_preLowest[0];
   }
   blockStart = trailingIdx;
   while( today <= endIdx )
   {
      /* Suffix extrema of the block [blockStart, blockStart+p-1], which
       * is fully available here: today == blockStart+p-1 <= endIdx.
       * Scanning backward while keeping the incumbent on a tie
       * leaves the later element holding a tie, which is what lets this
       * compile to a single min/max instruction.
       */
      i = blockStart + optInTimePeriod - 1;
      highest = inHigh[i];
      lowest = inLow[i];
      sufHighest[optInTimePeriod - 1] = highest;
      sufLowest[optInTimePeriod - 1] = lowest;
      TA_UNROLL(4)
      while( i > blockStart )
      {
         i -= 1;
         tmpHigh = inHigh[i];
         if( tmpHigh > highest )
         {
            highest = tmpHigh;
         }
         tmpLow = inLow[i];
         if( tmpLow < lowest )
         {
            lowest = tmpLow;
         }
         sufHighest[i - blockStart] = highest;
         sufLowest[i - blockStart] = lowest;
      }
      outReal[outIdx++] = (sufHighest[0] + sufLowest[0]) / 2.0;
      trailingIdx += 1;
      today += 1;
      if( today > endIdx )
      {
         blockStart = blockStart + optInTimePeriod;
      } else 
      {
         /* Prefix extrema of the next block, clamped to what remains.
          * Forward, keeping the incumbent on a tie: earliest wins again.
          */
         blockNext = blockStart + optInTimePeriod;
         nAvail = endIdx - blockNext + 1;
         if( nAvail > optInTimePeriod - 1 )
         {
            nAvail = optInTimePeriod - 1;
         }
         highest = inHigh[blockNext];
         lowest = inLow[blockNext];
         preHighest[0] = highest;
         preLowest[0] = lowest;
         i = 1;
         TA_UNROLL(4)
         while( i < nAvail )
         {
            tmpHigh = inHigh[blockNext + i];
            if( tmpHigh > highest )
            {
               highest = tmpHigh;
            }
            tmpLow = inLow[blockNext + i];
            if( tmpLow < lowest )
            {
               lowest = tmpLow;
            }
            preHighest[i] = highest;
            preLowest[i] = lowest;
            i += 1;
         }
         /* Combine. The suffix half is the older one, so preferring it
          * on a tie keeps the earliest-wins rule.
          */
         m = 1;
         while( m <= nAvail )
         {
            highest = sufHighest[m];
            if( preHighest[m - 1] > highest )
            {
               highest = preHighest[m - 1];
            }
            lowest = sufLowest[m];
            if( preLowest[m - 1] < lowest )
            {
               lowest = preLowest[m - 1];
            }
            outReal[outIdx++] = (highest + lowest) / 2.0;
            m += 1;
         }
         trailingIdx = trailingIdx + nAvail;
         today = today + nAvail;
         blockStart = blockStart + optInTimePeriod;
      }
   }
   if( sufHighest != &local_sufHighest[0] ) { TA_Free( sufHighest ); sufHighest = &local_sufHighest[0]; }
   if( preHighest != &local_preHighest[0] ) { TA_Free( preHighest ); preHighest = &local_preHighest[0]; }
   if( sufLowest != &local_sufLowest[0] ) { TA_Free( sufLowest ); sufLowest = &local_sufLowest[0]; }
   if( preLowest != &local_preLowest[0] ) { TA_Free( preLowest ); preLowest = &local_preLowest[0]; }
   /* Keep the outBegIdx relative to the
    * caller input before returning.
    */
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_MIDPRICE_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   return optInTimePeriod - 1;
}
