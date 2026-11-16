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

TA_FMA_MULTIVERSION
TA_LIB_API TA_RetCode TA_FISHER( int    startIdx,
                                 int    endIdx,
                                 const double inHigh[],
                                 const double inLow[],
                                 int optInTimePeriod,
                                 int          *outBegIdx,
                                 int          *outNBElement,
                                 double        outFisher[],
                                 double        outTrigger[] )
{
   double local_sufHighest[30];
   double *sufHighest = &local_sufHighest[0];
   double local_preHighest[30];
   double *preHighest = &local_preHighest[0];
   double local_sufLowest[30];
   double *sufLowest = &local_sufLowest[0];
   double local_preLowest[30];
   double *preLowest = &local_preLowest[0];
   double price;
   double highest;
   double lowest;
   double ratio;
   double smoothed;
   double fish;
   double prevFish;
   double tempReal;
   double tempHigh;
   double tempLow;
   int outIdx;
   int lookbackTotal;
   int unstablePeriod;
   int nbInitialElementNeeded;
   int today;
   int i;
   int m;
   int blockStart;
   int blockNext;
   int nAvail;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 10;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outFisher )
      return TA_BAD_PARAM;
   if( !outTrigger )
      return TA_BAD_PARAM;
   if( outFisher == outTrigger )
      return TA_BAD_PARAM;

   nbInitialElementNeeded = TA_FISHER_Lookback(optInTimePeriod);
   if( startIdx < nbInitialElementNeeded )
   {
      startIdx = nbInitialElementNeeded;
   }
   if( startIdx > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   lookbackTotal = optInTimePeriod - 1;
   unstablePeriod = nbInitialElementNeeded - lookbackTotal;
   /* Same values as fisher_ALT1 below, which carries the formula. Only the
    * channel differs: a Van Herk / Gil-Werman block scan (WILLR's, issue
    * #147) over the midpoints, so the cost per bar does not depend on the
    * period or on the shape of the input, where the cached extremum of
    * fisher_ALT1 rescans its whole window on every bar of a flat or
    * trending stretch. Every scratch array holds copies, so an output may
    * alias an input.
    */
   smoothed = 0.0;
   prevFish = 0.0;
   today = startIdx - unstablePeriod;
   blockStart = today - lookbackTotal;
   outIdx = 0;
   if( optInTimePeriod < 1 ) return TA_INTERNAL_ERROR(486);
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
   if( optInTimePeriod < 1 ) return TA_INTERNAL_ERROR(487);
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
   if( optInTimePeriod < 1 ) return TA_INTERNAL_ERROR(488);
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
   if( optInTimePeriod < 1 ) return TA_INTERNAL_ERROR(489);
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
   while( today <= endIdx )
   {
      /* Suffix extrema of the block [blockStart, today]. */
      i = today;
      tempHigh = inHigh[i];
      tempLow = inLow[i];
      highest = (tempHigh + tempLow) / 2.0;
      lowest = highest;
      sufHighest[optInTimePeriod - 1] = highest;
      sufLowest[optInTimePeriod - 1] = lowest;
      TA_UNROLL(4)
      while( i > blockStart )
      {
         i -= 1;
         tempHigh = inHigh[i];
         tempLow = inLow[i];
         tempReal = (tempHigh + tempLow) / 2.0;
         if( tempReal > highest )
         {
            highest = tempReal;
         }
         if( tempReal < lowest )
         {
            lowest = tempReal;
         }
         sufHighest[i - blockStart] = highest;
         sufLowest[i - blockStart] = lowest;
      }
      /* Prefix extrema of the next block, clamped to what remains, stored
       * one slot up: slot 0 repeats the suffix so that bar 'today', whose
       * window is the block itself, runs the same combine as the others.
       */
      blockNext = blockStart + optInTimePeriod;
      nAvail = endIdx + 1 - blockNext;
      if( nAvail > optInTimePeriod - 1 )
      {
         nAvail = optInTimePeriod - 1;
      }
      preHighest[0] = sufHighest[0];
      preLowest[0] = sufLowest[0];
      if( nAvail > 0 )
      {
         tempHigh = inHigh[blockNext];
         tempLow = inLow[blockNext];
         highest = (tempHigh + tempLow) / 2.0;
         lowest = highest;
         preHighest[1] = highest;
         preLowest[1] = lowest;
         i = 1;
         TA_UNROLL(4)
         while( i < nAvail )
         {
            tempHigh = inHigh[blockNext + i];
            tempLow = inLow[blockNext + i];
            tempReal = (tempHigh + tempLow) / 2.0;
            if( tempReal > highest )
            {
               highest = tempReal;
            }
            if( tempReal < lowest )
            {
               lowest = tempReal;
            }
            preHighest[i + 1] = highest;
            preLowest[i + 1] = lowest;
            i += 1;
         }
      }
      m = 0;
      while( m <= nAvail )
      {
         highest = sufHighest[m];
         if( preHighest[m] > highest )
         {
            highest = preHighest[m];
         }
         lowest = sufLowest[m];
         if( preLowest[m] < lowest )
         {
            lowest = preLowest[m];
         }
         tempHigh = inHigh[today + m];
         tempLow = inLow[today + m];
         price = (tempHigh + tempLow) / 2.0;
         tempReal = highest - lowest;
         if( !TA_IS_ZERO_SCALED(tempReal, fabs(highest) + fabs(lowest)) )
         {
            ratio = (price - lowest) / tempReal;
         } else 
         {
            ratio = 0.5;
         }
         smoothed = fma(0.67, smoothed, 0.33 * 2.0 * (ratio - 0.5));
         if( smoothed > 0.99 )
         {
            smoothed = 0.999;
         }
         if( smoothed < -0.99 )
         {
            smoothed = -0.999;
         }
         fish = fma(0.5, log((1.0 + smoothed) / (1.0 - smoothed)), 0.5 * prevFish);
         if( today + m >= startIdx )
         {
            outFisher[outIdx] = fish;
            outTrigger[outIdx] = prevFish;
            outIdx = outIdx + 1;
         }
         prevFish = fish;
         m += 1;
      }
      today = today + nAvail + 1;
      blockStart = blockNext;
   }
   if( sufHighest != &local_sufHighest[0] ) { TA_Free( sufHighest ); sufHighest = &local_sufHighest[0]; }
   if( preHighest != &local_preHighest[0] ) { TA_Free( preHighest ); preHighest = &local_preHighest[0]; }
   if( sufLowest != &local_sufLowest[0] ) { TA_Free( sufLowest ); sufLowest = &local_sufLowest[0]; }
   if( preLowest != &local_preLowest[0] ) { TA_Free( preLowest ); preLowest = &local_preLowest[0]; }
   *outNBElement= outIdx;
   *outBegIdx= startIdx;
   return TA_SUCCESS;
}

TA_NOINLINE TA_LIB_API int TA_FISHER_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 10;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   /* The smoothed values of two starts differ by 0.67 times less each bar on
    * any input, 2.5 bars per e-fold, while the limiter treats both alike.
    * The 6 e-folds added to K are the gain from that difference to the
    * outputs, against the largest difference the seed causes: the
    * transform's slope at the limit, 1/(1 - 0.99^2), about 50, times
    * 0.67/(0.67 - 0.5) for its own 0.5 pole, plus a bar for the trigger.
    * Not a bound when the limiter moves one start alone: 0.999 is fed back,
    * and a channel position that stays above 0.986 keeps it there.
    */
   return optInTimePeriod - 1 + TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_FISHER,Fisher,(5 * (10 + 6) + 1) / 2,(5 * (19 + 6) + 1) / 2);
}
