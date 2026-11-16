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
TA_LIB_API TA_RetCode TA_PSO( int    startIdx,
                              int    endIdx,
                              const double inHigh[],
                              const double inLow[],
                              const double inClose[],
                              int optInFastK_Period,
                              int optInEMAPeriod,
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
   double emaK;
   double emaBeta;
   double highest;
   double lowest;
   double tmp;
   double tempReal;
   double fastK;
   double nsk;
   double ema1;
   double ema2;
   double sum1;
   double sum2;
   int lookbackTotal;
   int lookbackEMA;
   int warmBars;
   int today;
   int i;
   int m;
   int blockStart;
   int blockNext;
   int nAvail;
   int outIdx;
   int nBar;
   int n2;
   int nOut;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInFastK_Period == TA_INTEGER_DEFAULT )
      optInFastK_Period = 8;
   else if( (int)optInFastK_Period < 1 || (int)optInFastK_Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInEMAPeriod == TA_INTEGER_DEFAULT )
      optInEMAPeriod = 5;
   else if( (int)optInEMAPeriod < 1 || (int)optInEMAPeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !inClose )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   lookbackTotal = TA_PSO_Lookback(optInFastK_Period,optInEMAPeriod);
   /* Move up the start index if there is not
    * enough initial data.
    */
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   /* Same values as pso_ALT1 below, which carries the formula. Only the
    * Fast-K window differs: a Van Herk / Gil-Werman block scan (WILLR's,
    * issue #147), so the cost per bar does not depend on the period or on
    * the shape of the input, where the cached extremum of pso_ALT1 rescans
    * its whole window on every bar of a flat or trending stretch. Every
    * scratch array holds copies, so the output may alias an input.
    */
   emaBeta = (double)(optInEMAPeriod - 1) / (double)(optInEMAPeriod + 1);
   emaK = 1.0 - emaBeta;
   if( emaBeta < 0.5 )
   {
      emaBeta = 1.0 - emaK;
   }
   lookbackEMA = TA_EMA_Lookback(optInEMAPeriod);
   warmBars = lookbackEMA + lookbackEMA;
   ema1 = 0.0;
   ema2 = 0.0;
   sum1 = 0.0;
   sum2 = 0.0;
   nBar = 0;
   today = startIdx - warmBars;
   blockStart = today - (optInFastK_Period - 1);
   outIdx = 0;
   if( optInFastK_Period < 1 ) return TA_INTERNAL_ERROR(491);
   if( (int)optInFastK_Period > (int)(sizeof(local_sufHighest)/sizeof(double)) )
   {
      sufHighest = TA_Malloc( sizeof(double)*optInFastK_Period );
      if( !sufHighest )
      {
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      sufHighest = &local_sufHighest[0];
   }
   if( optInFastK_Period < 1 ) return TA_INTERNAL_ERROR(492);
   if( (int)optInFastK_Period > (int)(sizeof(local_preHighest)/sizeof(double)) )
   {
      preHighest = TA_Malloc( sizeof(double)*optInFastK_Period );
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
   if( optInFastK_Period < 1 ) return TA_INTERNAL_ERROR(493);
   if( (int)optInFastK_Period > (int)(sizeof(local_sufLowest)/sizeof(double)) )
   {
      sufLowest = TA_Malloc( sizeof(double)*optInFastK_Period );
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
   if( optInFastK_Period < 1 ) return TA_INTERNAL_ERROR(494);
   if( (int)optInFastK_Period > (int)(sizeof(local_preLowest)/sizeof(double)) )
   {
      preLowest = TA_Malloc( sizeof(double)*optInFastK_Period );
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
      highest = inHigh[i];
      lowest = inLow[i];
      sufHighest[optInFastK_Period - 1] = highest;
      sufLowest[optInFastK_Period - 1] = lowest;
      TA_UNROLL(4)
      while( i > blockStart )
      {
         i -= 1;
         tmp = inHigh[i];
         if( tmp > highest )
         {
            highest = tmp;
         }
         tmp = inLow[i];
         if( tmp < lowest )
         {
            lowest = tmp;
         }
         sufHighest[i - blockStart] = highest;
         sufLowest[i - blockStart] = lowest;
      }
      /* Prefix extrema of the next block, clamped to what remains, stored
       * one slot up: slot 0 repeats the suffix so that bar 'today', whose
       * window is the block itself, runs the same combine as the others.
       */
      blockNext = blockStart + optInFastK_Period;
      nAvail = endIdx + 1 - blockNext;
      if( nAvail > optInFastK_Period - 1 )
      {
         nAvail = optInFastK_Period - 1;
      }
      preHighest[0] = sufHighest[0];
      preLowest[0] = sufLowest[0];
      if( nAvail > 0 )
      {
         i = 1;
         highest = inHigh[blockNext];
         lowest = inLow[blockNext];
         preHighest[1] = highest;
         preLowest[1] = lowest;
         TA_UNROLL(4)
         while( i < nAvail )
         {
            tmp = inHigh[blockNext + i];
            if( tmp > highest )
            {
               highest = tmp;
            }
            tmp = inLow[blockNext + i];
            if( tmp < lowest )
            {
               lowest = tmp;
            }
            preHighest[i + 1] = highest;
            preLowest[i + 1] = lowest;
            i += 1;
         }
      }
      /* The squash runs as a pass of its own over what the block wrote:
       * a call inside the loop above it would have every carried value
       * saved and restored around it on each bar.
       */
      nOut = 0;
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
         tempReal = highest - lowest;
         if( !TA_IS_ZERO_SCALED(tempReal, fabs(highest) + fabs(lowest)) )
         {
            fastK = (inClose[today + m] - lowest) / tempReal * 100.0;
         } else 
         {
            fastK = 50.0;
         }
         nsk = 0.1 * (fastK - 50.0);
         if( nBar > warmBars )
         {
            ema1 = fma(emaBeta, ema1, emaK * nsk);
            ema2 = fma(emaBeta, ema2, emaK * ema1);
            outReal[outIdx + nOut] = 0.5 * ema2;
            nOut = nOut + 1;
         } else 
         {
            /* The two seeds, staged as pso_ALT1 stages them. */
            if( nBar < optInEMAPeriod )
            {
               sum1 = sum1 + nsk;
               if( nBar == optInEMAPeriod - 1 )
               {
                  ema1 = sum1 / optInEMAPeriod;
               }
            } else 
            {
               ema1 = fma(emaBeta, ema1, emaK * nsk);
            }
            if( nBar >= lookbackEMA )
            {
               n2 = nBar - lookbackEMA;
               if( n2 < optInEMAPeriod )
               {
                  sum2 = sum2 + ema1;
                  if( n2 == optInEMAPeriod - 1 )
                  {
                     ema2 = sum2 / optInEMAPeriod;
                  }
               } else 
               {
                  ema2 = fma(emaBeta, ema2, emaK * ema1);
               }
            }
            if( nBar == warmBars )
            {
               outReal[outIdx + nOut] = 0.5 * ema2;
               nOut = nOut + 1;
            }
         }
         nBar = nBar + 1;
         m += 1;
      }
      i = 0;
      while( i < nOut )
      {
         outReal[outIdx] = tanh(outReal[outIdx]);
         outIdx = outIdx + 1;
         i += 1;
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

TA_LIB_API int TA_PSO_Lookback( int optInFastK_Period, int optInEMAPeriod )
{
   if( (int)optInFastK_Period == TA_INTEGER_DEFAULT )
      optInFastK_Period = 8;
   else if( (int)optInFastK_Period < 1 || (int)optInFastK_Period > 100000 )
      return -1;
   if( (int)optInEMAPeriod == TA_INTEGER_DEFAULT )
      optInEMAPeriod = 5;
   else if( (int)optInEMAPeriod < 1 || (int)optInEMAPeriod > 100000 )
      return -1;
   /* One Fast-K window, then the two EMA warm-ups the author stacks on top of
    * it: the first smooths the normalised Fast-K, the second smooths the
    * first. Both terms are exactly the lookback of the function they come
    * from, so neither is restated here -- which is also what makes PSO
    * inherit TA_FUNC_UNST_EMA from its callee rather than take an id of its
    * own, and what carries the Auto warm-up levels of #492 through both
    * passes without this file knowing their rule.
    */
   return optInFastK_Period - 1 + TA_EMA_Lookback(optInEMAPeriod) + TA_EMA_Lookback(optInEMAPeriod);
}
