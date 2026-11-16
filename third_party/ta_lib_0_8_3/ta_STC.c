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
TA_LIB_API TA_RetCode TA_STC( int    startIdx,
                              int    endIdx,
                              const double inReal[],
                              int optInFastPeriod,
                              int optInSlowPeriod,
                              int optInCyclePeriod,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outReal[] )
{
   double local_lineRing[30];
   double *lineRing = &local_lineRing[0];
   int lineRing_Idx;
   int maxIdx_lineRing;
   double local_lineSufHi[30];
   double *lineSufHi = &local_lineSufHi[0];
   double local_lineSufLo[30];
   double *lineSufLo = &local_lineSufLo[0];
   double local_pfRing[30];
   double *pfRing = &local_pfRing[0];
   int pfRing_Idx;
   int maxIdx_pfRing;
   double local_pfSufHi[30];
   double *pfSufHi = &local_pfSufHi[0];
   double local_pfSufLo[30];
   double *pfSufLo = &local_pfSufLo[0];
   double prevFast;
   double prevSlow;
   double fastK;
   double slowK;
   double tempReal;
   double lineValue;
   double fastBeta;
   double slowBeta;
   double lowest;
   double highest;
   double range;
   double frac1;
   double frac2;
   double pf;
   double pff;
   double lineHi;
   double lineLo;
   double pfHi;
   double pfLo;
   double sufHi;
   double sufLo;
   int i;
   int today;
   int fastToday;
   int lineStart;
   int outIdx;
   int tempInteger;
   int lookbackTotal;
   int lookbackSlow;
   int lastIdx;
   int nLine;
   int nPF;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInFastPeriod == TA_INTEGER_DEFAULT )
      optInFastPeriod = 23;
   else if( (int)optInFastPeriod < 2 || (int)optInFastPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSlowPeriod == TA_INTEGER_DEFAULT )
      optInSlowPeriod = 50;
   else if( (int)optInSlowPeriod < 2 || (int)optInSlowPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInCyclePeriod == TA_INTEGER_DEFAULT )
      optInCyclePeriod = 10;
   else if( (int)optInCyclePeriod < 2 || (int)optInCyclePeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   if( optInSlowPeriod < optInFastPeriod )
   {
      tempInteger = optInSlowPeriod;
      optInSlowPeriod = optInFastPeriod;
      optInFastPeriod = tempInteger;
   }
   lookbackTotal = TA_STC_Lookback(optInFastPeriod,optInSlowPeriod,optInCyclePeriod);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   if( startIdx > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   *outBegIdx= startIdx;
   fastBeta = (double)(optInFastPeriod - 1) / (double)(optInFastPeriod + 1);
   fastK = 1.0 - fastBeta;
   if( fastBeta < 0.5 )
   {
      fastBeta = 1.0 - fastK;
   }
   slowBeta = (double)(optInSlowPeriod - 1) / (double)(optInSlowPeriod + 1);
   slowK = 1.0 - slowBeta;
   if( slowBeta < 0.5 )
   {
      slowBeta = 1.0 - slowK;
   }
   /* Rolling extrema, van Herk / Gil-Werman: the window ending in slot j is
    * the current block's prefix extremum joined with the previous block's
    * suffix extremum from slot j+1. The extrema are exact, so the output must
    * stay bit-identical to a full rescan of each window.
    */
   if( optInCyclePeriod < 1 ) return TA_INTERNAL_ERROR(462);
   if( (int)optInCyclePeriod > (int)(sizeof(local_lineRing)/sizeof(double)) )
   {
      lineRing = TA_Malloc( sizeof(double)*optInCyclePeriod );
      if( !lineRing )
      {
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      lineRing = &local_lineRing[0];
   }
   maxIdx_lineRing = (optInCyclePeriod-1);
   lineRing_Idx = 0;
   if( optInCyclePeriod < 1 ) return TA_INTERNAL_ERROR(463);
   if( (int)optInCyclePeriod > (int)(sizeof(local_lineSufHi)/sizeof(double)) )
   {
      lineSufHi = TA_Malloc( sizeof(double)*optInCyclePeriod );
      if( !lineSufHi )
      {
         if( lineRing != &local_lineRing[0] ) TA_Free( lineRing );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      lineSufHi = &local_lineSufHi[0];
   }
   if( optInCyclePeriod < 1 ) return TA_INTERNAL_ERROR(464);
   if( (int)optInCyclePeriod > (int)(sizeof(local_lineSufLo)/sizeof(double)) )
   {
      lineSufLo = TA_Malloc( sizeof(double)*optInCyclePeriod );
      if( !lineSufLo )
      {
         if( lineRing != &local_lineRing[0] ) TA_Free( lineRing );
         if( lineSufHi != &local_lineSufHi[0] ) TA_Free( lineSufHi );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      lineSufLo = &local_lineSufLo[0];
   }
   if( optInCyclePeriod < 1 ) return TA_INTERNAL_ERROR(465);
   if( (int)optInCyclePeriod > (int)(sizeof(local_pfRing)/sizeof(double)) )
   {
      pfRing = TA_Malloc( sizeof(double)*optInCyclePeriod );
      if( !pfRing )
      {
         if( lineRing != &local_lineRing[0] ) TA_Free( lineRing );
         if( lineSufHi != &local_lineSufHi[0] ) TA_Free( lineSufHi );
         if( lineSufLo != &local_lineSufLo[0] ) TA_Free( lineSufLo );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      pfRing = &local_pfRing[0];
   }
   maxIdx_pfRing = (optInCyclePeriod-1);
   pfRing_Idx = 0;
   if( optInCyclePeriod < 1 ) return TA_INTERNAL_ERROR(466);
   if( (int)optInCyclePeriod > (int)(sizeof(local_pfSufHi)/sizeof(double)) )
   {
      pfSufHi = TA_Malloc( sizeof(double)*optInCyclePeriod );
      if( !pfSufHi )
      {
         if( lineRing != &local_lineRing[0] ) TA_Free( lineRing );
         if( lineSufHi != &local_lineSufHi[0] ) TA_Free( lineSufHi );
         if( lineSufLo != &local_lineSufLo[0] ) TA_Free( lineSufLo );
         if( pfRing != &local_pfRing[0] ) TA_Free( pfRing );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      pfSufHi = &local_pfSufHi[0];
   }
   if( optInCyclePeriod < 1 ) return TA_INTERNAL_ERROR(467);
   if( (int)optInCyclePeriod > (int)(sizeof(local_pfSufLo)/sizeof(double)) )
   {
      pfSufLo = TA_Malloc( sizeof(double)*optInCyclePeriod );
      if( !pfSufLo )
      {
         if( lineRing != &local_lineRing[0] ) TA_Free( lineRing );
         if( lineSufHi != &local_lineSufHi[0] ) TA_Free( lineSufHi );
         if( lineSufLo != &local_lineSufLo[0] ) TA_Free( lineSufLo );
         if( pfRing != &local_pfRing[0] ) TA_Free( pfRing );
         if( pfSufHi != &local_pfSufHi[0] ) TA_Free( pfSufHi );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      pfSufLo = &local_pfSufLo[0];
   }
   lastIdx = optInCyclePeriod - 1;
   /* The line is TA_MACD's: each SMA seed placed by its own EMA lookback, then
    * both EMAs advanced to lineStart, so that from lineStart on it is
    * TA_EMA(fast) - TA_EMA(slow) bit for bit. That placement reads out of
    * bounds or skips bars unless ema_lookback(n) - n never decreases as n
    * grows. The chain is fed from
    * lineStart, not from the EMA seed: TA_FUNC_UNST_EMA then reaches only the
    * line, while TA_FUNC_UNST_STC moves the whole chain back and so warms the
    * line and both smoothers.
    */
   lookbackSlow = TA_EMA_Lookback(optInSlowPeriod);
   lineStart = startIdx - (lookbackTotal - lookbackSlow);
   today = startIdx - lookbackTotal;
   tempReal = 0.0;
   i = optInSlowPeriod;
   while( i-- > 0 )
   {
      tempReal += inReal[today++];
   }
   prevSlow = tempReal / optInSlowPeriod;
   fastToday = startIdx - lookbackTotal + (lookbackSlow - TA_EMA_Lookback(optInFastPeriod));
   prevFast = 0.0;
   i = optInFastPeriod;
   while( i-- > 0 )
   {
      prevFast += inReal[fastToday++];
   }
   prevFast = prevFast / optInFastPeriod;
   while( today < fastToday )
   {
      tempReal = inReal[today++];
      prevSlow = fma(slowBeta, prevSlow, slowK * tempReal);
   }
   while( today <= lineStart )
   {
      tempReal = inReal[today++];
      prevFast = fma(fastBeta, prevFast, fastK * tempReal);
      prevSlow = fma(slowBeta, prevSlow, slowK * tempReal);
   }
   /* A zero range holds the previous fraction (0.0 before any), and the test
    * is exact: in a sustained trend PF saturates at 100 and the second
    * range reaches exactly 0 while the output must stay at 100.
    */
   frac1 = 0.0;
   frac2 = 0.0;
   pf = 0.0;
   pff = 0.0;
   pfHi = 0.0;
   pfLo = 0.0;
   nPF = 0;
   lineValue = prevFast - prevSlow;
   lineRing[lineRing_Idx] = lineValue;
   lineRing_Idx++;
   if( lineRing_Idx > maxIdx_lineRing ) lineRing_Idx = 0;
   lineHi = lineValue;
   lineLo = lineValue;
   nLine = 1;
   /* Warm-up, through startIdx inclusive. Each stage starts once its
    * window is full, and each smoother is seeded on its first input.
    */
   while( today <= startIdx )
   {
      tempReal = inReal[today];
      prevFast = fma(fastBeta, prevFast, fastK * tempReal);
      prevSlow = fma(slowBeta, prevSlow, slowK * tempReal);
      lineValue = prevFast - prevSlow;
      nLine = nLine + 1;
      lineRing[lineRing_Idx] = lineValue;
      if( lineRing_Idx == 0 )
      {
         lineHi = lineValue;
         lineLo = lineValue;
      } else 
      {
         if( lineValue > lineHi )
         {
            lineHi = lineValue;
         }
         if( lineValue < lineLo )
         {
            lineLo = lineValue;
         }
      }
      highest = lineHi;
      lowest = lineLo;
      if( nLine >= optInCyclePeriod && lineRing_Idx < lastIdx )
      {
         tempReal = lineSufHi[lineRing_Idx + 1];
         if( tempReal > highest )
         {
            highest = tempReal;
         }
         tempReal = lineSufLo[lineRing_Idx + 1];
         if( tempReal < lowest )
         {
            lowest = tempReal;
         }
      }
      lineRing_Idx++;
      if( lineRing_Idx > maxIdx_lineRing ) lineRing_Idx = 0;
      if( lineRing_Idx == 0 )
      {
         sufHi = lineRing[lastIdx];
         sufLo = sufHi;
         lineSufHi[lastIdx] = sufHi;
         lineSufLo[lastIdx] = sufLo;
         i = lastIdx;
         while( i > 0 )
         {
            i -= 1;
            tempReal = lineRing[i];
            if( tempReal > sufHi )
            {
               sufHi = tempReal;
            }
            if( tempReal < sufLo )
            {
               sufLo = tempReal;
            }
            lineSufHi[i] = sufHi;
            lineSufLo[i] = sufLo;
         }
      }
      if( nLine >= optInCyclePeriod )
      {
         range = highest - lowest;
         if( range > 0.0 )
         {
            frac1 = (lineValue - lowest) / range * 100.0;
         }
         if( nPF == 0 )
         {
            pf = frac1;
         } else 
         {
            pf = fma(0.5, frac1 - pf, pf);
         }
         nPF = nPF + 1;
         pfRing[pfRing_Idx] = pf;
         if( pfRing_Idx == 0 )
         {
            pfHi = pf;
            pfLo = pf;
         } else 
         {
            if( pf > pfHi )
            {
               pfHi = pf;
            }
            if( pf < pfLo )
            {
               pfLo = pf;
            }
         }
         highest = pfHi;
         lowest = pfLo;
         if( nPF >= optInCyclePeriod && pfRing_Idx < lastIdx )
         {
            tempReal = pfSufHi[pfRing_Idx + 1];
            if( tempReal > highest )
            {
               highest = tempReal;
            }
            tempReal = pfSufLo[pfRing_Idx + 1];
            if( tempReal < lowest )
            {
               lowest = tempReal;
            }
         }
         pfRing_Idx++;
         if( pfRing_Idx > maxIdx_pfRing ) pfRing_Idx = 0;
         if( pfRing_Idx == 0 )
         {
            sufHi = pfRing[lastIdx];
            sufLo = sufHi;
            pfSufHi[lastIdx] = sufHi;
            pfSufLo[lastIdx] = sufLo;
            i = lastIdx;
            while( i > 0 )
            {
               i -= 1;
               tempReal = pfRing[i];
               if( tempReal > sufHi )
               {
                  sufHi = tempReal;
               }
               if( tempReal < sufLo )
               {
                  sufLo = tempReal;
               }
               pfSufHi[i] = sufHi;
               pfSufLo[i] = sufLo;
            }
         }
         if( nPF >= optInCyclePeriod )
         {
            range = highest - lowest;
            if( range > 0.0 )
            {
               frac2 = (pf - lowest) / range * 100.0;
            }
            if( nPF == optInCyclePeriod )
            {
               pff = frac2;
            } else 
            {
               pff = fma(0.5, frac2 - pff, pff);
            }
         }
      }
      today = today + 1;
   }
   outReal[0] = pff;
   outIdx = 1;
   while( today <= endIdx )
   {
      tempReal = inReal[today];
      prevFast = fma(fastBeta, prevFast, fastK * tempReal);
      prevSlow = fma(slowBeta, prevSlow, slowK * tempReal);
      lineValue = prevFast - prevSlow;
      lineRing[lineRing_Idx] = lineValue;
      if( lineRing_Idx == 0 )
      {
         lineHi = lineValue;
         lineLo = lineValue;
      } else 
      {
         if( lineValue > lineHi )
         {
            lineHi = lineValue;
         }
         if( lineValue < lineLo )
         {
            lineLo = lineValue;
         }
      }
      highest = lineHi;
      lowest = lineLo;
      if( lineRing_Idx < lastIdx )
      {
         tempReal = lineSufHi[lineRing_Idx + 1];
         if( tempReal > highest )
         {
            highest = tempReal;
         }
         tempReal = lineSufLo[lineRing_Idx + 1];
         if( tempReal < lowest )
         {
            lowest = tempReal;
         }
      }
      lineRing_Idx++;
      if( lineRing_Idx > maxIdx_lineRing ) lineRing_Idx = 0;
      if( lineRing_Idx == 0 )
      {
         sufHi = lineRing[lastIdx];
         sufLo = sufHi;
         lineSufHi[lastIdx] = sufHi;
         lineSufLo[lastIdx] = sufLo;
         i = lastIdx;
         while( i > 0 )
         {
            i -= 1;
            tempReal = lineRing[i];
            if( tempReal > sufHi )
            {
               sufHi = tempReal;
            }
            if( tempReal < sufLo )
            {
               sufLo = tempReal;
            }
            lineSufHi[i] = sufHi;
            lineSufLo[i] = sufLo;
         }
      }
      range = highest - lowest;
      if( range > 0.0 )
      {
         frac1 = (lineValue - lowest) / range * 100.0;
      }
      pf = fma(0.5, frac1 - pf, pf);
      pfRing[pfRing_Idx] = pf;
      if( pfRing_Idx == 0 )
      {
         pfHi = pf;
         pfLo = pf;
      } else 
      {
         if( pf > pfHi )
         {
            pfHi = pf;
         }
         if( pf < pfLo )
         {
            pfLo = pf;
         }
      }
      highest = pfHi;
      lowest = pfLo;
      if( pfRing_Idx < lastIdx )
      {
         tempReal = pfSufHi[pfRing_Idx + 1];
         if( tempReal > highest )
         {
            highest = tempReal;
         }
         tempReal = pfSufLo[pfRing_Idx + 1];
         if( tempReal < lowest )
         {
            lowest = tempReal;
         }
      }
      pfRing_Idx++;
      if( pfRing_Idx > maxIdx_pfRing ) pfRing_Idx = 0;
      if( pfRing_Idx == 0 )
      {
         sufHi = pfRing[lastIdx];
         sufLo = sufHi;
         pfSufHi[lastIdx] = sufHi;
         pfSufLo[lastIdx] = sufLo;
         i = lastIdx;
         while( i > 0 )
         {
            i -= 1;
            tempReal = pfRing[i];
            if( tempReal > sufHi )
            {
               sufHi = tempReal;
            }
            if( tempReal < sufLo )
            {
               sufLo = tempReal;
            }
            pfSufHi[i] = sufHi;
            pfSufLo[i] = sufLo;
         }
      }
      range = highest - lowest;
      if( range > 0.0 )
      {
         frac2 = (pf - lowest) / range * 100.0;
      }
      pff = fma(0.5, frac2 - pff, pff);
      outReal[outIdx++] = pff;
      today = today + 1;
   }
   if( lineRing != &local_lineRing[0] ) { TA_Free( lineRing ); lineRing = &local_lineRing[0]; }
   if( lineSufHi != &local_lineSufHi[0] ) { TA_Free( lineSufHi ); lineSufHi = &local_lineSufHi[0]; }
   if( lineSufLo != &local_lineSufLo[0] ) { TA_Free( lineSufLo ); lineSufLo = &local_lineSufLo[0]; }
   if( pfRing != &local_pfRing[0] ) { TA_Free( pfRing ); pfRing = &local_pfRing[0]; }
   if( pfSufHi != &local_pfSufHi[0] ) { TA_Free( pfSufHi ); pfSufHi = &local_pfSufHi[0]; }
   if( pfSufLo != &local_pfSufLo[0] ) { TA_Free( pfSufLo ); pfSufLo = &local_pfSufLo[0]; }
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_NOINLINE TA_LIB_API int TA_STC_Lookback( int optInFastPeriod, int optInSlowPeriod, int optInCyclePeriod )
{
   int tempInteger;
   if( (int)optInFastPeriod == TA_INTEGER_DEFAULT )
      optInFastPeriod = 23;
   else if( (int)optInFastPeriod < 2 || (int)optInFastPeriod > 100000 )
      return -1;
   if( (int)optInSlowPeriod == TA_INTEGER_DEFAULT )
      optInSlowPeriod = 50;
   else if( (int)optInSlowPeriod < 2 || (int)optInSlowPeriod > 100000 )
      return -1;
   if( (int)optInCyclePeriod == TA_INTEGER_DEFAULT )
      optInCyclePeriod = 10;
   else if( (int)optInCyclePeriod < 2 || (int)optInCyclePeriod > 100000 )
      return -1;
   if( optInSlowPeriod < optInFastPeriod )
   {
      tempInteger = optInSlowPeriod;
      optInSlowPeriod = optInFastPeriod;
      optInFastPeriod = tempInteger;
   }
   /* The MACD line's own lookback, which is what inherits TA_FUNC_UNST_EMA,
    * then one window per stochastic stage. The two 0.5 smoothers seed on
    * their first input, so they add only the unstable period.
    */
   return TA_EMA_Lookback(optInSlowPeriod) + 2 * (optInCyclePeriod - 1) + TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_STC,Stc,(5 * 10 + 1) / 2 + 3 * (optInSlowPeriod + 1),(5 * 19 + 1) / 2 + 3 * (optInSlowPeriod + 1));
}
