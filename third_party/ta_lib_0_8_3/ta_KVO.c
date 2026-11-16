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
TA_LIB_API TA_RetCode TA_KVO( int    startIdx,
                              int    endIdx,
                              const double inHigh[],
                              const double inLow[],
                              const double inClose[],
                              const double inVolume[],
                              int optInFastPeriod,
                              int optInSlowPeriod,
                              int optInSignalPeriod,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outKVO[],
                              double        outKVOSignal[] )
{
   double kFast;
   double kSlow;
   double kSignal;
   double betaFast;
   double betaSlow;
   double betaSignal;
   double hlc;
   double prevHlc;
   double dmToday;
   double prevDm;
   double emaFast;
   double emaSlow;
   double cm;
   double vf;
   double trend;
   double prevTrend;
   double kvoValue;
   double signalValue;
   int lookbackTotal;
   int today;
   int outIdx;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInFastPeriod == TA_INTEGER_DEFAULT )
      optInFastPeriod = 34;
   else if( (int)optInFastPeriod < 2 || (int)optInFastPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSlowPeriod == TA_INTEGER_DEFAULT )
      optInSlowPeriod = 55;
   else if( (int)optInSlowPeriod < 2 || (int)optInSlowPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSignalPeriod == TA_INTEGER_DEFAULT )
      optInSignalPeriod = 13;
   else if( (int)optInSignalPeriod < 2 || (int)optInSignalPeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !inClose )
      return TA_BAD_PARAM;
   if( !inVolume )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outKVO )
      return TA_BAD_PARAM;
   if( !outKVOSignal )
      return TA_BAD_PARAM;
   if( outKVO == outKVOSignal )
      return TA_BAD_PARAM;

   /* The C stream struct follows this order, and a bad one can stall every
    * streamed update without changing a value. prevHlc/prevDm and
    * emaFast/emaSlow are each stored as one 16-byte pair and must sit 16-byte
    * aligned in the struct; prevTrend keeps cm and signalValue apart so no
    * load spans two stores.
    */
   /* Stephen J. Klinger, "Identifying Trends With Volume Analysis", Technical
    * Analysis of Stocks & Commodities V15:12 (December 1997), with the 1997
    * factor |2*(dm/cm) - 1| (#484, form A).
    *
    *    trend = +1 if H+L+C rose, -1 if it fell, unchanged on a tie
    *    cm    = previous + current range on a trend change, else cm + range
    *    vf    = volume * |2*(dm/cm) - 1| * 100 * trend
    *
    * The committed corpus holds a single tie in H+L+C, and reading it any other
    * way moves KVO by up to 1e7 from the next bar on: the tie rule is gated by
    * a built series in test_kvo.c, not by the corpus.
    *
    * All three EMAs seed raw on the first volume force, not on an SMA as
    * TA_EMA does. Each k + beta pair must sum to exactly 1.0, as in ema.c.
    */
   lookbackTotal = TA_KVO_Lookback(optInFastPeriod,optInSlowPeriod,optInSignalPeriod);
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
   betaFast = (double)(optInFastPeriod - 1) / (double)(optInFastPeriod + 1);
   kFast = 1.0 - betaFast;
   if( betaFast < 0.5 )
   {
      betaFast = 1.0 - kFast;
   }
   betaSlow = (double)(optInSlowPeriod - 1) / (double)(optInSlowPeriod + 1);
   kSlow = 1.0 - betaSlow;
   if( betaSlow < 0.5 )
   {
      betaSlow = 1.0 - kSlow;
   }
   betaSignal = (double)(optInSignalPeriod - 1) / (double)(optInSignalPeriod + 1);
   kSignal = 1.0 - betaSignal;
   if( betaSignal < 0.5 )
   {
      betaSignal = 1.0 - kSignal;
   }
   /* The anchor: only its H+L+C and range are read, and the trend starts at +1. */
   today = startIdx - lookbackTotal;
   prevHlc = inHigh[today] + inLow[today] + inClose[today];
   prevDm = inHigh[today] - inLow[today];
   prevTrend = 1.0;
   today = today + 1;
   /* The seed bar: cm opens on both ranges whatever the trend, and both
    * averages start at this volume force, so KVO and the trigger are exactly 0.
    */
   hlc = inHigh[today] + inLow[today] + inClose[today];
   dmToday = inHigh[today] - inLow[today];
   if( hlc > prevHlc )
   {
      trend = 1.0;
   } else if( hlc < prevHlc )
   {
      trend = -1.0;
   } else 
   {
      trend = prevTrend;
   }
   cm = prevDm + dmToday;
   if( cm == 0.0 )
   {
      vf = 0.0;
   } else 
   {
      vf = inVolume[today] * fabs(2.0 * (dmToday / cm) - 1.0) * 100.0 * trend;
   }
   emaFast = vf;
   emaSlow = vf;
   kvoValue = emaFast - emaSlow;
   signalValue = kvoValue;
   prevHlc = hlc;
   prevDm = dmToday;
   prevTrend = trend;
   today = today + 1;
   while( today < startIdx )
   {
      hlc = inHigh[today] + inLow[today] + inClose[today];
      dmToday = inHigh[today] - inLow[today];
      if( hlc > prevHlc )
      {
         trend = 1.0;
      } else if( hlc < prevHlc )
      {
         trend = -1.0;
      } else 
      {
         trend = prevTrend;
      }
      /* A branch, not a select: LLVM turns a select between the two stored
       * ranges into a select of their address, which serializes each streamed
       * update behind the previous one on ARM64.
       */
      if( trend != prevTrend )
      {
         cm = prevDm + dmToday;
      } else 
      {
         cm = cm + dmToday;
      }
      if( cm == 0.0 )
      {
         vf = 0.0;
      } else 
      {
         vf = inVolume[today] * fabs(2.0 * (dmToday / cm) - 1.0) * 100.0 * trend;
      }
      emaFast = fma(betaFast, emaFast, kFast * vf);
      emaSlow = fma(betaSlow, emaSlow, kSlow * vf);
      kvoValue = emaFast - emaSlow;
      signalValue = fma(betaSignal, signalValue, kSignal * kvoValue);
      prevHlc = hlc;
      prevDm = dmToday;
      prevTrend = trend;
      today = today + 1;
   }
   outIdx = 0;
   while( today <= endIdx )
   {
      hlc = inHigh[today] + inLow[today] + inClose[today];
      dmToday = inHigh[today] - inLow[today];
      if( hlc > prevHlc )
      {
         trend = 1.0;
      } else if( hlc < prevHlc )
      {
         trend = -1.0;
      } else 
      {
         trend = prevTrend;
      }
      if( trend != prevTrend )
      {
         cm = prevDm + dmToday;
      } else 
      {
         cm = cm + dmToday;
      }
      if( cm == 0.0 )
      {
         vf = 0.0;
      } else 
      {
         vf = inVolume[today] * fabs(2.0 * (dmToday / cm) - 1.0) * 100.0 * trend;
      }
      emaFast = fma(betaFast, emaFast, kFast * vf);
      emaSlow = fma(betaSlow, emaSlow, kSlow * vf);
      kvoValue = emaFast - emaSlow;
      signalValue = fma(betaSignal, signalValue, kSignal * kvoValue);
      outKVO[outIdx] = kvoValue;
      outKVOSignal[outIdx] = signalValue;
      outIdx = outIdx + 1;
      prevHlc = hlc;
      prevDm = dmToday;
      prevTrend = trend;
      today = today + 1;
   }
   *outNBElement= outIdx;
   *outBegIdx= startIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_KVO_Lookback( int optInFastPeriod, int optInSlowPeriod, int optInSignalPeriod )
{
   int longestPeriod;
   if( (int)optInFastPeriod == TA_INTEGER_DEFAULT )
      optInFastPeriod = 34;
   else if( (int)optInFastPeriod < 2 || (int)optInFastPeriod > 100000 )
      return -1;
   if( (int)optInSlowPeriod == TA_INTEGER_DEFAULT )
      optInSlowPeriod = 55;
   else if( (int)optInSlowPeriod < 2 || (int)optInSlowPeriod > 100000 )
      return -1;
   if( (int)optInSignalPeriod == TA_INTEGER_DEFAULT )
      optInSignalPeriod = 13;
   else if( (int)optInSignalPeriod < 2 || (int)optInSignalPeriod > 100000 )
      return -1;
   /* The three EMAs seed on one bar, so one EMA lookback of the longest
    * covers all three. The 1 is the reference bar before it.
    */
   longestPeriod = optInFastPeriod;
   if( optInSlowPeriod > longestPeriod )
   {
      longestPeriod = optInSlowPeriod;
   }
   if( optInSignalPeriod > longestPeriod )
   {
      longestPeriod = optInSignalPeriod;
   }
   return 1 + TA_EMA_Lookback(longestPeriod) + TA_GLOBALS_UNSTABLE_OFFSET(TA_FUNC_UNST_EMA,kvo_auto_offset(optInFastPeriod, optInSlowPeriod, optInSignalPeriod));
}
