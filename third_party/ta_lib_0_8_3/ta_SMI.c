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
TA_LIB_API TA_RetCode TA_SMI( int    startIdx,
                              int    endIdx,
                              const double inHigh[],
                              const double inLow[],
                              const double inClose[],
                              int optInTimePeriod,
                              int optInFastPeriod,
                              int optInSlowPeriod,
                              int optInSignalPeriod,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outSMI[],
                              double        outSMISignal[] )
{
   double kSlow;
   double kFast;
   double kSignal;
   double betaSlow;
   double betaFast;
   double betaSignal;
   double highest;
   double lowest;
   double tmp;
   double emaSlowNum;
   double emaFastNum;
   double emaSlowDen;
   double emaFastDen;
   double sumSlowNum;
   double sumSlowDen;
   double sumFastNum;
   double sumFastDen;
   double sumSignal;
   double num;
   double den;
   double halfDen;
   double smiValue;
   double prevSignal;
   int lookbackTotal;
   int lookbackSlow;
   int lookbackFast;
   int today;
   int trailingIdx;
   int highestIdx;
   int lowestIdx;
   int i;
   int outIdx;
   int nBar;
   int nFast;
   int nSignal;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 13;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInFastPeriod == TA_INTEGER_DEFAULT )
      optInFastPeriod = 2;
   else if( (int)optInFastPeriod < 2 || (int)optInFastPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSlowPeriod == TA_INTEGER_DEFAULT )
      optInSlowPeriod = 25;
   else if( (int)optInSlowPeriod < 2 || (int)optInSlowPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSignalPeriod == TA_INTEGER_DEFAULT )
      optInSignalPeriod = 9;
   else if( (int)optInSignalPeriod < 2 || (int)optInSignalPeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !inClose )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outSMI )
      return TA_BAD_PARAM;
   if( !outSMISignal )
      return TA_BAD_PARAM;
   if( outSMI == outSMISignal )
      return TA_BAD_PARAM;

   /* Declared Num, Num, Den, Den: the stream state keeps this order, and with
    * each stage's pair adjacent gcc 13 packs the steps two by two and then
    * reads 16 bytes across two of its own stores, which cannot be forwarded.
    */
   lookbackTotal = TA_SMI_Lookback(optInTimePeriod,optInFastPeriod,optInSlowPeriod,optInSignalPeriod);
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
   /* Blau's pipeline in one pass. A composed form is not available: the
    * streaming producer model carries exactly one intermediate series
    * (streaming.rs:2076) and SMI has two, the numerator and the denominator,
    * which must be smoothed separately before they are divided.
    *
    * Each of the three stages seeds the way ema.c does -- a simple average of
    * that stage's first 'period' inputs -- so the result is bit-identical to
    * TA_MAX + TA_MIN followed by four TA_EMA calls. The stage boundaries below
    * are the callee LOOKBACKS, not (period-1), so that a warm
    * TA_SetUnstablePeriod(TA_FUNC_UNST_EMA) folds in: each stage then seeds
    * from the values its predecessor would have published, exactly as the
    * composed form does. The seed sums accumulate from 0.0 in production
    * order; do not reorder or fuse them (0.0+x is not x for x=-0.0).
    */
   betaSlow = (double)(optInSlowPeriod - 1) / (double)(optInSlowPeriod + 1);
   kSlow = 1.0 - betaSlow;
   if( betaSlow < 0.5 )
   {
      betaSlow = 1.0 - kSlow;
   }
   betaFast = (double)(optInFastPeriod - 1) / (double)(optInFastPeriod + 1);
   kFast = 1.0 - betaFast;
   if( betaFast < 0.5 )
   {
      betaFast = 1.0 - kFast;
   }
   betaSignal = (double)(optInSignalPeriod - 1) / (double)(optInSignalPeriod + 1);
   kSignal = 1.0 - betaSignal;
   if( betaSignal < 0.5 )
   {
      betaSignal = 1.0 - kSignal;
   }
   lookbackSlow = TA_EMA_Lookback(optInSlowPeriod);
   lookbackFast = TA_EMA_Lookback(optInFastPeriod);
   emaSlowNum = 0.0;
   emaSlowDen = 0.0;
   emaFastNum = 0.0;
   emaFastDen = 0.0;
   prevSignal = 0.0;
   smiValue = 0.0;
   sumSlowNum = 0.0;
   sumSlowDen = 0.0;
   sumFastNum = 0.0;
   sumFastDen = 0.0;
   sumSignal = 0.0;
   highest = 0.0;
   lowest = 0.0;
   highestIdx = -1;
   lowestIdx = -1;
   /* The first bar carrying a full high/low window. */
   trailingIdx = startIdx - lookbackTotal;
   today = trailingIdx + (optInTimePeriod - 1);
   nBar = 0;
   /* Warm-up. Runs through startIdx inclusive: the last pass here is the one
    * that completes the signal seed, so it produces the first output pair.
    */
   while( today <= startIdx )
   {
      /* Set the lowest low */
      tmp = inLow[today];
      if( lowestIdx < trailingIdx )
      {
         lowestIdx = trailingIdx;
         lowest = inLow[lowestIdx];
         i = lowestIdx;
         while( ++i <= today )
         {
            tmp = inLow[i];
            if( tmp < lowest )
            {
               lowestIdx = i;
               lowest = tmp;
            }
         }
      } else if( tmp <= lowest )
      {
         lowestIdx = today;
         lowest = tmp;
      }
      /* Set the highest high */
      tmp = inHigh[today];
      if( highestIdx < trailingIdx )
      {
         highestIdx = trailingIdx;
         highest = inHigh[highestIdx];
         i = highestIdx;
         while( ++i <= today )
         {
            tmp = inHigh[i];
            if( tmp > highest )
            {
               highestIdx = i;
               highest = tmp;
            }
         }
      } else if( tmp >= highest )
      {
         highestIdx = today;
         highest = tmp;
      }
      den = highest - lowest;
      num = inClose[today] - (highest + lowest) * 0.5;
      /* Stage 1: the slow EMA, over the raw momentum. */
      if( nBar < optInSlowPeriod )
      {
         sumSlowNum = sumSlowNum + num;
         sumSlowDen = sumSlowDen + den;
         if( nBar == optInSlowPeriod - 1 )
         {
            emaSlowNum = sumSlowNum / optInSlowPeriod;
            emaSlowDen = sumSlowDen / optInSlowPeriod;
         }
      } else 
      {
         emaSlowNum = fma(betaSlow, emaSlowNum, kSlow * num);
         emaSlowDen = fma(betaSlow, emaSlowDen, kSlow * den);
      }
      /* Stage 2: the fast EMA, over what stage 1 publishes.
       *
       * The stage counters are compared BEFORE they are subtracted, never
       * after. Writing this as `nFast = nBar - lookbackSlow; if( nFast >= 0 )`
       * is correct in C, where the counters are signed, and broken everywhere
       * else: the Rust backend renders them as `usize`, so the subtraction
       * underflows for the first lookbackSlow bars -- a panic in a debug build
       * and a wrap in release. It would also be invisible to the cross-language
       * gate, which runs release servers at unstable period 0, where the branch
       * the wrap wrongly takes happens to be a no-op because both accumulators
       * are still 0.0.
       */
      if( nBar >= lookbackSlow )
      {
         nFast = nBar - lookbackSlow;
         if( nFast < optInFastPeriod )
         {
            sumFastNum = sumFastNum + emaSlowNum;
            sumFastDen = sumFastDen + emaSlowDen;
            if( nFast == optInFastPeriod - 1 )
            {
               emaFastNum = sumFastNum / optInFastPeriod;
               emaFastDen = sumFastDen / optInFastPeriod;
            }
         } else 
         {
            emaFastNum = fma(betaFast, emaFastNum, kFast * emaSlowNum);
            emaFastDen = fma(betaFast, emaFastDen, kFast * emaSlowDen);
         }
      }
      /* Stage 3: the SMI line, then the signal EMA over it. */
      if( nBar >= lookbackSlow + lookbackFast )
      {
         nSignal = nBar - lookbackSlow - lookbackFast;
         halfDen = 0.5 * emaFastDen;
         if( halfDen > 0.0 )
         {
            smiValue = 100.0 * emaFastNum / halfDen;
         } else 
         {
            smiValue = 0.0;
         }
         if( nSignal < optInSignalPeriod )
         {
            sumSignal = sumSignal + smiValue;
            if( nSignal == optInSignalPeriod - 1 )
            {
               prevSignal = sumSignal / optInSignalPeriod;
            }
         } else 
         {
            prevSignal = fma(betaSignal, prevSignal, kSignal * smiValue);
         }
      }
      nBar = nBar + 1;
      trailingIdx = trailingIdx + 1;
      today = today + 1;
   }
   outSMI[0] = smiValue;
   outSMISignal[0] = prevSignal;
   outIdx = 1;
   /* Stable zone. Every stage is a pure recursion from here on. */
   while( today <= endIdx )
   {
      /* Set the lowest low */
      tmp = inLow[today];
      if( lowestIdx < trailingIdx )
      {
         lowestIdx = trailingIdx;
         lowest = inLow[lowestIdx];
         i = lowestIdx;
         while( ++i <= today )
         {
            tmp = inLow[i];
            if( tmp < lowest )
            {
               lowestIdx = i;
               lowest = tmp;
            }
         }
      } else if( tmp <= lowest )
      {
         lowestIdx = today;
         lowest = tmp;
      }
      /* Set the highest high */
      tmp = inHigh[today];
      if( highestIdx < trailingIdx )
      {
         highestIdx = trailingIdx;
         highest = inHigh[highestIdx];
         i = highestIdx;
         while( ++i <= today )
         {
            tmp = inHigh[i];
            if( tmp > highest )
            {
               highestIdx = i;
               highest = tmp;
            }
         }
      } else if( tmp >= highest )
      {
         highestIdx = today;
         highest = tmp;
      }
      den = highest - lowest;
      num = inClose[today] - (highest + lowest) * 0.5;
      emaSlowNum = fma(betaSlow, emaSlowNum, kSlow * num);
      emaSlowDen = fma(betaSlow, emaSlowDen, kSlow * den);
      emaFastNum = fma(betaFast, emaFastNum, kFast * emaSlowNum);
      emaFastDen = fma(betaFast, emaFastDen, kFast * emaSlowDen);
      /* The denominator is an EMA of an EMA of the high-low range: every term
       * is non-negative and every weight is positive, so it carries no
       * cancellation residue and is zero only when every range that reached it
       * was exactly zero -- 0/0, since a window of H == L bars makes num zero
       * too, reported as the neutral 0.0 by the CCI (#7) and IMI (#112)
       * convention. Test it exactly: the range carries the quote unit, so the
       * fixed TA_IS_ZERO band this used to be zeroed the oscillator for any
       * instrument quoted below it (issue #253). Issue #107's machine-flat
       * window is caught by the exact test as well, since the residue an
       * EMA leaves there is zero, not sub-epsilon.
       */
      halfDen = 0.5 * emaFastDen;
      if( halfDen > 0.0 )
      {
         smiValue = 100.0 * emaFastNum / halfDen;
      } else 
      {
         smiValue = 0.0;
      }
      prevSignal = fma(betaSignal, prevSignal, kSignal * smiValue);
      outSMI[outIdx] = smiValue;
      outSMISignal[outIdx] = prevSignal;
      outIdx = outIdx + 1;
      trailingIdx = trailingIdx + 1;
      today = today + 1;
   }
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_SMI_Lookback( int optInTimePeriod, int optInFastPeriod, int optInSlowPeriod, int optInSignalPeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 13;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   if( (int)optInFastPeriod == TA_INTEGER_DEFAULT )
      optInFastPeriod = 2;
   else if( (int)optInFastPeriod < 2 || (int)optInFastPeriod > 100000 )
      return -1;
   if( (int)optInSlowPeriod == TA_INTEGER_DEFAULT )
      optInSlowPeriod = 25;
   else if( (int)optInSlowPeriod < 2 || (int)optInSlowPeriod > 100000 )
      return -1;
   if( (int)optInSignalPeriod == TA_INTEGER_DEFAULT )
      optInSignalPeriod = 9;
   else if( (int)optInSignalPeriod < 2 || (int)optInSignalPeriod > 100000 )
      return -1;
   /* One high/low window, then the three EMA warm-ups the pipeline stacks on
    * top of it: slow smooths the raw momentum, fast smooths that, and signal
    * smooths the finished SMI line. Every term is exactly the lookback of the
    * function it comes from, so none of them is restated here -- which is also
    * what makes SMI inherit TA_FUNC_UNST_EMA from its callee.
    */
   return optInTimePeriod - 1 + TA_EMA_Lookback(optInSlowPeriod) + TA_EMA_Lookback(optInFastPeriod) + TA_EMA_Lookback(optInSignalPeriod);
}
