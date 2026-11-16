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
TA_LIB_API TA_RetCode TA_TSI( int    startIdx,
                              int    endIdx,
                              const double inReal[],
                              int optInFirstPeriod,
                              int optInSecondPeriod,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outReal[] )
{
   double kFirst;
   double kSecond;
   double betaFirst;
   double betaSecond;
   double emaFirstNum;
   double emaFirstDen;
   double emaSecondNum;
   double emaSecondDen;
   double sumFirstNum;
   double sumFirstDen;
   double sumSecondNum;
   double sumSecondDen;
   double prevClose;
   double mom;
   double absMom;
   double tsiValue;
   int lookbackTotal;
   int lookbackFirst;
   int today;
   int outIdx;
   int nBar;
   int nSecond;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInFirstPeriod == TA_INTEGER_DEFAULT )
      optInFirstPeriod = 25;
   else if( (int)optInFirstPeriod < 2 || (int)optInFirstPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSecondPeriod == TA_INTEGER_DEFAULT )
      optInSecondPeriod = 13;
   else if( (int)optInSecondPeriod < 2 || (int)optInSecondPeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   lookbackTotal = TA_TSI_Lookback(optInFirstPeriod,optInSecondPeriod);
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
   /* Blau's double smoothing in one pass: the signed momentum and its
    * magnitude are carried through the same two EMA stages, then divided.
    *
    * Each stage seeds the way ema.c does -- a simple average of that stage's
    * first 'period' inputs -- so the result is bit-identical to TA_MOM(1)
    * followed by two TA_EMA calls on each chain. The stage boundary below is
    * the callee LOOKBACK, not (period-1), so that a warm
    * TA_SetUnstablePeriod(TA_FUNC_UNST_EMA) folds in: the second stage then
    * seeds from the values the first would have published. The seed sums
    * accumulate from 0.0 in production order and the recurrence is
    * k*x + beta*prev with ema.c's k and beta; do
    * not reorder or fuse them (0.0+x is not x for x=-0.0). That order IS the
    * bit-exactness contract against the composed reference.
    *
    * prevClose is carried in a scalar rather than re-read from inReal[t-1]
    * because outReal may alias inReal: the slot holding close[t-1] may already
    * hold an output written a bar earlier.
    */
   betaFirst = (double)(optInFirstPeriod - 1) / (double)(optInFirstPeriod + 1);
   kFirst = 1.0 - betaFirst;
   if( betaFirst < 0.5 )
   {
      betaFirst = 1.0 - kFirst;
   }
   betaSecond = (double)(optInSecondPeriod - 1) / (double)(optInSecondPeriod + 1);
   kSecond = 1.0 - betaSecond;
   if( betaSecond < 0.5 )
   {
      betaSecond = 1.0 - kSecond;
   }
   lookbackFirst = TA_EMA_Lookback(optInFirstPeriod);
   emaFirstNum = 0.0;
   emaFirstDen = 0.0;
   emaSecondNum = 0.0;
   emaSecondDen = 0.0;
   sumFirstNum = 0.0;
   sumFirstDen = 0.0;
   sumSecondNum = 0.0;
   sumSecondDen = 0.0;
   /* The first bar carrying a close-to-close change. */
   today = startIdx - lookbackTotal + 1;
   prevClose = inReal[today - 1];
   nBar = 0;
   /* Warm-up. Runs through startIdx inclusive: the last pass here completes
    * the second stage's seed, so it produces the first output.
    */
   while( today <= startIdx )
   {
      mom = inReal[today] - prevClose;
      prevClose = inReal[today];
      absMom = fabs(mom);
      /* Stage 1: the first EMA, over the raw momentum and its magnitude. */
      if( nBar < optInFirstPeriod )
      {
         sumFirstNum = sumFirstNum + mom;
         sumFirstDen = sumFirstDen + absMom;
         if( nBar == optInFirstPeriod - 1 )
         {
            emaFirstNum = sumFirstNum / optInFirstPeriod;
            emaFirstDen = sumFirstDen / optInFirstPeriod;
         }
      } else 
      {
         emaFirstNum = fma(betaFirst, emaFirstNum, kFirst * mom);
         emaFirstDen = fma(betaFirst, emaFirstDen, kFirst * absMom);
      }
      /* Stage 2: the second EMA, over what stage 1 publishes.
       *
       * The stage counter is compared BEFORE it is subtracted, never after.
       * Writing this as `nSecond = nBar - lookbackFirst; if( nSecond >= 0 )`
       * is correct in C, where the counters are signed, and broken everywhere
       * else: the Rust backend renders them as usize, so the subtraction
       * underflows for the first lookbackFirst bars -- a panic in a debug
       * build and a wrap in release. It would also be invisible to the
       * cross-language gate, which runs release servers at unstable period 0,
       * where the branch the wrap wrongly takes happens to be a no-op because
       * both accumulators are still 0.0. smi.c states the same rule.
       */
      if( nBar >= lookbackFirst )
      {
         nSecond = nBar - lookbackFirst;
         if( nSecond < optInSecondPeriod )
         {
            sumSecondNum = sumSecondNum + emaFirstNum;
            sumSecondDen = sumSecondDen + emaFirstDen;
            if( nSecond == optInSecondPeriod - 1 )
            {
               emaSecondNum = sumSecondNum / optInSecondPeriod;
               emaSecondDen = sumSecondDen / optInSecondPeriod;
            }
         } else 
         {
            emaSecondNum = fma(betaSecond, emaSecondNum, kSecond * emaFirstNum);
            emaSecondDen = fma(betaSecond, emaSecondDen, kSecond * emaFirstDen);
         }
      }
      nBar = nBar + 1;
      today = today + 1;
   }
   /* The denominator is an EMA of an EMA of |momentum|: every term is
    * non-negative and every weight positive, so it is zero only when every
    * change that reached it was exactly zero -- 0/0, since the numerator is
    * zero with it, reported as the neutral 0.0 by the CCI (#7) and IMI (#112)
    * convention. Tested exactly rather than against a fixed band: a price
    * change carries the quote unit, and TA_IS_ZERO zeroes the oscillator for
    * any instrument quoted below it (issue #253).
    */
   if( emaSecondDen > 0.0 )
   {
      tsiValue = 100.0 * emaSecondNum / emaSecondDen;
   } else 
   {
      tsiValue = 0.0;
   }
   outReal[0] = tsiValue;
   outIdx = 1;
   /* Stable zone. Both stages are a pure recursion from here on. */
   while( today <= endIdx )
   {
      mom = inReal[today] - prevClose;
      prevClose = inReal[today];
      absMom = fabs(mom);
      emaFirstNum = fma(betaFirst, emaFirstNum, kFirst * mom);
      emaFirstDen = fma(betaFirst, emaFirstDen, kFirst * absMom);
      emaSecondNum = fma(betaSecond, emaSecondNum, kSecond * emaFirstNum);
      emaSecondDen = fma(betaSecond, emaSecondDen, kSecond * emaFirstDen);
      if( emaSecondDen > 0.0 )
      {
         tsiValue = 100.0 * emaSecondNum / emaSecondDen;
      } else 
      {
         tsiValue = 0.0;
      }
      outReal[outIdx] = tsiValue;
      outIdx = outIdx + 1;
      today = today + 1;
   }
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_TSI_Lookback( int optInFirstPeriod, int optInSecondPeriod )
{
   if( (int)optInFirstPeriod == TA_INTEGER_DEFAULT )
      optInFirstPeriod = 25;
   else if( (int)optInFirstPeriod < 2 || (int)optInFirstPeriod > 100000 )
      return -1;
   if( (int)optInSecondPeriod == TA_INTEGER_DEFAULT )
      optInSecondPeriod = 13;
   else if( (int)optInSecondPeriod < 2 || (int)optInSecondPeriod > 100000 )
      return -1;
   /* One bar forms the first close-to-close change, then the two EMA warm-ups
    * the pipeline stacks on it. Each term is exactly the lookback of the
    * function it comes from, which is also what makes TSI inherit
    * TA_FUNC_UNST_EMA from its callee.
    */
   return 1 + TA_EMA_Lookback(optInFirstPeriod) + TA_EMA_Lookback(optInSecondPeriod);
}
