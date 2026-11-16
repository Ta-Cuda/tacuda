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
TA_LIB_API TA_RetCode TA_DEMA( int    startIdx,
                               int    endIdx,
                               const double inReal[],
                               int optInTimePeriod,
                               int          *outBegIdx,
                               int          *outNBElement,
                               double        outReal[] )
{
   double prevEMA1;
   double prevEMA2;
   double tempReal;
   double optInK_1;
   double emaBeta;
   int i;
   int today;
   int outIdx;
   int lookbackEMA;
   int lookbackTotal;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 30;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* For an explanation of this function, please read
    *
    * Stocks & Commodities V. 12:1 (11-19):
    *   Smoothing Data With Faster Moving Averages
    * Stocks & Commodities V. 12:2 (72-80):
    *   Smoothing Data With Less Lag
    *
    * Both magazine articles written by Patrick G. Mulloy
    *
    * Essentially, a DEMA of time serie 't' is:
    *   EMA2 = EMA(EMA(t,period),period)
    *   DEMA = 2*EMA(t,period)- EMA2
    *
    * DEMA offers a moving average with less lags then the
    * traditional EMA.
    *
    * Do not confuse a DEMA with the EMA2. Both are called
    * "Double EMA" in the litterature, but EMA2 is a simple
    * EMA of an EMA, while DEMA is a compostie of a single
    * EMA with EMA2.
    *
    * TEMA is very similar (and from the same author).
    */
   /* Will change only on success. */
   *outNBElement= 0;
   *outBegIdx= 0;
   /* Adjust startIdx to account for the lookback period. */
   lookbackEMA = TA_EMA_Lookback(optInTimePeriod);
   lookbackTotal = lookbackEMA * 2;
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   /* No smoothing at period of 1: the output is a copy of the input
    * (same convention as TA_MA for every MAType). Explicit and separate
    * from TA_EMA's own copy because the two EMA below are inlined here,
    * not delegated -- at period 1 they run at a k of 1.0 and a beta of
    * 0.0, which does not keep the sign of a -0.0 input. The unstable
    * period still delays the first output, and at twice EMA's rate: TA_MA
    * reports lookback 0 at period 1, so the two disagree on alignment when
    * it is non-zero.
    */
   if( optInTimePeriod == 1 )
   {
      *outBegIdx= startIdx;
      outIdx = 0;
      today = startIdx;
      while( today <= endIdx )
      {
         outReal[outIdx++] = inReal[today++];
      }
      *outNBElement= outIdx;
      return TA_SUCCESS;
   }
   /* Both EMA are computed in a single lockstep pass: each new
    * EMA1 value is immediately fed into EMA2. No temporary
    * buffers are needed.
    *
    * The arithmetic order below is the bit-exactness contract
    * (do not reorder or fuse operations):
    *  - EMA recursion: k*x + beta*prev, with ema.c's k and beta.
    *  - Each EMA is seeded with the sum of its first 'period'
    *    inputs, accumulated from 0.0 in input order (0.0+x is
    *    not x for x=-0.0), divided by the period.
    *
    * In-place (inReal == outReal) is supported: outReal[outIdx]
    * is written only after inReal[startIdx+outIdx] was read.
    */
   emaBeta = (double)(optInTimePeriod - 1) / (double)(optInTimePeriod + 1);
   optInK_1 = 1.0 - emaBeta;
   if( emaBeta < 0.5 )
   {
      emaBeta = 1.0 - optInK_1;
   }
   /* Seed EMA1 with a simple average of the first
    * 'period' price bars.
    */
   today = startIdx - lookbackTotal;
   i = optInTimePeriod;
   tempReal = 0.0;
   while( i-- > 0 )
   {
      tempReal += inReal[today++];
   }
   prevEMA1 = tempReal / optInTimePeriod;
   /* Advance EMA1 alone through its unstable period, up to
    * the bar where EMA2 seeding begins.
    */
   while( today <= startIdx - lookbackEMA )
   {
      prevEMA1 = fma(emaBeta, prevEMA1, optInK_1 * inReal[today++]);
   }
   /* Seed EMA2 with a simple average of the first 'period'
    * EMA1 values, accumulated as EMA1 produces them.
    */
   tempReal = 0.0;
   tempReal += prevEMA1;
   i = optInTimePeriod - 1;
   while( i-- > 0 )
   {
      prevEMA1 = fma(emaBeta, prevEMA1, optInK_1 * inReal[today++]);
      tempReal += prevEMA1;
   }
   prevEMA2 = tempReal / optInTimePeriod;
   /* Advance both EMA in lockstep through the unstable period
    * of EMA2, up to the first output bar.
    */
   while( today <= startIdx )
   {
      prevEMA1 = fma(emaBeta, prevEMA1, optInK_1 * inReal[today++]);
      prevEMA2 = fma(emaBeta, prevEMA2, optInK_1 * prevEMA1);
   }
   /* Stable zone: keep advancing both EMA in lockstep and
    * write the DEMA into the output.
    */
   outReal[0] = 2.0 * prevEMA1 - prevEMA2;
   outIdx = 1;
   while( today <= endIdx )
   {
      prevEMA1 = fma(emaBeta, prevEMA1, optInK_1 * inReal[today++]);
      prevEMA2 = fma(emaBeta, prevEMA2, optInK_1 * prevEMA1);
      outReal[outIdx++] = 2.0 * prevEMA1 - prevEMA2;
   }
   /* Succeed. Indicate where the output starts relative to
    * the caller input.
    */
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_DEMA_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 30;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   /* Get lookback for one EMA.
    * Multiply by two (because double smoothing).
    */
   return TA_EMA_Lookback(optInTimePeriod) * 2;
}
