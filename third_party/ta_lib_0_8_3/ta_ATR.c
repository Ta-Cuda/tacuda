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
TA_LIB_API TA_RetCode TA_ATR( int    startIdx,
                              int    endIdx,
                              const double inHigh[],
                              const double inLow[],
                              const double inClose[],
                              int optInTimePeriod,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outReal[] )
{
   int i;
   int outIdx;
   int today;
   int lookbackTotal;
   int nbATR;
   double prevATR;
   double periodTotal;
   double wAlpha;
   double wBeta;
   double val2;
   double val3;
   double greatest;
   double tempCY;
   double tempLT;
   double tempHT;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
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

   /* Average True Range is the greatest of the following:
    *
    *  val1 = distance from today's high to today's low.
    *  val2 = distance from yesterday's close to today's high.
    *  val3 = distance from yesterday's close to today's low.
    *
    * These value are averaged for the specified period using
    * Wilder method. This method have an unstable period comparable
    * to and Exponential Moving Average (EMA).
    */
   *outBegIdx= 0;
   *outNBElement= 0;
   /* Adjust startIdx to account for the lookback period. */
   lookbackTotal = TA_ATR_Lookback(optInTimePeriod);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   /* wAlpha is derived FROM wBeta, never the reverse: only that order makes
    * wAlpha + wBeta exactly 1 (Sterbenz -- wBeta lands in [0.5, 1)), and it
    * measures closer to the exact recursion than the 1/period-first spelling
    * at nearly every period. The order is a gated contract, not a preference:
    * swapping it reddens the frozen-release comparison, and breaks the
    * bit-for-bit identity TA_RMA(TA_TRANGE(h,l,c),n) == TA_ATR(n).
    * The pair is exactly (1, 0) at period 1 -- hence no period-1 arm.
    */
   wBeta = (double)(optInTimePeriod - 1) / (double)optInTimePeriod;
   wAlpha = 1.0 - wBeta;
   /* The True Range of each bar is computed inline in a single
    * pass. No temporary buffer is needed.
    *
    * The arithmetic order below is the bit-exactness contract
    * (do not reorder):
    *  - True Range: start from high-low, then compare/replace
    *    with the two previous-close distances, in that order.
    *  - Seed: the first 'period' True Range values are summed,
    *    accumulated from 0.0 in input order, then divided by
    *    the period.
    *  - Wilder smoothing: ONE statement. Splitting it back
    *    unfuses the multiply-add and puts a second latency on
    *    the recurrence's dependency chain.
    *
    * In-place (outReal being one of the input arrays) is
    * supported: each output is written only after every input
    * read at or before its bar, and the output index is always
    * smaller than the bar index of any remaining read.
    */
   /* The first True Range needs the two price bars at
    * startIdx-lookbackTotal+1 (a previous close is consumed).
    */
   today = startIdx - lookbackTotal + 1;
   /* Seed the ATR with a simple average of the True Range
    * for the first 'period' bars.
    */
   periodTotal = 0.0;
   i = optInTimePeriod;
   while( i-- > 0 )
   {
      /* Find the greatest of the 3 values. */
      tempLT = inLow[today];
      tempHT = inHigh[today];
      tempCY = inClose[today - 1];
      greatest = tempHT - tempLT;
      /* val1 */
      val2 = fabs(tempCY - tempHT);
      if( val2 > greatest )
      {
         greatest = val2;
      }
      val3 = fabs(tempCY - tempLT);
      if( val3 > greatest )
      {
         greatest = val3;
      }
      periodTotal += greatest;
      today += 1;
   }
   prevATR = periodTotal / optInTimePeriod;
   /* Skip the unstable period. */
   i = lookbackTotal - optInTimePeriod;
   while( i != 0 )
   {
      /* Find the greatest of the 3 values. */
      tempLT = inLow[today];
      tempHT = inHigh[today];
      tempCY = inClose[today - 1];
      greatest = tempHT - tempLT;
      /* val1 */
      val2 = fabs(tempCY - tempHT);
      if( val2 > greatest )
      {
         greatest = val2;
      }
      val3 = fabs(tempCY - tempLT);
      if( val3 > greatest )
      {
         greatest = val3;
      }
      prevATR = fma(wBeta, prevATR, wAlpha * greatest);
      today += 1;
      i -= 1;
   }
   /* Now start to write the final ATR in the caller
    * provided outReal.
    */
   outIdx = 1;
   outReal[0] = prevATR;
   /* Now do the number of requested ATR. */
   nbATR = endIdx - startIdx + 1;
   while( --nbATR != 0 )
   {
      /* Find the greatest of the 3 values. */
      tempLT = inLow[today];
      tempHT = inHigh[today];
      tempCY = inClose[today - 1];
      greatest = tempHT - tempLT;
      /* val1 */
      val2 = fabs(tempCY - tempHT);
      if( val2 > greatest )
      {
         greatest = val2;
      }
      val3 = fabs(tempCY - tempLT);
      if( val3 > greatest )
      {
         greatest = val3;
      }
      prevATR = fma(wBeta, prevATR, wAlpha * greatest);
      outReal[outIdx++] = prevATR;
      today += 1;
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_NOINLINE TA_LIB_API int TA_ATR_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   /* The ATR lookback is the sum of:
    *    1 + (optInTimePeriod - 1)
    *
    * Where 1 is for the True Range, and
    * (optInTimePeriod-1) is for the simple
    * moving average.
    */
   return optInTimePeriod + TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_ATR,Atr,((optInTimePeriod > 1) ? (10 * (2 * optInTimePeriod - 1) + 1) / 2 : 0),((optInTimePeriod > 1) ? (19 * (2 * optInTimePeriod - 1) + 1) / 2 : 0));
}
