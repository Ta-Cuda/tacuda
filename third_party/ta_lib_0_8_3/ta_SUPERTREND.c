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
TA_LIB_API TA_RetCode TA_SUPERTREND( int    startIdx,
                                     int    endIdx,
                                     const double inHigh[],
                                     const double inLow[],
                                     const double inClose[],
                                     int optInTimePeriod,
                                     double optInMultiplier,
                                     int          *outBegIdx,
                                     int          *outNBElement,
                                     double        outSupertrend[],
                                     int        outTrend[] )
{
   int i;
   int today;
   int outIdx;
   int lookbackTotal;
   int isUptrend;
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
   double medianPrice;
   double band;
   double basicUpper;
   double basicLower;
   double finalUpper;
   double finalLower;
   double closeToday;
   double prevClose;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 10;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( optInMultiplier == TA_REAL_DEFAULT )
      optInMultiplier = 3;
   else if( !(optInMultiplier >= 0e0 && optInMultiplier <= TA_REAL_MAX) )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !inClose )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outSupertrend )
      return TA_BAD_PARAM;
   if( !outTrend )
      return TA_BAD_PARAM;
   if( (const void *)outSupertrend == (const void *)outTrend )
      return TA_BAD_PARAM;

   *outBegIdx= 0;
   *outNBElement= 0;
   lookbackTotal = TA_SUPERTREND_Lookback(optInTimePeriod,optInMultiplier);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   /* The Average True Range is carried inline rather than taken from a call,
    * because the band and the ratchet advance together one bar at a time and a
    * whole-range buffer between them would not stream.
    *
    * The arithmetic order below is the bit-exactness contract with TA_ATR (do
    * not reorder): True Range from high-low, then the two previous-close
    * distances in that order; the seed summed from 0.0 over the first 'period'
    * True Ranges and divided once; the same two Wilder coefficients, wBeta
    * rounded and wAlpha derived from it, in one fused statement.
    */
   wBeta = (double)(optInTimePeriod - 1) / (double)optInTimePeriod;
   wAlpha = 1.0 - wBeta;
   today = startIdx - lookbackTotal + 1;
   periodTotal = 0.0;
   i = optInTimePeriod;
   while( i-- > 0 )
   {
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
   /* Skip the Average True Range's unstable period. Taking the count from the
    * lookback rather than naming the setting keeps the two from disagreeing.
    */
   i = lookbackTotal - optInTimePeriod;
   while( i != 0 )
   {
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
   /* The first bar has no band to ratchet against and no trend to carry, so
    * both bands take their unclamped value and the trend is seeded long, as
    * ta4j's SuperTrendIndicator does. The formula does not settle this and the
    * published implementations are split on it, so the choice is a convention:
    * it stays visible for as long as the first trend lasts, and on a series
    * whose close never leaves the band it never washes out at all.
    */
   medianPrice = (inHigh[startIdx] + inLow[startIdx]) / 2.0;
   band = optInMultiplier * prevATR;
   finalUpper = medianPrice + band;
   finalLower = medianPrice - band;
   isUptrend = 1;
   prevClose = inClose[startIdx];
   outSupertrend[0] = finalLower;
   outTrend[0] = 1;
   outIdx = 1;
   today = startIdx + 1;
   while( today <= endIdx )
   {
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
      medianPrice = (tempHT + tempLT) / 2.0;
      band = optInMultiplier * prevATR;
      basicUpper = medianPrice + band;
      basicLower = medianPrice - band;
      /* Each band ratchets toward price and is released only by a close on its
       * far side. Nothing keeps the two ordered -- each is released by its own
       * condition, so the lower one can end up above the upper one -- and where
       * that happens a flip puts the emitted band on the far side of the close.
       * The line is therefore NOT monotone within a trend, and the flag does not
       * always say which side of the line price is on; both read as invariants
       * and neither is one.
       */
      if( basicUpper < finalUpper || prevClose > finalUpper )
      {
         finalUpper = basicUpper;
      }
      if( basicLower > finalLower || prevClose < finalLower )
      {
         finalLower = basicLower;
      }
      closeToday = inClose[today];
      /* The trend is carried in its own variable rather than recovered by
       * comparing the previous output against the previous bands. The two
       * carried bands do coincide on some bars -- always on a bar with no true
       * range, and on some of them at a multiplier of zero -- and there that
       * comparison cannot tell them apart, so it would silently lose the
       * hysteresis on exactly the flat input a corpus of real prices lacks.
       */
      if( isUptrend )
      {
         if( closeToday < finalLower )
         {
            isUptrend = 0;
         }
      } else if( closeToday > finalUpper )
      {
         isUptrend = 1;
      }
      if( isUptrend )
      {
         outSupertrend[outIdx] = finalLower;
         outTrend[outIdx] = 1;
      } else 
      {
         outSupertrend[outIdx] = finalUpper;
         outTrend[outIdx] = -1;
      }
      prevClose = closeToday;
      outIdx += 1;
      today += 1;
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_SUPERTREND_Lookback( int optInTimePeriod, double optInMultiplier )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 10;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   if( optInMultiplier == TA_REAL_DEFAULT )
      optInMultiplier = 3;
   else if( !(optInMultiplier >= 0e0 && optInMultiplier <= TA_REAL_MAX) )
      return -1;
   /* Every output bar needs the Average True Range at the same bar, and nothing
    * else reaches further back, so the lookback is exactly the callee's. Never
    * restated here, which is what makes SUPERTREND inherit TA_FUNC_UNST_ATR.
    */
   return TA_ATR_Lookback(optInTimePeriod);
}
