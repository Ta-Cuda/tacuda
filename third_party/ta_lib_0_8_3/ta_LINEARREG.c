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
TA_LIB_API TA_RetCode TA_LINEARREG( int    startIdx,
                                    int    endIdx,
                                    const double inReal[],
                                    int optInTimePeriod,
                                    int          *outBegIdx,
                                    int          *outNBElement,
                                    double        outReal[] )
{
   int outIdx;
   int today;
   int lookbackTotal;
   int trailingIdx;
   double SumX;
   double SumXY;
   double SumY;
   double SumXSqr;
   double Divisor;
   double m;
   double b;
   int i;
   int j;
   int windowStart;
   int barsSinceReseed;
   double tempValue1;
   double tempValue2;
   double trailingValue;
   double weightedTrailing;
   double sumAbs;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* Linear Regression is a concept also known as the
    * "least squares method" or "best fit." Linear
    * Regression attempts to fit a straight line between
    * several data points in such a way that distance
    * between each data point and the line is minimized.
    *
    * For each point, a straight line over the specified
    * previous bar period is determined in terms
    * of y = b + m*x:
    *
    * TA_LINEARREG          : Returns b+m*(period-1)
    * TA_LINEARREG_SLOPE    : Returns 'm'
    * TA_LINEARREG_ANGLE    : Returns 'm' in degree.
    * TA_LINEARREG_INTERCEPT: Returns 'b'
    * TA_TSF                : Returns b+m*(period)
    */
   /* Adjust startIdx to account for the lookback period. */
   lookbackTotal = TA_LINEARREG_Lookback(optInTimePeriod);
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
   outIdx = 0;
   /* Index into the output. */
   today = startIdx;
   trailingIdx = startIdx - lookbackTotal;
   SumX = (double)optInTimePeriod * (optInTimePeriod - 1) * 0.5;
   SumXSqr = (double)optInTimePeriod * (optInTimePeriod - 1) * (2 * optInTimePeriod - 1) / 6.0;
   Divisor = SumX * SumX - optInTimePeriod * SumXSqr;
   /* Prime the two data-dependent window sums for the first output with a
    * one-time full-window scan. SumX/SumXSqr/Divisor are period-only constants;
    * SumY = sum of the window, SumXY = sum of i*value (i the reversed
    * 0..period-1 position).
    */
   SumXY = 0;
   SumY = 0;
   sumAbs = 0;
   for( i = optInTimePeriod; i-- != 0;  )
   {
      tempValue1 = inReal[today - i];
      SumY += tempValue1;
      SumXY += (double)i * tempValue1;
      sumAbs += fabs(tempValue1);
   }
   m = (optInTimePeriod * SumXY - SumX * SumY) / Divisor;
   b = (SumY - m * SumX) / (double)optInTimePeriod;
   barsSinceReseed = 32 * optInTimePeriod;
   trailingValue = inReal[trailingIdx];
   trailingIdx += 1;
   outReal[outIdx++] = fma(m, (double)(optInTimePeriod - 1), b);
   today += 1;
   /* Slide the window one bar at a time, keeping both sums in O(1): advancing
    * the window raises every retained value's weight by 1 (adds SumY) and drops
    * the departing value at full weight (subtracts period*trailingValue). Same
    * incremental identity as WMA/CORREL; the output arithmetic is unchanged.
    * (perf #103 -- numerics-changing: running total vs per-bar fresh sum.)
    * Each departing value is read before the output write of the same bar:
    * with outReal==inReal (in-place, #130) that write lands on the cell the
    * next iteration departs from.
    */
   while( today <= endIdx )
   {
      weightedTrailing = (double)optInTimePeriod * trailingValue;
      SumXY = SumXY + SumY - weightedTrailing;
      SumY = SumY - trailingValue + inReal[today];
      sumAbs = sumAbs - fabs(trailingValue) + fabs(inReal[today]);
      /* Re-anchor: rebuild both sums from the window itself. #103 left them as
       * running totals that are never rebuilt, so each bar's rounding joins a
       * residue no later bar can subtract -- unbounded in the length of the
       * call, and scaled by the largest value the sums have EVER held rather
       * than by what the window holds now. Two triggers, and they cover
       * different failures (issue #254):
       *
       *   - every 32*period bars, so a slow drift stays bounded however long
       *     the series runs. Same interval as TA_VAR / TA_CORREL / TA_BETA.
       *
       *   - when the value the window just dropped carries more weight than
       *     everything left in it. That is the one the interval cannot cover:
       *     one large print inflates the residue for up to 32*period bars
       *     after it is gone (measured 31x at period 5), and this rebuilds on
       *     the bar it leaves instead.
       *
       * The threshold compares two DEGREE-1 quantities, so it is 100 rather
       * than the 1e6 a degree-2 ratio would take. On ordinary prices it is ~1 and this
       * never fires; it is a compare, not work. The constant is 100 rather than
       * 10 because at 10 a zero-mean oscillator rebuilds on 8.8% of bars for no
       * measured accuracy gain.
       *
       * THE DENOMINATOR IS sumAbs, NOT SumY, AND THAT IS THE WHOLE POINT.
       * SumY is a CANCELLING sum: on a zero-mean window it collapses toward 0
       * while the departing value does not, so |weightedTrailing|/|SumY| is
       * unbounded and the rebuild fires on EVERY bar -- an alternating +/-1
       * series measured 10.9x slower at period 30, which is precisely the
       * O(n*period) cost #103 removed. Same shape of error as #242's absolute
       * guard on a quartic quantity: a ratio test is ill-posed when its
       * denominator can cancel. sumAbs is a sum of magnitudes, so it is 0 only
       * when every value in the window is 0 -- and then the numerator is 0 too
       * and the test is false. There is no window it can misjudge.
       *
       * It is also the RIGHT quantity on the merits, not just the safe one: a
       * fresh rebuild's own error is ~eps*sum|y|, so comparing the departing
       * term against sum|y| asks exactly "would rebuilding beat what we are
       * carrying?".
       *
       * Carrying it is free in practice. Measured on the shipped libta-lib.a it
       * costs nothing against the |SumY| form on a price series (1.541 vs 1.605
       * ns/bar at period 14) because the update is INDEPENDENT of the serial
       * SumXY -> SumY dependency chain and fills slots that were idle. The
       * rejected alternative -- keeping |SumY| and rate-limiting the trigger to
       * once per `period` bars -- bounded the cliff at 1.2x rather than removing
       * it, and silently dropped any print departing within `period` bars of a
       * rebuild (~3% of them).
       *
       * The scan walks the window oldest-first with the weight counting DOWN,
       * which is the priming scan's order and weighting -- so a reseeded bar is
       * bit-identical to the same bar computed by a call that started there.
       * That identity is the whole point: it is what the range-stability
       * contract measures.
       *
       * Reading the window is safe when outReal aliases inReal (#130): the
       * outputs written so far occupy [0, outIdx-1], and windowStart is
       * today-lookbackTotal, which is >= outIdx because startIdx was clamped
       * to at least lookbackTotal.
       */
      barsSinceReseed -= 1;
      if( barsSinceReseed <= 0 || fabs(weightedTrailing) > 100.0 * sumAbs )
      {
         barsSinceReseed = 32 * optInTimePeriod;
         windowStart = today - lookbackTotal;
         SumY = 0;
         SumXY = 0;
         sumAbs = 0;
         tempValue2 = (double)lookbackTotal;
         for( j = windowStart; j <= today; j += 1 )
         {
            tempValue1 = inReal[j];
            SumY += tempValue1;
            SumXY += tempValue2 * tempValue1;
            sumAbs += fabs(tempValue1);
            tempValue2 -= 1.0;
         }
      }
      m = (optInTimePeriod * SumXY - SumX * SumY) / Divisor;
      b = (SumY - m * SumX) / (double)optInTimePeriod;
      trailingValue = inReal[trailingIdx];
      trailingIdx += 1;
      outReal[outIdx++] = fma(m, (double)(optInTimePeriod - 1), b);
      today += 1;
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_LINEARREG_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   return optInTimePeriod - 1;
}
