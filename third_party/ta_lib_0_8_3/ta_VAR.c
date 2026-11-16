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

TA_LIB_API TA_RetCode TA_VAR( int    startIdx,
                              int    endIdx,
                              const double inReal[],
                              int optInTimePeriod,
                              double optInNbDev,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outReal[] )
{
   double tempReal;
   double shift;
   double periodTotal1;
   double periodTotal2;
   double meanValue1;
   double variance;
   double invPeriod;
   double peakTotal2;
   int i;
   int j;
   int outIdx;
   int trailingIdx;
   int windowStart;
   int nbInitialElementNeeded;
   int barsSinceReseed;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 5;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( optInNbDev == TA_REAL_DEFAULT )
      optInNbDev = 1;
   else if( !(optInNbDev >= TA_REAL_MIN && optInNbDev <= TA_REAL_MAX) )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* Identify the minimum number of price bar needed to calculate
    * at least one output.
    */
   nbInitialElementNeeded = optInTimePeriod - 1;
   /* Move up the start index if there is not enough initial data. */
   if( startIdx < nbInitialElementNeeded )
   {
      startIdx = nbInitialElementNeeded;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   invPeriod = 1.0 / (double)optInTimePeriod;
   /* Measure deviations against a shift near the window: the running sums
    * periodTotal1 = sum(inReal-shift) and periodTotal2 = sum((inReal-shift)^2)
    * stay at variance scale, so variance = periodTotal2/period - mean^2 no longer
    * subtracts two ~mean^2 quantities. Anchor the shift to the first window value
    * (also gives an exact 0 for period 1, with no division by period-1).
    */
   trailingIdx = startIdx - nbInitialElementNeeded;
   shift = inReal[trailingIdx];
   periodTotal1 = 0.0;
   periodTotal2 = 0.0;
   for( j = trailingIdx; j < startIdx; j += 1 )
   {
      tempReal = inReal[j] - shift;
      periodTotal1 += tempReal;
      tempReal *= tempReal;
      periodTotal2 += tempReal;
   }
   /* inReal and outReal may be the same buffer: each trailing value is consumed
    * before its slot is overwritten by the output.
    */
   i = startIdx;
   outIdx = 0;
   barsSinceReseed = 32 * optInTimePeriod;
   peakTotal2 = periodTotal2;
   do
   {
      /* Add the incoming value, measured against the shift. */
      tempReal = inReal[i] - shift;
      periodTotal1 += tempReal;
      tempReal *= tempReal;
      periodTotal2 += tempReal;
      peakTotal2 = (periodTotal2 > peakTotal2) ? periodTotal2 : peakTotal2;
      meanValue1 = periodTotal1 * invPeriod;
      variance = periodTotal2 * invPeriod - meanValue1 * meanValue1;
      /* Remove the trailing value (prepares the next window). */
      tempReal = inReal[trailingIdx] - shift;
      periodTotal1 -= tempReal;
      tempReal *= tempReal;
      periodTotal2 -= tempReal;
      trailingIdx += 1;
      /* Rebuild with a fresh two-pass when the variance has shrunk below 1e-6
       * of the LARGEST mean squared deviation held since the last rebuild, or at
       * least every 32 windows. Measure against that peak, not the current sum:
       * the rounding the running sums carry scales with the peak, so once a
       * series settles back near the shift, or an outlier leaves the window,
       * the current sum holds nothing but that rounding. The collapse is seen
       * on the first bar whose sums carry it, and the rebuild recomputes that
       * bar.
       */
      barsSinceReseed -= 1;
      if( variance < 0.000001 * (peakTotal2 * invPeriod) || barsSinceReseed <= 0 )
      {
         barsSinceReseed = 32 * optInTimePeriod;
         windowStart = i - nbInitialElementNeeded;
         tempReal = 0.0;
         for( j = windowStart; j <= i; j += 1 )
         {
            tempReal += inReal[j];
         }
         shift = tempReal * invPeriod;
         periodTotal1 = 0.0;
         periodTotal2 = 0.0;
         for( j = windowStart; j <= i; j += 1 )
         {
            tempReal = inReal[j] - shift;
            periodTotal1 += tempReal;
            tempReal *= tempReal;
            periodTotal2 += tempReal;
         }
         meanValue1 = periodTotal1 * invPeriod;
         variance = periodTotal2 * invPeriod - meanValue1 * meanValue1;
         /* A window flat to within the rounding of its own mean leaves the
          * variance at that rounding, which would fire the trigger again on
          * every bar. Anchored on one of its own values it cannot: the variance
          * is then at least 1/(2n) of the mean square it is extracted from.
          */
         if( variance < 0.000001 * (periodTotal2 * invPeriod) )
         {
            shift = inReal[i];
            periodTotal1 = 0.0;
            periodTotal2 = 0.0;
            for( j = windowStart; j <= i; j += 1 )
            {
               tempReal = inReal[j] - shift;
               periodTotal1 += tempReal;
               tempReal *= tempReal;
               periodTotal2 += tempReal;
            }
            meanValue1 = periodTotal1 * invPeriod;
            variance = periodTotal2 * invPeriod - meanValue1 * meanValue1;
         }
         /* Before the re-remove below: the peak must hold the whole window. */
         peakTotal2 = periodTotal2;
         /* After the re-anchor a window with any spread sits orders above this
          * floor, so it catches only a variance that rounding left at or below
          * 0. That keeps the output non-negative, which lets STDDEV and BBANDS
          * square-root it unconditionally (#243). A negative variance always
          * gets here: the peak is never negative, so the trigger fires on it.
          */
         if( variance < 0.000000000001 * (periodTotal2 * invPeriod) )
         {
            variance = 0.0;
         }
         /* Re-remove the trailing value under the new shift so the carried state
          * matches the non-reseed path.
          */
         tempReal = inReal[windowStart] - shift;
         periodTotal1 -= tempReal;
         tempReal *= tempReal;
         periodTotal2 -= tempReal;
      }
      outReal[outIdx++] = variance;
      i += 1;
   } while( i <= endIdx );
   /* All done. Indicate the output limits and return. */
   *outNBElement= outIdx;
   *outBegIdx= startIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_VAR_Lookback( int optInTimePeriod, double optInNbDev )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 5;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   if( optInNbDev == TA_REAL_DEFAULT )
      optInNbDev = 1;
   else if( !(optInNbDev >= TA_REAL_MIN && optInNbDev <= TA_REAL_MAX) )
      return -1;
   return optInTimePeriod - 1;
}
