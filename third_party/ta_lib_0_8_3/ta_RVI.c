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
TA_LIB_API TA_RetCode TA_RVI( int    startIdx,
                              int    endIdx,
                              const double inReal[],
                              int optInTimePeriod,
                              int optInStdDevPeriod,
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
   double peakTotal2;
   double invPeriod;
   double sigma;
   double delta;
   double upValue;
   double dnValue;
   double upTotal;
   double dnTotal;
   double prevUp;
   double prevDn;
   double wAlpha;
   double wBeta;
   double total;
   int i;
   int j;
   int outIdx;
   int today;
   int anchorIdx;
   int trailingIdx;
   int windowStart;
   int barsSinceReseed;
   int nbInitialElementNeeded;
   int lookbackTotal;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInStdDevPeriod == TA_INTEGER_DEFAULT )
      optInStdDevPeriod = 10;
   else if( (int)optInStdDevPeriod < 2 || (int)optInStdDevPeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   *outBegIdx= 0;
   *outNBElement= 0;
   lookbackTotal = TA_RVI_Lookback(optInTimePeriod,optInStdDevPeriod);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   /* wAlpha is derived FROM wBeta, never the reverse (rma.c): only that order
    * makes the pair sum to exactly 1, and TA_RMA over this function's two legs
    * has to be bit for bit what the fused step below computes.
    */
   wBeta = (double)(optInTimePeriod - 1) / (double)optInTimePeriod;
   wAlpha = 1.0 - wBeta;
   /* The per-bar sigma is var.c's step transcribed unchanged -- shifted running
    * sums against a near-window anchor, the reseed trigger, its floor, and the
    * re-remove under the new shift. Any algebraically equal spelling is a
    * different double, and the reference this is differenced against is
    * TA_STDDEV anchored at this call's own start.
    */
   nbInitialElementNeeded = optInStdDevPeriod - 1;
   invPeriod = 1.0 / (double)optInStdDevPeriod;
   anchorIdx = startIdx - lookbackTotal;
   trailingIdx = anchorIdx;
   shift = inReal[anchorIdx];
   periodTotal1 = 0.0;
   periodTotal2 = 0.0;
   today = anchorIdx + nbInitialElementNeeded;
   for( j = anchorIdx; j < today; j += 1 )
   {
      tempReal = inReal[j] - shift;
      periodTotal1 += tempReal;
      tempReal *= tempReal;
      periodTotal2 += tempReal;
   }
   barsSinceReseed = 32 * optInStdDevPeriod;
   peakTotal2 = periodTotal2;
   /* Seed both legs with the simple average of the first 'optInTimePeriod'
    * volatilities, as rma.c seeds. optInStdDevPeriod >= 2 is what keeps the
    * inReal[today-1] below in bounds on the very first bar.
    */
   upTotal = 0.0;
   dnTotal = 0.0;
   for( i = optInTimePeriod; i > 0; i -= 1 )
   {
      tempReal = inReal[today] - shift;
      periodTotal1 += tempReal;
      tempReal *= tempReal;
      periodTotal2 += tempReal;
      peakTotal2 = (periodTotal2 > peakTotal2) ? periodTotal2 : peakTotal2;
      meanValue1 = periodTotal1 * invPeriod;
      variance = periodTotal2 * invPeriod - meanValue1 * meanValue1;
      tempReal = inReal[trailingIdx] - shift;
      periodTotal1 -= tempReal;
      tempReal *= tempReal;
      periodTotal2 -= tempReal;
      trailingIdx += 1;
      barsSinceReseed -= 1;
      if( variance < 0.000001 * (peakTotal2 * invPeriod) || barsSinceReseed <= 0 )
      {
         barsSinceReseed = 32 * optInStdDevPeriod;
         windowStart = today - nbInitialElementNeeded;
         tempReal = 0.0;
         for( j = windowStart; j <= today; j += 1 )
         {
            tempReal += inReal[j];
         }
         shift = tempReal * invPeriod;
         periodTotal1 = 0.0;
         periodTotal2 = 0.0;
         for( j = windowStart; j <= today; j += 1 )
         {
            tempReal = inReal[j] - shift;
            periodTotal1 += tempReal;
            tempReal *= tempReal;
            periodTotal2 += tempReal;
         }
         meanValue1 = periodTotal1 * invPeriod;
         variance = periodTotal2 * invPeriod - meanValue1 * meanValue1;
         if( variance < 0.000001 * (periodTotal2 * invPeriod) )
         {
            shift = inReal[today];
            periodTotal1 = 0.0;
            periodTotal2 = 0.0;
            for( j = windowStart; j <= today; j += 1 )
            {
               tempReal = inReal[j] - shift;
               periodTotal1 += tempReal;
               tempReal *= tempReal;
               periodTotal2 += tempReal;
            }
            meanValue1 = periodTotal1 * invPeriod;
            variance = periodTotal2 * invPeriod - meanValue1 * meanValue1;
         }
         peakTotal2 = periodTotal2;
         if( variance < 0.000000000001 * (periodTotal2 * invPeriod) )
         {
            variance = 0.0;
         }
         tempReal = inReal[windowStart] - shift;
         periodTotal1 -= tempReal;
         tempReal *= tempReal;
         periodTotal2 -= tempReal;
      }
      sigma = sqrt(variance);
      delta = inReal[today] - inReal[today - 1];
      if( delta > 0.0 )
      {
         upTotal += sigma;
      } else if( delta < 0.0 )
      {
         dnTotal += sigma;
      }
      today += 1;
   }
   prevUp = upTotal / optInTimePeriod;
   prevDn = dnTotal / optInTimePeriod;
   /* Skip the unstable period. Same step, smoothed but not stored. */
   i = lookbackTotal - (optInStdDevPeriod - 1 + (optInTimePeriod - 1));
   while( i != 0 )
   {
      tempReal = inReal[today] - shift;
      periodTotal1 += tempReal;
      tempReal *= tempReal;
      periodTotal2 += tempReal;
      peakTotal2 = (periodTotal2 > peakTotal2) ? periodTotal2 : peakTotal2;
      meanValue1 = periodTotal1 * invPeriod;
      variance = periodTotal2 * invPeriod - meanValue1 * meanValue1;
      tempReal = inReal[trailingIdx] - shift;
      periodTotal1 -= tempReal;
      tempReal *= tempReal;
      periodTotal2 -= tempReal;
      trailingIdx += 1;
      barsSinceReseed -= 1;
      if( variance < 0.000001 * (peakTotal2 * invPeriod) || barsSinceReseed <= 0 )
      {
         barsSinceReseed = 32 * optInStdDevPeriod;
         windowStart = today - nbInitialElementNeeded;
         tempReal = 0.0;
         for( j = windowStart; j <= today; j += 1 )
         {
            tempReal += inReal[j];
         }
         shift = tempReal * invPeriod;
         periodTotal1 = 0.0;
         periodTotal2 = 0.0;
         for( j = windowStart; j <= today; j += 1 )
         {
            tempReal = inReal[j] - shift;
            periodTotal1 += tempReal;
            tempReal *= tempReal;
            periodTotal2 += tempReal;
         }
         meanValue1 = periodTotal1 * invPeriod;
         variance = periodTotal2 * invPeriod - meanValue1 * meanValue1;
         if( variance < 0.000001 * (periodTotal2 * invPeriod) )
         {
            shift = inReal[today];
            periodTotal1 = 0.0;
            periodTotal2 = 0.0;
            for( j = windowStart; j <= today; j += 1 )
            {
               tempReal = inReal[j] - shift;
               periodTotal1 += tempReal;
               tempReal *= tempReal;
               periodTotal2 += tempReal;
            }
            meanValue1 = periodTotal1 * invPeriod;
            variance = periodTotal2 * invPeriod - meanValue1 * meanValue1;
         }
         peakTotal2 = periodTotal2;
         if( variance < 0.000000000001 * (periodTotal2 * invPeriod) )
         {
            variance = 0.0;
         }
         tempReal = inReal[windowStart] - shift;
         periodTotal1 -= tempReal;
         tempReal *= tempReal;
         periodTotal2 -= tempReal;
      }
      sigma = sqrt(variance);
      delta = inReal[today] - inReal[today - 1];
      upValue = 0.0;
      dnValue = 0.0;
      if( delta > 0.0 )
      {
         upValue = sigma;
      } else if( delta < 0.0 )
      {
         dnValue = sigma;
      }
      prevUp = fma(wBeta, prevUp, wAlpha * upValue);
      prevDn = fma(wBeta, prevDn, wAlpha * dnValue);
      today += 1;
      i -= 1;
   }
   /* A tie feeds neither leg, so both can be exactly zero at the same bar --
    * reachable on real data whenever the smoothing has no memory. Test the
    * total exactly: a band would carry the quote unit and zero the oscillator
    * for any instrument priced under it.
    */
   total = prevUp + prevDn;
   outReal[0] = (total == 0.0) ? 50.0 : 100.0 * (prevUp / total);
   outIdx = 1;
   while( today <= endIdx )
   {
      tempReal = inReal[today] - shift;
      periodTotal1 += tempReal;
      tempReal *= tempReal;
      periodTotal2 += tempReal;
      peakTotal2 = (periodTotal2 > peakTotal2) ? periodTotal2 : peakTotal2;
      meanValue1 = periodTotal1 * invPeriod;
      variance = periodTotal2 * invPeriod - meanValue1 * meanValue1;
      tempReal = inReal[trailingIdx] - shift;
      periodTotal1 -= tempReal;
      tempReal *= tempReal;
      periodTotal2 -= tempReal;
      trailingIdx += 1;
      barsSinceReseed -= 1;
      if( variance < 0.000001 * (peakTotal2 * invPeriod) || barsSinceReseed <= 0 )
      {
         barsSinceReseed = 32 * optInStdDevPeriod;
         windowStart = today - nbInitialElementNeeded;
         tempReal = 0.0;
         for( j = windowStart; j <= today; j += 1 )
         {
            tempReal += inReal[j];
         }
         shift = tempReal * invPeriod;
         periodTotal1 = 0.0;
         periodTotal2 = 0.0;
         for( j = windowStart; j <= today; j += 1 )
         {
            tempReal = inReal[j] - shift;
            periodTotal1 += tempReal;
            tempReal *= tempReal;
            periodTotal2 += tempReal;
         }
         meanValue1 = periodTotal1 * invPeriod;
         variance = periodTotal2 * invPeriod - meanValue1 * meanValue1;
         if( variance < 0.000001 * (periodTotal2 * invPeriod) )
         {
            shift = inReal[today];
            periodTotal1 = 0.0;
            periodTotal2 = 0.0;
            for( j = windowStart; j <= today; j += 1 )
            {
               tempReal = inReal[j] - shift;
               periodTotal1 += tempReal;
               tempReal *= tempReal;
               periodTotal2 += tempReal;
            }
            meanValue1 = periodTotal1 * invPeriod;
            variance = periodTotal2 * invPeriod - meanValue1 * meanValue1;
         }
         peakTotal2 = periodTotal2;
         if( variance < 0.000000000001 * (periodTotal2 * invPeriod) )
         {
            variance = 0.0;
         }
         tempReal = inReal[windowStart] - shift;
         periodTotal1 -= tempReal;
         tempReal *= tempReal;
         periodTotal2 -= tempReal;
      }
      sigma = sqrt(variance);
      delta = inReal[today] - inReal[today - 1];
      upValue = 0.0;
      dnValue = 0.0;
      if( delta > 0.0 )
      {
         upValue = sigma;
      } else if( delta < 0.0 )
      {
         dnValue = sigma;
      }
      prevUp = fma(wBeta, prevUp, wAlpha * upValue);
      prevDn = fma(wBeta, prevDn, wAlpha * dnValue);
      total = prevUp + prevDn;
      outReal[outIdx++] = (total == 0.0) ? 50.0 : 100.0 * (prevUp / total);
      today += 1;
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_NOINLINE TA_LIB_API int TA_RVI_Lookback( int optInTimePeriod, int optInStdDevPeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   if( (int)optInStdDevPeriod == TA_INTEGER_DEFAULT )
      optInStdDevPeriod = 10;
   else if( (int)optInStdDevPeriod < 2 || (int)optInStdDevPeriod > 100000 )
      return -1;
   return optInStdDevPeriod - 1 + (optInTimePeriod - 1) + TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_RVI,Rvi,((optInTimePeriod > 1) ? 10 * optInTimePeriod : 0),((optInTimePeriod > 1) ? 19 * optInTimePeriod : 0));
}
