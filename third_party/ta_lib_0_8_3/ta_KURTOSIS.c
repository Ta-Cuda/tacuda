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
TA_LIB_API TA_RetCode TA_KURTOSIS( int    startIdx,
                                   int    endIdx,
                                   const double inReal[],
                                   int optInTimePeriod,
                                   int          *outBegIdx,
                                   int          *outNBElement,
                                   double        outReal[] )
{
   double tempReal;
   double shift;
   double total1;
   double total2;
   double total3;
   double total4;
   double dev;
   double dev2;
   double residue;
   double residueSq;
   double moment2;
   double moment4;
   double sampleVar;
   double varSquared;
   double dPeriod;
   double coefA;
   double coefB;
   double invPeriod;
   double kurt;
   double peak2;
   double peak4;
   int i;
   int j;
   int outIdx;
   int trailingIdx;
   int windowStart;
   int nbInitialElementNeeded;
   int barsSinceReseed;
   int reseedPeriod;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 30;
   else if( (int)optInTimePeriod < 4 || (int)optInTimePeriod > 10000 )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   nbInitialElementNeeded = optInTimePeriod - 1;
   if( startIdx < nbInitialElementNeeded )
   {
      startIdx = nbInitialElementNeeded;
   }
   if( startIdx > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   /* G2, the sample-adjusted Fisher excess kurtosis. The coefficients are
    * computed in double: at the top of the parameter range coefA's denominator
    * (n-1)(n-2)(n-3) is ~1e12, far past what an int holds.
    *
    * The (n-2)(n-3) denominators are why the range starts at 4 rather than 1.
    * The argument contract rejects anything below it, and the n-1 lookback
    * guarantees a full window at the first emitted bar, so the estimator is
    * never handed fewer than four points and needs no runtime branch for it.
    */
   dPeriod = (double)optInTimePeriod;
   coefA = dPeriod * (dPeriod + 1.0) / ((dPeriod - 1.0) * (dPeriod - 2.0) * (dPeriod - 3.0));
   coefB = 3.0 * (dPeriod - 1.0) * (dPeriod - 1.0) / ((dPeriod - 2.0) * (dPeriod - 3.0));
   invPeriod = 1.0 / dPeriod;
   /* Deviations are measured against a shift near the window, so the running
    * sums stay at deviation scale instead of at price scale, and the central
    * moments are recovered from them by the binomial expansion below. The
    * period bounds the rounding the slide accumulates and is the only way out
    * of a NaN or Inf input, on which every trigger comparison is false; a stale
    * shift is caught by the triggers in the loop, not by this.
    */
   reseedPeriod = 32 * optInTimePeriod;
   trailingIdx = startIdx - nbInitialElementNeeded;
   shift = inReal[trailingIdx];
   total1 = 0.0;
   total2 = 0.0;
   total3 = 0.0;
   total4 = 0.0;
   for( j = trailingIdx; j < startIdx; j += 1 )
   {
      dev = inReal[j] - shift;
      dev2 = dev * dev;
      total1 += dev;
      total2 += dev2;
      total3 += dev2 * dev;
      total4 += dev2 * dev2;
   }
   i = startIdx;
   outIdx = 0;
   barsSinceReseed = reseedPeriod;
   peak2 = total2;
   peak4 = total4;
   do
   {
      dev = inReal[i] - shift;
      dev2 = dev * dev;
      total1 += dev;
      total2 += dev2;
      total3 += dev2 * dev;
      total4 += dev2 * dev2;
      if( total2 > peak2 )
      {
         peak2 = total2;
      }
      if( total4 > peak4 )
      {
         peak4 = total4;
      }
      residue = total1 * invPeriod;
      residueSq = residue * residue;
      moment2 = total2 - dPeriod * residueSq;
      moment4 = fma(6.0 * residueSq, total2, total4 - 4.0 * residue * total3) - 3.0 * dPeriod * residueSq * residueSq;
      /* Rebuild once either central moment falls below 1% of the largest
       * shifted sum of its power held since the last rebuild. That catches a
       * shift several sigma stale, where the expansion above cancels as
       * (u/sigma)^4, and a series decaying back onto the shift or an outlier
       * leaving the window, which leave current sums made mostly of rounding
       * that scales with the peaks.
       */
      barsSinceReseed -= 1;
      if( moment2 < 0.01 * peak2 || moment4 < 0.01 * peak4 || barsSinceReseed <= 0 )
      {
         barsSinceReseed = reseedPeriod;
         windowStart = i - nbInitialElementNeeded;
         tempReal = 0.0;
         for( j = windowStart; j <= i; j += 1 )
         {
            tempReal += inReal[j];
         }
         shift = tempReal * invPeriod;
         total1 = 0.0;
         total2 = 0.0;
         total3 = 0.0;
         total4 = 0.0;
         for( j = windowStart; j <= i; j += 1 )
         {
            dev = inReal[j] - shift;
            dev2 = dev * dev;
            total1 += dev;
            total2 += dev2;
            total3 += dev2 * dev;
            total4 += dev2 * dev2;
         }
         residue = total1 * invPeriod;
         residueSq = residue * residue;
         moment2 = total2 - dPeriod * residueSq;
         moment4 = fma(6.0 * residueSq, total2, total4 - 4.0 * residue * total3) - 3.0 * dPeriod * residueSq * residueSq;
         /* The left-to-right sum misses the mean by many ulps at a long
          * period, and a window whose spread is under that miss would fire a
          * trigger again on every bar. The residue just measured is the
          * miss: adding it back lands within about an ulp of the mean, where
          * a flat window's deviations are exactly 0.
          */
         if( moment2 < 0.01 * total2 || moment4 < 0.01 * total4 )
         {
            shift += residue;
            total1 = 0.0;
            total2 = 0.0;
            total3 = 0.0;
            total4 = 0.0;
            for( j = windowStart; j <= i; j += 1 )
            {
               dev = inReal[j] - shift;
               dev2 = dev * dev;
               total1 += dev;
               total2 += dev2;
               total3 += dev2 * dev;
               total4 += dev2 * dev2;
            }
            residue = total1 * invPeriod;
            residueSq = residue * residue;
            moment2 = total2 - dPeriod * residueSq;
            moment4 = fma(6.0 * residueSq, total2, total4 - 4.0 * residue * total3) - 3.0 * dPeriod * residueSq * residueSq;
         }
         peak2 = total2;
         peak4 = total4;
      }
      /* No spread, no kurtosis: 0 would assert normality and -1.2 uniformity.
       * A flat window reaches here with every deviation exactly 0, so this is
       * 0/0 and IEEE answers NaN without a branch. A guard with an absolute
       * epsilon would be a cliff at a price level, not a noise floor (#243).
       */
      sampleVar = moment2 / (dPeriod - 1.0);
      varSquared = sampleVar * sampleVar;
      kurt = coefA * (moment4 / varSquared) - coefB;
      /* inReal and outReal may be the same buffer: read the trailing value
       * before the output write.
       */
      dev = inReal[trailingIdx] - shift;
      dev2 = dev * dev;
      total1 -= dev;
      total2 -= dev2;
      total3 -= dev2 * dev;
      total4 -= dev2 * dev2;
      trailingIdx += 1;
      outReal[outIdx++] = kurt;
      i += 1;
   } while( i <= endIdx );
   *outNBElement= outIdx;
   *outBegIdx= startIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_KURTOSIS_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 30;
   else if( (int)optInTimePeriod < 4 || (int)optInTimePeriod > 10000 )
      return -1;
   return optInTimePeriod - 1;
}
