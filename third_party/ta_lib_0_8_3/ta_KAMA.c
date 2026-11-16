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
TA_LIB_API TA_RetCode TA_KAMA( int    startIdx,
                               int    endIdx,
                               const double inReal[],
                               int optInTimePeriod,
                               int          *outBegIdx,
                               int          *outNBElement,
                               double        outReal[] )
{
   double constMax;
   double constDiff;
   double tempReal;
   double tempReal2;
   double sumROC1;
   double periodROC;
   double prevKAMA;
   int i;
   int today;
   int outIdx;
   int lookbackTotal;
   int trailingIdx;
   int nullRun;
   int live;
   double trailingValue;

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

   constMax = 2.0 / (30.0 + 1.0);
   constDiff = 2.0 / (2.0 + 1.0) - constMax;
   /* Default return values */
   *outBegIdx= 0;
   *outNBElement= 0;
   /* No smoothing at period of 1: the output is a copy of the input
    * (same convention as TA_MA for every MAType). The unstable period
    * still delays the first output for API consistency.
    */
   if( optInTimePeriod == 1 )
   {
      lookbackTotal = TA_KAMA_Lookback(optInTimePeriod);
      if( startIdx < lookbackTotal )
      {
         startIdx = lookbackTotal;
      }
      if( startIdx > endIdx )
      {
         return TA_SUCCESS;
      }
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
   /* Identify the minimum number of price bar needed
    * to calculate at least one output.
    */
   lookbackTotal = TA_KAMA_Lookback(optInTimePeriod);
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
   /* Initialize the variables by going through
    * the lookback period.
    */
   sumROC1 = 0.0;
   /* Consecutive 1-day changes of exactly zero, counted so that a flat window
    * can be recognized exactly (the shape #244 needed for MFI). sumROC1 cannot
    * answer that question itself once the window starts sliding: it is
    * maintained by add-then-subtract, so a window that has gone flat leaves it
    * holding rounding residue of arbitrary sign rather than zero, and the
    * efficiency ratio then divides that residue into itself.
    */
   nullRun = 0;
   today = startIdx - lookbackTotal;
   trailingIdx = today;
   i = optInTimePeriod;
   while( i-- > 0 )
   {
      tempReal = inReal[today++];
      tempReal -= inReal[today];
      sumROC1 += fabs(tempReal);
      live = (fabs(tempReal) <= 0.0) ? 0 : 1;
      nullRun = (nullRun + 1) & (live - 1);
   }
   /* At this point sumROC1 represent the
    * summation of the 1-day price difference
    * over the (optInTimePeriod-1)
    */
   /* Calculate the first KAMA */
   /* The yesterday price is used here as the previous KAMA. */
   prevKAMA = inReal[today - 1];
   tempReal = inReal[today];
   tempReal2 = inReal[trailingIdx++];
   periodROC = tempReal - tempReal2;
   /* Save the trailing value. Do this because inReal
    * and outReal can be pointers to the same buffer.
    */
   trailingValue = tempReal2;
   /* Calculate the efficiency ratio.
    *
    * The ratio cannot exceed 1 in exact arithmetic, but sumROC1 drifts, so
    * clamp it: past 1 the squared smoothing constant amplifies instead of
    * averaging and the recurrence below stops being a convex combination.
    *
    * `sumROC1 <= 0.0` (#385) is now numerically redundant -- the clamp maps its
    * +Inf onto the same 1.0 -- so a value test written against it would pass
    * with it deleted. Keep it anyway: the divisor sweep requires a division to
    * be dominated by a test of its own divisor against a literal zero.
    */
   if( sumROC1 <= 0.0 || sumROC1 <= periodROC )
   {
      tempReal = 1.0;
   } else 
   {
      tempReal = fabs(periodROC / sumROC1);
      if( tempReal > 1.0 )
      {
         tempReal = 1.0;
      }
   }
   /* Calculate the smoothing constant */
   tempReal = fma(tempReal, constDiff, constMax);
   tempReal *= tempReal;
   /* Calculate the KAMA like an EMA, using the
    * smoothing constant as the adaptive factor.
    */
   prevKAMA = fma(inReal[today++] - prevKAMA, tempReal, prevKAMA);
   /* 'today' keep track of where the processing is within the
    * input.
    */
   /* Skip the unstable period. Do the whole processing
    * needed for KAMA, but do not write it in the output.
    */
   while( today <= startIdx )
   {
      tempReal = inReal[today];
      tempReal2 = inReal[trailingIdx++];
      periodROC = tempReal - tempReal2;
      /* Adjust sumROC1:
       *  - Remove trailing ROC1
       *  - Add new ROC1
       */
      sumROC1 -= fabs(trailingValue - tempReal2);
      sumROC1 += fabs(tempReal - inReal[today - 1]);
      /* Once a whole window of flat bars has gone by, every 1-day change it
       * spans is exactly zero, so the sum is known to be exactly zero, the
       * residue can be dropped, and the efficiency ratio is 1.
       */
      live = (fabs(tempReal - inReal[today - 1]) <= 0.0) ? 0 : 1;
      nullRun = (nullRun + 1) & (live - 1);
      if( TA_UNLIKELY(nullRun >= optInTimePeriod) )
      {
         nullRun = optInTimePeriod;
         sumROC1 = 0.0;
         tempReal = 1.0;
      } else if( sumROC1 <= 0.0 ||
          sumROC1 <= periodROC ) /* Calculate the efficiency ratio */
      {
         tempReal = 1.0;
      } else 
      {
         tempReal = fabs(periodROC / sumROC1);
         if( tempReal > 1.0 )
         {
            tempReal = 1.0;
         }
      }
      /* Save the trailing value. Do this because inReal
       * and outReal can be pointers to the same buffer.
       */
      trailingValue = tempReal2;
      /* Calculate the smoothing constant */
      tempReal = fma(tempReal, constDiff, constMax);
      tempReal *= tempReal;
      /* Calculate the KAMA like an EMA, using the
       * smoothing constant as the adaptive factor.
       */
      prevKAMA = fma(inReal[today++] - prevKAMA, tempReal, prevKAMA);
   }
   /* Write the first value. */
   outReal[0] = prevKAMA;
   outIdx = 1;
   *outBegIdx= today - 1;
   /* Do the KAMA calculation for the requested range. */
   while( today <= endIdx )
   {
      tempReal = inReal[today];
      tempReal2 = inReal[trailingIdx++];
      periodROC = tempReal - tempReal2;
      /* Adjust sumROC1:
       *  - Remove trailing ROC1
       *  - Add new ROC1
       */
      sumROC1 -= fabs(trailingValue - tempReal2);
      sumROC1 += fabs(tempReal - inReal[today - 1]);
      /* Once a whole window of flat bars has gone by, every 1-day change it
       * spans is exactly zero, so the sum is known to be exactly zero, the
       * residue can be dropped, and the efficiency ratio is 1.
       */
      live = (fabs(tempReal - inReal[today - 1]) <= 0.0) ? 0 : 1;
      nullRun = (nullRun + 1) & (live - 1);
      if( TA_UNLIKELY(nullRun >= optInTimePeriod) )
      {
         nullRun = optInTimePeriod;
         sumROC1 = 0.0;
         tempReal = 1.0;
      } else if( sumROC1 <= 0.0 ||
          sumROC1 <= periodROC ) /* Calculate the efficiency ratio */
      {
         tempReal = 1.0;
      } else 
      {
         tempReal = fabs(periodROC / sumROC1);
         if( tempReal > 1.0 )
         {
            tempReal = 1.0;
         }
      }
      /* Save the trailing value. Do this because inReal
       * and outReal can be pointers to the same buffer.
       */
      trailingValue = tempReal2;
      /* Calculate the smoothing constant */
      tempReal = fma(tempReal, constDiff, constMax);
      tempReal *= tempReal;
      /* Calculate the KAMA like an EMA, using the
       * smoothing constant as the adaptive factor.
       */
      prevKAMA = fma(inReal[today++] - prevKAMA, tempReal, prevKAMA);
      outReal[outIdx++] = prevKAMA;
   }
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_NOINLINE TA_LIB_API int TA_KAMA_Lookback( int optInTimePeriod )
{
   int root;
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 30;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   root = (int)sqrt((double)optInTimePeriod);
   if( optInTimePeriod == 1 )
   {
      return TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_KAMA,Kama,0,0);
   }
   return optInTimePeriod + TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_KAMA,Kama,25 * 4 * root,25 * 8 * root);
}
