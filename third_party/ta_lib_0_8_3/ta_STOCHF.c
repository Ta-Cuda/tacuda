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

TA_LIB_API TA_RetCode TA_STOCHF( int    startIdx,
                                 int    endIdx,
                                 const double inHigh[],
                                 const double inLow[],
                                 const double inClose[],
                                 int optInFastK_Period,
                                 int optInFastD_Period,
                                 TA_MAType optInFastD_MAType,
                                 int          *outBegIdx,
                                 int          *outNBElement,
                                 double        outFastK[],
                                 double        outFastD[] )
{
   TA_RetCode retCode;
   double lowest;
   double highest;
   double tmp;
   double *tempBuffer;
   int outIdx;
   int lowestIdx;
   int highestIdx;
   int lookbackTotal;
   int lookbackK;
   int lookbackFastD;
   int trailingIdx;
   int today;
   int i;
   int bufferIsAllocated;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInFastK_Period == TA_INTEGER_DEFAULT )
      optInFastK_Period = 5;
   else if( (int)optInFastK_Period < 1 || (int)optInFastK_Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInFastD_Period == TA_INTEGER_DEFAULT )
      optInFastD_Period = 3;
   else if( (int)optInFastD_Period < 1 || (int)optInFastD_Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInFastD_MAType == TA_INTEGER_DEFAULT || optInFastD_MAType == TA_MAType_DEFAULT )
      optInFastD_MAType = 0;
   else if( (int)optInFastD_MAType < TA_MATYPE_MIN || (int)optInFastD_MAType > TA_MATYPE_MAX )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !inClose )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outFastK )
      return TA_BAD_PARAM;
   if( !outFastD )
      return TA_BAD_PARAM;
   if( outFastK == outFastD )
      return TA_BAD_PARAM;

   i = 0;
   /* With stochastic, there is a total of 4 different lines that
    * are defined: FASTK, FASTD, SLOWK and SLOWD.
    *
    * The D is the signal line usually drawn over its
    * corresponding K function.
    *
    *                    (Today's Close - LowestLow)
    *  FASTK(Kperiod) =  --------------------------- * 100
    *                     (HighestHigh - LowestLow)
    *
    *  FASTD(FastDperiod, MA type) = MA Smoothed FASTK over FastDperiod
    *
    *  SLOWK(SlowKperiod, MA type) = MA Smoothed FASTK over SlowKperiod
    *
    *  SLOWD(SlowDperiod, MA Type) = MA Smoothed SLOWK over SlowDperiod
    *
    * The HighestHigh and LowestLow are the extreme values among the
    * last 'Kperiod'.
    *
    * SLOWK and FASTD are equivalent when using the same period.
    *
    * The following shows how these four lines are made available in TA-LIB:
    *
    *  TA_STOCH  : Returns the SLOWK and SLOWD
    *  TA_STOCHF : Returns the FASTK and FASTD
    *
    * The TA_STOCH function correspond to the more widely implemented version
    * found in many software/charting package. The TA_STOCHF is more rarely
    * used because its higher volatility cause often whipsaws.
    */
   /* Identify the lookback needed. */
   lookbackK = optInFastK_Period - 1;
   lookbackFastD = TA_MA_Lookback(optInFastD_Period,optInFastD_MAType);
   lookbackTotal = lookbackK + lookbackFastD;
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
      /* Succeed... but no data in the output. */
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   /* Do the K calculation:
    *
    *    Kt = 100 x ((Ct-Lt)/(Ht-Lt))
    *
    * Kt is today stochastic
    * Ct is today closing price.
    * Lt is the lowest price of the last K Period (including today)
    * Ht is the highest price of the last K Period (including today)
    */
   /* Proceed with the calculation for the requested range.
    * Note that this algorithm allows the input and
    * output to be the same buffer.
    */
   outIdx = 0;
   /* Calculate just enough K for ending up with the caller
    * requested range. (The range of k must consider all
    * the lookback involve with the smoothing).
    */
   trailingIdx = startIdx - lookbackTotal;
   today = trailingIdx + lookbackK;
   highestIdx = -1;
   lowestIdx = highestIdx;
   lowest = 0.0;
   highest = lowest;
   /* Allocate a temporary buffer large enough to
    * store the K.
    *
    * When outFastK aliases a price input the caller buffer doubles as the
    * scratch, saving one allocation: the K writes trail the min/max window
    * reads, and the final memmove is overlap-safe. outFastD must NOT be
    * elected: the %D ma() below would then run in place over the raw K
    * that the memmove into outFastK still needs (issue #130).
    */
   bufferIsAllocated = 0;
   if( outFastK == inHigh || outFastK == inLow || outFastK == inClose )
   {
      tempBuffer = outFastK;
   } else 
   {
      bufferIsAllocated = 1;
      tempBuffer = malloc((endIdx - today + 1) * sizeof(double));
      if( !tempBuffer )
      {
         *outBegIdx= 0;
         *outNBElement= 0;
         return TA_ALLOC_ERR;
      }
   }
   /* Do the K calculation */
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
      /* Divide by the range itself and scale after: the guard has to test the
       * very expression the division uses, or a scaling step can carry a
       * guarded-non-zero into a zero divisor.
       *
       * The band is the range against ITS OWN two extremes, not a fixed
       * constant: the range carries the quote unit, so a constant answers
       * "flat" for every window of an instrument quoted below it (issue #253).
       * It absorbs the machine-flat window an exact test would divide into
       * [0,100] noise (issue #107 / STOCHRSI).
       */
      if( !TA_IS_ZERO_SCALED(highest - lowest, fabs(highest) + fabs(lowest)) )
      {
         tempBuffer[outIdx++] = (inClose[today] - lowest) / (highest - lowest) * 100.0;
      } else 
      {
         tempBuffer[outIdx++] = 0.0;
      }
      trailingIdx += 1;
      today += 1;
   }
   /* Fast-K calculation completed. This K calculation is returned
    * to the caller. It is smoothed to become Fast-D.
    */
   retCode = TA_MA(0,outIdx - 1,tempBuffer,optInFastD_Period,optInFastD_MAType,outBegIdx,outNBElement,outFastD);
   if( retCode != TA_SUCCESS || (int)*outNBElement == 0 )
   {
      if( bufferIsAllocated )
      {
         free(tempBuffer);
      }
      /* Something wrong happen? No further data? */
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   /* Copy tempBuffer into the caller buffer.
    * (Calculation could not be done directly in the
    *  caller buffer because more input data then the
    *  requested range was needed for doing %D).
    */
   /* memmove, not memcpy: tempBuffer aliases outFastK when the caller buffer is
    * reused as scratch, so source and destination overlap (issue #94).
    */
   memmove(outFastK,&tempBuffer[lookbackFastD],(int)*outNBElement * sizeof(double));
   if( bufferIsAllocated )
   {
      free(tempBuffer);
   }
   /* Note: Keep the outBegIdx relative to the
    *       caller input before returning.
    */
   *outBegIdx= startIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_STOCHF_Lookback( int optInFastK_Period, int optInFastD_Period, TA_MAType optInFastD_MAType )
{
   int retValue;
   if( (int)optInFastK_Period == TA_INTEGER_DEFAULT )
      optInFastK_Period = 5;
   else if( (int)optInFastK_Period < 1 || (int)optInFastK_Period > 100000 )
      return -1;
   if( (int)optInFastD_Period == TA_INTEGER_DEFAULT )
      optInFastD_Period = 3;
   else if( (int)optInFastD_Period < 1 || (int)optInFastD_Period > 100000 )
      return -1;
   if( (int)optInFastD_MAType == TA_INTEGER_DEFAULT || optInFastD_MAType == TA_MAType_DEFAULT )
      optInFastD_MAType = 0;
   else if( (int)optInFastD_MAType < TA_MATYPE_MIN || (int)optInFastD_MAType > TA_MATYPE_MAX )
      return -1;
   /* Account for the initial data needed for Fast-K. */
   retValue = optInFastK_Period - 1;
   /* Add the smoothing being done for Fast-D */
   retValue += TA_MA_Lookback(optInFastD_Period,optInFastD_MAType);
   return retValue;
}
