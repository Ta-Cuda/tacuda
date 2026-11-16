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

TA_LIB_API TA_RetCode TA_VWMA( int    startIdx,
                               int    endIdx,
                               const double inReal[],
                               const double inVolume[],
                               int optInTimePeriod,
                               int          *outBegIdx,
                               int          *outNBElement,
                               double        outReal[] )
{
   double sumPV;
   double sumV;
   double tempPV;
   double tempV;
   double tempReal;
   double trailingVolume;
   int i;
   int outIdx;
   int trailingIdx;
   int lookbackTotal;
   int zeroCount;

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
   if( !inVolume )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* Identify the minimum number of price bar needed
    * to calculate at least one output.
    */
   lookbackTotal = (int)(optInTimePeriod - 1);
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
   /* No smoothing at period of 1: the output is a copy of the input
    * (same convention as TA_MA for every MAType). Explicit because
    * (P*V)/V round-trips only ~97% of the time in IEEE double, and
    * because a lone zero volume must give the price, not NaN.
    */
   if( optInTimePeriod == 1 )
   {
      *outBegIdx= startIdx;
      outIdx = 0;
      i = startIdx;
      while( i <= (int)endIdx )
      {
         outReal[outIdx++] = inReal[i++];
      }
      *outNBElement= outIdx;
      return TA_SUCCESS;
   }
   /* Add-up the initial period, except for the last value.
    *
    * The price*volume product is kept in its own statement so no compiler may
    * contract it into an FMA: that would make this function disagree with the
    * Rust/Java backends under the cross-language bitwise gate, and with the
    * two-TA_SMA composite reference.
    */
   sumPV = 0.0;
   sumV = 0.0;
   trailingIdx = startIdx - lookbackTotal;
   /* Zero-volume bars in the window. Once they fill it both sums are exactly
    * zero, where add-then-subtract would leave the rounding residue of the bars
    * that departed, of either sign. The test is fabs(v) <= 0.0 rather than
    * == 0.0: the same result, NaN included, from one flag instead of two.
    */
   zeroCount = 0;
   i = trailingIdx;
   if( optInTimePeriod > 1 )
   {
      while( i < startIdx )
      {
         tempReal = inReal[i] * inVolume[i];
         sumPV += tempReal;
         sumV += inVolume[i];
         zeroCount += (fabs(inVolume[i]) <= 0.0) ? 1 : 0;
         i = i + 1;
      }
   }
   /* Proceed with the calculation for the requested range.
    * Note that this algorithm allows the inReal and
    * outReal to be the same buffer.
    */
   outIdx = 0;
   while( i <= endIdx )
   {
      tempReal = inReal[i] * inVolume[i];
      sumPV += tempReal;
      sumV += inVolume[i];
      zeroCount += (fabs(inVolume[i]) <= 0.0) ? 1 : 0;
      i = i + 1;
      /* Read the trailing values before writing the output, since the caller
       * may pass the same buffer for an input and the output.
       */
      trailingVolume = inVolume[trailingIdx];
      tempReal = inReal[trailingIdx] * trailingVolume;
      /* Each branch writes its own output: a branch that only zeroes the sums
       * is if-converted into a mask on their dependency chain, hint or not.
       */
      if( TA_UNLIKELY(zeroCount >= optInTimePeriod) )
      {
         /* Zero, then subtract the departing bar: a non-finite price times its
          * zero volume is NaN, not zero.
          */
         tempPV = 0.0;
         tempV = 0.0;
         sumPV = 0.0 - tempReal;
         sumV = 0.0 - trailingVolume;
         outReal[outIdx] = tempPV / (double)optInTimePeriod / (tempV / (double)optInTimePeriod);
      } else 
      {
         /* Snapshot both sums before removing the trailing bar, mirroring the
          * add-new / snapshot / subtract-old order of TA_SMA. Up to the first
          * dead window, that order is what makes this bit-identical to
          * SMA(inReal*inVolume)/SMA(inVolume).
          */
         tempPV = sumPV;
         tempV = sumV;
         sumPV -= tempReal;
         sumV -= trailingVolume;
         outReal[outIdx] = tempPV / (double)optInTimePeriod / (tempV / (double)optInTimePeriod);
      }
      zeroCount -= (fabs(trailingVolume) <= 0.0) ? 1 : 0;
      trailingIdx = trailingIdx + 1;
      outIdx = outIdx + 1;
   }
   /* All done. Indicate the output limits and return. */
   *outNBElement= outIdx;
   *outBegIdx= startIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_VWMA_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 30;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   return optInTimePeriod - 1;
}
