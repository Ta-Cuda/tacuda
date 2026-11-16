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

TA_LIB_API TA_RetCode TA_ADR( int    startIdx,
                              int    endIdx,
                              const double inHigh[],
                              const double inLow[],
                              int optInTimePeriod,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outReal[] )
{
   double periodTotal;
   double tempReal;
   int i;
   int outIdx;
   int trailingIdx;
   int lookbackTotal;

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
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* This is ta_codegen/input/sma/sma.c with inReal[x] replaced by
    * (inHigh[x] - inLow[x]), and deliberately nothing else -- the same
    * running-sum order, the same read-before-write of the trailing term, and
    * the same divide by the period. Keeping the arithmetic identical is what
    * makes the composed reference in test_adr.c -- TA_SUB followed by TA_SMA
    * -- bit-exact rather than merely close.
    *
    * In particular the last statement divides; it does NOT multiply by a
    * precomputed 1/period, which would cost up to one ULP against TA_SMA and
    * silently downgrade that memcmp into a tolerance argument.
    *
    * Averaging the ranges is not the same computation as differencing two
    * averages: SMA(high) - SMA(low) is algebraically equal but subtracts two
    * price-magnitude means to reach a range-magnitude answer, so it inherits
    * the larger scale's rounding.
    */
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
   /* Do the MA calculation using tight loops. */
   /* Add-up the initial period, except for the last value. */
   periodTotal = 0.0;
   trailingIdx = startIdx - lookbackTotal;
   i = trailingIdx;
   if( optInTimePeriod > 1 )
   {
      while( i < startIdx )
      {
         periodTotal += (double)(inHigh[i] - inLow[i]);
         i = i + 1;
      }
   }
   /* Proceed with the calculation for the requested range.
    * Note that this algorithm allows outReal to be the same
    * buffer as either input.
    */
   outIdx = 0;
   while( i <= endIdx )
   {
      periodTotal += (double)(inHigh[i] - inLow[i]);
      i = i + 1;
      tempReal = periodTotal;
      periodTotal -= (double)(inHigh[trailingIdx] - inLow[trailingIdx]);
      trailingIdx = trailingIdx + 1;
      outReal[outIdx] = tempReal / (double)optInTimePeriod;
      outIdx = outIdx + 1;
   }
   /* All done. Indicate the output limits and return. */
   *outNBElement= outIdx;
   *outBegIdx= startIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_ADR_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   return optInTimePeriod - 1;
}
