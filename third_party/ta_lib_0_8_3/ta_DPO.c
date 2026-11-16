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

TA_LIB_API TA_RetCode TA_DPO( int    startIdx,
                              int    endIdx,
                              const double inReal[],
                              int optInTimePeriod,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outReal[] )
{
   double periodTotal;
   double tempReal;
   double dispVal;
   int i;
   int outIdx;
   int trailingIdx;
   int dispIdx;
   int lookbackTotal;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* The running sum is accumulated, snapshot and divided in exactly the order
    * ta_codegen/input/sma/sma.c uses, so this fused loop is bit-for-bit equal
    * to a TA_SMA anchored at the same startIdx. That equality is what makes
    * the composite differential in test_dpo.c a memcmp instead of a tolerance
    * argument, and it is lost by dividing once into a reciprocal or by
    * reordering the add/snapshot/subtract.
    */
   lookbackTotal = TA_DPO_Lookback(optInTimePeriod);
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
   periodTotal = 0.0;
   trailingIdx = startIdx - (optInTimePeriod - 1);
   dispIdx = startIdx - (optInTimePeriod / 2 + 1);
   i = trailingIdx;
   while( i < startIdx )
   {
      periodTotal += inReal[i];
      i = i + 1;
   }
   outIdx = 0;
   while( i <= endIdx )
   {
      periodTotal += inReal[i];
      i = i + 1;
      tempReal = periodTotal;
      periodTotal -= inReal[trailingIdx];
      trailingIdx = trailingIdx + 1;
      /* Both reads precede the store. Either cursor can EQUAL outIdx -- the
       * displaced one whenever startIdx equals the displacement, the trailing
       * one whenever startIdx sits at the lookback -- so a store hoisted above
       * them would read back what it had just overwritten when the caller
       * aliases outReal over inReal.
       */
      dispVal = inReal[dispIdx];
      dispIdx = dispIdx + 1;
      outReal[outIdx] = dispVal - tempReal / (double)optInTimePeriod;
      outIdx = outIdx + 1;
   }
   *outNBElement= outIdx;
   *outBegIdx= startIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_DPO_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   /* The max is load-bearing, not cosmetic. The displaced read reaches back
    * optInTimePeriod/2+1 bars, which exceeds the moving average's own
    * optInTimePeriod-1 at a period of 2, and a bare optInTimePeriod-1 would
    * then read inReal[-1].
    */
   return max(optInTimePeriod - 1,optInTimePeriod / 2 + 1);
}
