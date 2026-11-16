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
TA_LIB_API TA_RetCode TA_ZLEMA( int    startIdx,
                                int    endIdx,
                                const double inReal[],
                                int optInTimePeriod,
                                int          *outBegIdx,
                                int          *outNBElement,
                                double        outReal[] )
{
   double emaBeta;
   double optInK_1;
   double tempReal;
   double prevMA;
   int i;
   int today;
   int trailingIdx;
   int outIdx;
   int lag;
   int lookbackTotal;

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

   emaBeta = (double)(optInTimePeriod - 1) / (double)(optInTimePeriod + 1);
   optInK_1 = 1.0 - emaBeta;
   if( emaBeta < 0.5 )
   {
      emaBeta = 1.0 - optInK_1;
   }
   /* KEEP THIS ARITHMETIC EXACTLY AS WRITTEN -- the de-lag in one rounding
    * (2.0*c - l, not c + (c - l)), the seed sum accumulating from 0.0, and
    * k*v + beta*prevMA with ema.c's k and beta. Together they make ZLEMA
    * bit-for-bit equal to an EMA over a materialised de-lagged series, which
    * is the strongest gate this function has. Reordering any one breaks that equality silently, and
    * the de-lag spelling is worth more than rounding noise: c + (c - l) rounds
    * twice, which is 5e-12 relative where 2c - l cancels.
    */
   lag = (optInTimePeriod - 1) / 2;
   lookbackTotal = TA_ZLEMA_Lookback(optInTimePeriod);
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
   /* No smoothing at period of 1: the output is a copy of the input, the
    * convention TA_MA applies to every MAType. Explicit, because at period 1
    * lag is 0 and the recursion below, at a k of 1.0 and a beta of 0.0, does
    * not keep the sign of a -0.0 input. The unstable period still delays the
    * first output.
    */
   if( optInTimePeriod == 1 )
   {
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
   *outBegIdx= startIdx;
   /* In-place safe (outReal == inReal): both of a bar's reads precede its
    * write, and trailingIdx >= outIdx + optInTimePeriod - 1, so the trailing
    * read never reaches a slot already written.
    */
   trailingIdx = startIdx - lookbackTotal;
   today = trailingIdx + lag;
   i = optInTimePeriod;
   tempReal = 0.0;
   while( i-- > 0 )
   {
      tempReal += 2.0 * inReal[today] - inReal[trailingIdx];
      today += 1;
      trailingIdx += 1;
   }
   prevMA = tempReal / optInTimePeriod;
   while( today <= startIdx )
   {
      prevMA = fma(emaBeta, prevMA, optInK_1 * (2.0 * inReal[today] - inReal[trailingIdx]));
      today += 1;
      trailingIdx += 1;
   }
   outReal[0] = prevMA;
   outIdx = 1;
   while( today <= endIdx )
   {
      prevMA = fma(emaBeta, prevMA, optInK_1 * (2.0 * inReal[today] - inReal[trailingIdx]));
      today += 1;
      trailingIdx += 1;
      outReal[outIdx++] = prevMA;
   }
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_ZLEMA_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 30;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   /* ZLEMA owns no TA_FUNC_UNST_ id. It borrows EMA's through this call, which
    * is why zlema.yaml must not declare `unstable_period`.
    */
   return (optInTimePeriod - 1) / 2 + TA_EMA_Lookback(optInTimePeriod);
}
