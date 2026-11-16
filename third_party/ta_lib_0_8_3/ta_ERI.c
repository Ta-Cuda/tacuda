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
TA_LIB_API TA_RetCode TA_ERI( int    startIdx,
                              int    endIdx,
                              const double inHigh[],
                              const double inLow[],
                              const double inClose[],
                              int optInTimePeriod,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outBullPower[],
                              double        outBearPower[] )
{
   int outIdx;
   int today;
   int lookbackTotal;
   int i;
   double prevMA;
   double tempReal;
   double k;
   double beta;
   double tempHT;
   double tempLT;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 13;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !inClose )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outBullPower )
      return TA_BAD_PARAM;
   if( !outBearPower )
      return TA_BAD_PARAM;
   if( outBullPower == outBearPower )
      return TA_BAD_PARAM;

   /* Elder Ray Index (Alexander Elder, Trading for a Living, 1993): how far
    * the bar's extremes sit from one shared EMA of close.
    *
    *   Bull Power = High - EMA(Close, n)
    *   Bear Power = Low  - EMA(Close, n)
    *
    * One fused loop, not ema() + a combine map: a composed form cannot
    * stream (raw bar inputs are outside check_map_step's provenance), which
    * is the same reason ACCBANDS is fused. The EMA is ema.c op for op --
    * sequential seed sum from 0.0 then one divide, the unstable-period
    * warm-up consumed bar by bar -- so the differential against shipped
    * TA_EMA holds bitwise.
    *
    * No division in the per-bar map: no 0/0, no NaN path (#112 by
    * construction). Bull >= Bear on every bar since high >= low.
    */
   lookbackTotal = TA_ERI_Lookback(optInTimePeriod);
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
   /* Period 1: ema.c's explicit copy arm, kept here for the same reason it
    * exists there. At n == 1 the recursion below does not keep the sign
    * of a -0.0 close, so without this arm `High - TA_EMA(Close, 1)` is not
    * what this function returns. The unstable period still delays the first
    * output, through the shared lookback above.
    */
   if( optInTimePeriod == 1 )
   {
      outIdx = 0;
      today = startIdx;
      while( today <= endIdx )
      {
         tempHT = inHigh[today];
         tempLT = inLow[today];
         tempReal = inClose[today];
         outBullPower[outIdx] = tempHT - tempReal;
         outBearPower[outIdx] = tempLT - tempReal;
         outIdx += 1;
         today += 1;
      }
      *outBegIdx= startIdx;
      *outNBElement= outIdx;
      return TA_SUCCESS;
   }
   beta = (double)(optInTimePeriod - 1) / (double)(optInTimePeriod + 1);
   k = 1.0 - beta;
   if( beta < 0.5 )
   {
      beta = 1.0 - k;
   }
   /* Seed: ema.c's DEFAULT arm, op for op. */
   today = startIdx - lookbackTotal;
   i = optInTimePeriod;
   tempReal = 0.0;
   while( i-- > 0 )
   {
      tempReal += inClose[today++];
   }
   prevMA = tempReal / optInTimePeriod;
   /* The warm-up also consumes the EMA unstable period. */
   while( today <= startIdx )
   {
      prevMA = fma(beta, prevMA, k * inClose[today++]);
   }
   /* prevMA is the EMA at bar startIdx; today == startIdx + 1. Load the
    * extremes into temps BEFORE writing either output: with two outputs
    * over three inputs the caller may alias any pair, and the second write
    * must not read a clobbered bar.
    */
   tempHT = inHigh[startIdx];
   tempLT = inLow[startIdx];
   outBullPower[0] = tempHT - prevMA;
   outBearPower[0] = tempLT - prevMA;
   outIdx = 1;
   while( today <= endIdx )
   {
      prevMA = fma(beta, prevMA, k * inClose[today]);
      tempHT = inHigh[today];
      tempLT = inLow[today];
      outBullPower[outIdx] = tempHT - prevMA;
      outBearPower[outIdx] = tempLT - prevMA;
      outIdx += 1;
      today += 1;
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_ERI_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 13;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   /* Exactly the EMA of close underneath: its lookback, unstable period
    * included, is this function's lookback.
    */
   return TA_EMA_Lookback(optInTimePeriod);
}
