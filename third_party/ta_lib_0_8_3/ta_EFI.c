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
TA_LIB_API TA_RetCode TA_EFI( int    startIdx,
                              int    endIdx,
                              const double inClose[],
                              const double inVolume[],
                              int optInTimePeriod,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outReal[] )
{
   double emaBeta;
   double optInK_1;
   double tempReal;
   double prevMA;
   double prevClose;
   double force;
   int i;
   int today;
   int outIdx;
   int lookbackTotal;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 13;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inClose )
      return TA_BAD_PARAM;
   if( !inVolume )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* Alexander Elder's Force Index (Trading for a Living, 1993): the one-bar
    * close-to-close move weighted by that bar's volume, then smoothed with an
    * EMA. Elder's 2-period reading is the short-term form and 13 the
    * intermediate-term one -- that is the parameter, not a second formula.
    *
    *    force[t] = ( close[t] - close[t-1] ) * volume[t]
    *    EFI      = EMA( force, optInTimePeriod )
    *
    * The arithmetic below is ema.c's with inReal[t] replaced by force[t], kept
    * in exactly that shape on purpose: the seed accumulates from 0.0 in the
    * same order, and the recurrence is k*x + beta*prevMA with ema.c's k and
    * beta. That order IS the bit-exactness contract against the composed
    * reference in test_composite.c -- MOM, then MULT, then EMA -- so do not
    * tidy it. TRIX carries the same warning.
    *
    * Nothing on the data path divides by an input, so issue #112 is satisfied
    * structurally: a flat close gives force exactly 0.0 and output exactly
    * 0.0, and zero volume likewise. The only division is by the period, a
    * positive integer parameter.
    *
    * prevClose is carried in a scalar rather than re-read from inClose[t-1]
    * because the C API allows outReal to alias an input: at bar t the slot
    * holding close[t-1] may already have been overwritten by the output
    * written a bar earlier. cmou.c carries its trailing value for the same
    * reason.
    */
   /* Identify the minimum number of price bar needed
    * to calculate at least one output.
    */
   lookbackTotal = TA_EFI_Lookback(optInTimePeriod);
   /* After the lookback call: a double live across a call is saved and
    * restored around every fma call of the loops below, one more instruction
    * per bar.
    */
   emaBeta = (double)(optInTimePeriod - 1) / (double)(optInTimePeriod + 1);
   optInK_1 = 1.0 - emaBeta;
   if( emaBeta < 0.5 )
   {
      emaBeta = 1.0 - optInK_1;
   }
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
   /* No smoothing at a period of 1: the output is the raw Force Index.
    * Explicit because the recursion, at a k of 1.0 and a beta of 0.0, does
    * not keep the sign of a -0.0 force: a down bar on zero volume.
    */
   if( optInTimePeriod == 1 )
   {
      *outBegIdx= startIdx;
      outIdx = 0;
      today = startIdx;
      prevClose = inClose[today - 1];
      while( today <= endIdx )
      {
         force = (inClose[today] - prevClose) * inVolume[today];
         prevClose = inClose[today];
         outReal[outIdx] = force;
         outIdx = outIdx + 1;
         today = today + 1;
      }
      *outNBElement= outIdx;
      return TA_SUCCESS;
   }
   *outBegIdx= startIdx;
   /* The first EMA value is a simple average of the first 'period' force
    * values; it then seeds the recursion. This is ema.c's seeding applied
    * to the force series rather than to the input array.
    */
   today = startIdx - lookbackTotal + 1;
   prevClose = inClose[today - 1];
   i = optInTimePeriod;
   tempReal = 0.0;
   while( i-- > 0 )
   {
      force = (inClose[today] - prevClose) * inVolume[today];
      prevClose = inClose[today];
      tempReal += force;
      today = today + 1;
   }
   prevMA = tempReal / optInTimePeriod;
   while( today <= startIdx )
   {
      force = (inClose[today] - prevClose) * inVolume[today];
      prevClose = inClose[today];
      prevMA = fma(emaBeta, prevMA, optInK_1 * force);
      today = today + 1;
   }
   outReal[0] = prevMA;
   outIdx = 1;
   while( today <= endIdx )
   {
      force = (inClose[today] - prevClose) * inVolume[today];
      prevClose = inClose[today];
      prevMA = fma(emaBeta, prevMA, optInK_1 * force);
      outReal[outIdx] = prevMA;
      outIdx = outIdx + 1;
      today = today + 1;
   }
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_EFI_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 13;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   /* One bar forms the first close-to-close change. */
   return 1 + TA_EMA_Lookback(optInTimePeriod);
}
