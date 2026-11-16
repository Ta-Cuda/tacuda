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

TA_LIB_API TA_RetCode TA_RSI( int    startIdx,
                              int    endIdx,
                              const double inReal[],
                              int optInTimePeriod,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outReal[] )
{
   int outIdx;
   int today;
   int lookbackTotal;
   int i;
   double gainDelta;
   double invPeriod;
   double prevGain;
   double prevLoss;
   double prevValue;
   double tempValue1;
   double tempValue2;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* The following algorithm is base on the original
    * work from Wilder's and shall represent the
    * original idea behind the classic RSI.
    */
   *outBegIdx= 0;
   *outNBElement= 0;
   /* Adjust startIdx to account for the lookback period. */
   lookbackTotal = (int)TA_RSI_Lookback(optInTimePeriod);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   outIdx = 0;
   /* Index into the output. */
   invPeriod = 1.0 / (double)optInTimePeriod;
   /* Accumulate Wilder's "Average Gain" and "Average Loss"
    * among the initial period.
    */
   today = startIdx - lookbackTotal;
   prevValue = (double)inReal[today];
   prevGain = 0.0;
   prevLoss = 0.0;
   today = today + 1;
   for( i = optInTimePeriod; i > 0; i -= 1 )
   {
      tempValue1 = (double)inReal[today];
      today = today + 1;
      tempValue2 = tempValue1 - prevValue;
      prevValue = tempValue1;
      gainDelta = (tempValue2 > 0.0) ? tempValue2 : 0.0;
      prevGain += gainDelta;
      prevLoss += gainDelta - tempValue2;
   }
   /* Subsequent prevLoss and prevGain are smoothed
    * using the previous values (Wilder's approach).
    *  1) Multiply the previous by 'period-1'.
    *  2) Add today's gain to one accumulator and today's loss to the other,
    *     both unconditionally: gainDelta - tempValue2 is the exact loss delta
    *     for every finite tempValue2, so no branch on a 50/50 predicate.
    *  3) Scale by 1/'period'.
    */
   prevLoss *= invPeriod;
   prevGain *= invPeriod;
   /* Often documentation present the RSI calculation as follow:
    *    RSI = 100 - (100 / 1 + (prevGain/prevLoss))
    *
    * The following is equivalent:
    *    RSI = 100 * (prevGain/(prevGain+prevLoss))
    *
    * The second equation is used here for speed optimization.
    *
    * prevGain+prevLoss is a sum of non-negative magnitudes, so it is zero when
    * every change since the seed was exactly zero, or once both have decayed to
    * zero. Test it exactly, never against a fixed band: a gain carries the
    * quote unit, so a constant put against it zeroes a healthy oscillator for
    * an instrument quoted below it (issue #253).
    *
    * A zero total is 0/0, no gain against no loss, so it answers the neutral
    * 50 (issue #480). Keep it apart from the one-sided cases, which are 0 and
    * 100 and reach the division.
    */
   if( today > startIdx )
   {
      tempValue1 = prevGain + prevLoss;
      if( tempValue1 > 0.0 )
      {
         outReal[outIdx] = 100.0 * (prevGain / tempValue1);
         outIdx = outIdx + 1;
      } else 
      {
         outReal[outIdx] = 50.0;
         outIdx = outIdx + 1;
      }
   } else 
   {
      /* Skip the unstable period. Do the processing
       * but do not write it in the output.
       */
      while( today < startIdx )
      {
         tempValue1 = (double)inReal[today];
         tempValue2 = tempValue1 - prevValue;
         prevValue = tempValue1;
         prevLoss *= (double)(optInTimePeriod - 1);
         prevGain *= (double)(optInTimePeriod - 1);
         gainDelta = (tempValue2 > 0.0) ? tempValue2 : 0.0;
         prevGain += gainDelta;
         prevLoss += gainDelta - tempValue2;
         prevLoss *= invPeriod;
         prevGain *= invPeriod;
         today = today + 1;
      }
   }
   /* Unstable period skipped... now continue
    * processing if needed.
    */
   while( today <= endIdx )
   {
      tempValue1 = (double)inReal[today];
      today = today + 1;
      tempValue2 = tempValue1 - prevValue;
      prevValue = tempValue1;
      prevLoss *= (double)(optInTimePeriod - 1);
      prevGain *= (double)(optInTimePeriod - 1);
      gainDelta = (tempValue2 > 0.0) ? tempValue2 : 0.0;
      prevGain += gainDelta;
      prevLoss += gainDelta - tempValue2;
      prevLoss *= invPeriod;
      prevGain *= invPeriod;
      tempValue1 = prevGain + prevLoss;
      if( tempValue1 > 0.0 )
      {
         outReal[outIdx] = 100.0 * (prevGain / tempValue1);
         outIdx = outIdx + 1;
      } else 
      {
         outReal[outIdx] = 50.0;
         outIdx = outIdx + 1;
      }
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_NOINLINE TA_LIB_API int TA_RSI_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   return optInTimePeriod + TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_RSI,Rsi,((optInTimePeriod > 1) ? 10 * optInTimePeriod : 0),((optInTimePeriod > 1) ? 19 * optInTimePeriod : 0));
}
