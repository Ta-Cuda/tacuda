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
TA_LIB_API TA_RetCode TA_EMA( int    startIdx,
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
   int outIdx;
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

   /* Identify the minimum number of price bar needed
    * to calculate at least one output.
    */
   lookbackTotal = TA_EMA_Lookback(optInTimePeriod);
   /* After the lookback call: a double live across a call is saved and
    * restored around every fma call of the loops below, one more instruction
    * per bar.
    *
    * emaBeta + optInK_1 must be exactly 1.0, or a flat input drifts off
    * its level. Each subtraction is exact only from an operand in
    * [0.5,1): at a period of 2 that is optInK_1, above it emaBeta.
    *
    * Above it emaBeta stays as the divide wrote it: the second subtraction
    * would not change a bit, and a register last written by a subtraction
    * costs each FMA reading it one more cycle on Intel P-cores.
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
   /* No smoothing at period of 1: the output is a copy of the input
    * (same convention as TA_MA for every MAType). Explicit because the
    * recursion below, at a k of 1.0 and a beta of 0.0, does not keep the
    * sign of a -0.0 input. The unstable period still delays the first
    * output.
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
   /* Do the EMA calculation using tight loops. */
   today = startIdx - lookbackTotal;
   i = optInTimePeriod;
   tempReal = 0.0;
   while( i-- > 0 )
   {
      tempReal += inReal[today++];
   }
   prevMA = tempReal / optInTimePeriod;
   while( today <= startIdx )
   {
      prevMA = fma(emaBeta, prevMA, optInK_1 * inReal[today++]);
   }
   outReal[0] = prevMA;
   outIdx = 1;
   while( today <= endIdx )
   {
      prevMA = fma(emaBeta, prevMA, optInK_1 * inReal[today++]);
      outReal[outIdx++] = prevMA;
   }
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_NOINLINE TA_LIB_API int TA_EMA_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 30;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   return optInTimePeriod - 1 + TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_EMA,Ema,((optInTimePeriod > 1) ? (10 * optInTimePeriod + 1) / 2 : 0),((optInTimePeriod > 1) ? (19 * optInTimePeriod + 1) / 2 : 0));
}
