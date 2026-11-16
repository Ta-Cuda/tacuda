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
TA_LIB_API TA_RetCode TA_RMA( int    startIdx,
                              int    endIdx,
                              const double inReal[],
                              int optInTimePeriod,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outReal[] )
{
   int i;
   int outIdx;
   int today;
   int lookbackTotal;
   int nbRMA;
   double prevRMA;
   double periodTotal;
   double wAlpha;
   double wBeta;

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

   *outBegIdx= 0;
   *outNBElement= 0;
   /* Adjust startIdx to account for the lookback period. */
   lookbackTotal = TA_RMA_Lookback(optInTimePeriod);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   /* wAlpha is derived FROM wBeta, never the reverse: only that order makes
    * wAlpha + wBeta exactly 1 (Sterbenz -- wBeta lands in [0.5, 1)), and it
    * measures closer to the exact recursion than the 1/period-first spelling
    * at nearly every period. The order is a gated contract, not a preference:
    * TA_RMA(TA_TRANGE(h,l,c),n) must equal TA_ATR(n) bit for bit.
    * The pair is exactly (1, 0) at period 1 -- hence no period-1 arm.
    */
   wBeta = (double)(optInTimePeriod - 1) / (double)optInTimePeriod;
   wAlpha = 1.0 - wBeta;
   /* The arithmetic order is the rest of that contract: the seed accumulates
    * from 0.0 in input order, and the smoothing step stays ONE statement --
    * splitting it unfuses the multiply-add and puts a second latency on the
    * recurrence's dependency chain.
    *
    * In-place (outReal being inReal) is supported: the output index never
    * passes the bar index of any remaining read.
    */
   today = startIdx - lookbackTotal;
   /* Seed with a simple average of the first 'period' values. */
   periodTotal = 0.0;
   i = optInTimePeriod;
   while( i-- > 0 )
   {
      periodTotal += inReal[today];
      today += 1;
   }
   prevRMA = periodTotal / optInTimePeriod;
   /* Skip the unstable period. */
   i = lookbackTotal - (optInTimePeriod - 1);
   while( i != 0 )
   {
      prevRMA = fma(wBeta, prevRMA, wAlpha * inReal[today]);
      today += 1;
      i -= 1;
   }
   /* Now start to write the final RMA in the caller
    * provided outReal.
    */
   outIdx = 1;
   outReal[0] = prevRMA;
   /* Now do the number of requested RMA. */
   nbRMA = endIdx - startIdx + 1;
   while( --nbRMA != 0 )
   {
      prevRMA = fma(wBeta, prevRMA, wAlpha * inReal[today]);
      outReal[outIdx++] = prevRMA;
      today += 1;
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_NOINLINE TA_LIB_API int TA_RMA_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 30;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   return optInTimePeriod - 1 + TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_RMA,Rma,((optInTimePeriod > 1) ? (10 * (2 * optInTimePeriod - 1) + 1) / 2 : 0),((optInTimePeriod > 1) ? (19 * (2 * optInTimePeriod - 1) + 1) / 2 : 0));
}
