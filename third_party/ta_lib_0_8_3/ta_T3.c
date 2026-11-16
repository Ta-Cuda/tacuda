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
TA_LIB_API TA_RetCode TA_T3( int    startIdx,
                             int    endIdx,
                             const double inReal[],
                             int optInTimePeriod,
                             double optInVFactor,
                             int          *outBegIdx,
                             int          *outNBElement,
                             double        outReal[] )
{
   int outIdx;
   int lookbackTotal;
   int today;
   int i;
   double k;
   double one_minus_k;
   double e1;
   double e2;
   double e3;
   double e4;
   double e5;
   double e6;
   double c1;
   double c2;
   double c3;
   double c4;
   double tempReal;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 5;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( optInVFactor == TA_REAL_DEFAULT )
      optInVFactor = 0.7;
   else if( !(optInVFactor >= 0e0 && optInVFactor <= 1e0) )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* For an explanation of this function, please read:
    *
    * Magazine articles written by Tim Tillson
    *
    * Essentially, a T3 of time serie 't' is:
    *   EMA1(x,Period) = EMA(x,Period)
    *   EMA2(x,Period) = EMA(EMA1(x,Period),Period)
    *   GD(x,Period,vFactor) = (EMA1(x,Period)*(1+vFactor)) - (EMA2(x,Period)*vFactor)
    *   T3 = GD (GD ( GD(t, Period, vFactor), Period, vFactor), Period, vFactor);
    *
    * T3 offers a moving average with less lags then the
    * traditional EMA.
    *
    * Do not confuse a T3 with EMA3. Both are called "Triple EMA"
    * in the litterature.
    */
   lookbackTotal = TA_T3_Lookback(optInTimePeriod,optInVFactor);
   if( startIdx <= lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      *outNBElement= 0;
      *outBegIdx= 0;
      return TA_SUCCESS;
   }
   /* No smoothing at period of 1: the output is a copy of the input
    * (same convention as TA_MA for every MAType). Explicit because the
    * coefficients below sum to 1 only in real arithmetic; going through
    * the math would leave ~1e-14 floating-point drift on every value.
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
   today = startIdx - lookbackTotal;
   k = 2.0 / (optInTimePeriod + 1.0);
   one_minus_k = 1.0 - k;
   /* Initialize e1 */
   tempReal = inReal[today++];
   for( i = optInTimePeriod - 1; i > 0; i -= 1 )
   {
      tempReal += inReal[today++];
   }
   e1 = tempReal / optInTimePeriod;
   /* Initialize e2 */
   tempReal = e1;
   for( i = optInTimePeriod - 1; i > 0; i -= 1 )
   {
      e1 = fma(one_minus_k, e1, k * inReal[today++]);
      tempReal += e1;
   }
   e2 = tempReal / optInTimePeriod;
   /* Initialize e3 */
   tempReal = e2;
   for( i = optInTimePeriod - 1; i > 0; i -= 1 )
   {
      e1 = fma(one_minus_k, e1, k * inReal[today++]);
      e2 = fma(one_minus_k, e2, k * e1);
      tempReal += e2;
   }
   e3 = tempReal / optInTimePeriod;
   /* Initialize e4 */
   tempReal = e3;
   for( i = optInTimePeriod - 1; i > 0; i -= 1 )
   {
      e1 = fma(one_minus_k, e1, k * inReal[today++]);
      e2 = fma(one_minus_k, e2, k * e1);
      e3 = fma(one_minus_k, e3, k * e2);
      tempReal += e3;
   }
   e4 = tempReal / optInTimePeriod;
   /* Initialize e5 */
   tempReal = e4;
   for( i = optInTimePeriod - 1; i > 0; i -= 1 )
   {
      e1 = fma(one_minus_k, e1, k * inReal[today++]);
      e2 = fma(one_minus_k, e2, k * e1);
      e3 = fma(one_minus_k, e3, k * e2);
      e4 = fma(one_minus_k, e4, k * e3);
      tempReal += e4;
   }
   e5 = tempReal / optInTimePeriod;
   /* Initialize e6 */
   tempReal = e5;
   for( i = optInTimePeriod - 1; i > 0; i -= 1 )
   {
      e1 = fma(one_minus_k, e1, k * inReal[today++]);
      e2 = fma(one_minus_k, e2, k * e1);
      e3 = fma(one_minus_k, e3, k * e2);
      e4 = fma(one_minus_k, e4, k * e3);
      e5 = fma(one_minus_k, e5, k * e4);
      tempReal += e5;
   }
   e6 = tempReal / optInTimePeriod;
   /* Skip the unstable period */
   while( today <= startIdx )
   {
      /* Do the calculation but do not write the output */
      e1 = fma(one_minus_k, e1, k * inReal[today++]);
      e2 = fma(one_minus_k, e2, k * e1);
      e3 = fma(one_minus_k, e3, k * e2);
      e4 = fma(one_minus_k, e4, k * e3);
      e5 = fma(one_minus_k, e5, k * e4);
      e6 = fma(one_minus_k, e6, k * e5);
   }
   /* Calculate the constants */
   tempReal = optInVFactor * optInVFactor;
   c1 = -(tempReal * optInVFactor);
   c2 = 3.0 * (tempReal - c1);
   c3 = -6.0 * tempReal - 3.0 * (optInVFactor - c1);
   c4 = fma(3.0, tempReal, fma(3.0, optInVFactor, 1.0) - c1);
   /* Write the first output */
   outIdx = 0;
   outReal[outIdx++] = fma(c4, e3, fma(c3, e4, fma(c1, e6, c2 * e5)));
   /* Calculate and output the remaining of the range. */
   while( today <= endIdx )
   {
      e1 = fma(one_minus_k, e1, k * inReal[today++]);
      e2 = fma(one_minus_k, e2, k * e1);
      e3 = fma(one_minus_k, e3, k * e2);
      e4 = fma(one_minus_k, e4, k * e3);
      e5 = fma(one_minus_k, e5, k * e4);
      e6 = fma(one_minus_k, e6, k * e5);
      outReal[outIdx++] = fma(c4, e3, fma(c3, e4, fma(c1, e6, c2 * e5)));
   }
   /* Indicates to the caller the number of output
    * successfully calculated.
    */
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_NOINLINE TA_LIB_API int TA_T3_Lookback( int optInTimePeriod, double optInVFactor )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 5;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   if( optInVFactor == TA_REAL_DEFAULT )
      optInVFactor = 0.7;
   else if( !(optInVFactor >= 0e0 && optInVFactor <= 1e0) )
      return -1;
   /* Sized by measurement on price series; it is not a bound. The six stages
    * share the pole (n-1)/(n+1) and are stepped together, so a seed can cancel
    * itself in the early outputs and show late: such a seed needs up to
    * 13*n bars at PREC_4 and 18.5*n at PREC_8.
    */
   return 6 * (optInTimePeriod - 1) + TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_T3,T3,(optInTimePeriod > 1) ? (11 * (4 + 4) * optInTimePeriod + 7) / 8 : 0,(optInTimePeriod > 1) ? (11 * (8 + 4) * optInTimePeriod + 7) / 8 : 0);
}
