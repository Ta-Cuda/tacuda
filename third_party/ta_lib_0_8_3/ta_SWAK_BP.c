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
TA_LIB_API TA_RetCode TA_SWAK_BP( int    startIdx,
                                  int    endIdx,
                                  const double inReal[],
                                  int optInTimePeriod,
                                  double optInDelta,
                                  int          *outBegIdx,
                                  int          *outNBElement,
                                  double        outReal[] )
{
   int i;
   int outIdx;
   int today;
   int lookbackTotal;
   double w;
   double beta;
   double t;
   double abp;
   double c0;
   double a1;
   double a2;
   double x0;
   double x1;
   double x2;
   double y;
   double y1;
   double y2;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 5 || (int)optInTimePeriod > 2000 )
      return TA_BAD_PARAM;
   if( optInDelta == TA_REAL_DEFAULT )
      optInDelta = 0.1;
   else if( !(optInDelta >= 5e-2 && optInDelta <= 5e-1) )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   *outBegIdx= 0;
   *outNBElement= 0;
   lookbackTotal = TA_SWAK_BP_Lookback(optInTimePeriod,optInDelta);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   /* The band-pass row (Ehlers, "Swiss Army Knife Indicator", Figure 5).
    * The paper's 360/P is a full turn, so 2*pi/P; beta is the cosine of the
    * centre frequency and t is the half-bandwidth angle.
    */
   w = 2.0 * 3.141592653589793 / (double)optInTimePeriod;
   beta = cos(w);
   t = 4.0 * 3.141592653589793 * optInDelta / (double)optInTimePeriod;
   /* abp is written as (1 - sin t)/cos t rather than the paper's
    * gamma - sqrt(gamma^2 - 1) with gamma = 1/cos t. The two are equal for
    * 0 < t < pi/2, which this function's ranges guarantee (t <= 0.4*pi), but
    * the published form cancels as t -> 0 because gamma^2 - 1 goes as t^2.
    * Against a 60-digit reference the published form is already 4.5e-13 off
    * at the period cap; this one is not, and that is what lets the cap be
    * 2000 rather than 1000.
    */
   abp = (1.0 - sin(t)) / cos(t);
   /* The numerator (1, 0, -1) is zero both on a constant and on a Nyquist
    * alternation, so this row answers 0 at DC and at Nyquist, and exactly 1
    * with zero phase at the centre period itself.
    */
   c0 = (1.0 - abp) / 2.0;
   a1 = beta * (1.0 + abp);
   a2 = -abp;
   today = startIdx - lookbackTotal;
   /* Start in the steady state of a constant input equal to the first bar: a
    * band-pass of a constant is 0, so both output slots start there while both
    * input slots hold the bar. The input slots are carried in locals and never
    * re-read from inReal, because outReal may alias it.
    */
   x1 = inReal[today];
   x2 = x1;
   y1 = 0.0;
   y2 = 0.0;
   /* Skip the unstable period: run the recurrence but publish nothing. */
   i = lookbackTotal;
   while( i != 0 )
   {
      x0 = inReal[today];
      y = fma(a1, y1, fma(a2, y2, c0 * (x0 - x2)));
      x2 = x1;
      x1 = x0;
      y2 = y1;
      y1 = y;
      today += 1;
      i -= 1;
   }
   outIdx = 0;
   while( today <= endIdx )
   {
      /* a1*y1 stays the outermost term. y1 is the newest output, so the bar
       * rate is the latency of whatever y1 crosses to become y: outermost, that
       * is one fused step. Nested inside, it is three, and every backend's last
       * bit moves with it.
       */
      x0 = inReal[today];
      y = fma(a1, y1, fma(a2, y2, c0 * (x0 - x2)));
      x2 = x1;
      x1 = x0;
      y2 = y1;
      y1 = y;
      outReal[outIdx] = y;
      outIdx = outIdx + 1;
      today += 1;
   }
   *outNBElement= outIdx;
   *outBegIdx= startIdx;
   return TA_SUCCESS;
}

TA_NOINLINE TA_LIB_API int TA_SWAK_BP_Lookback( int optInTimePeriod, double optInDelta )
{
   int count4;
   int count8;
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 5 || (int)optInTimePeriod > 2000 )
      return -1;
   if( optInDelta == TA_REAL_DEFAULT )
      optInDelta = 0.1;
   else if( !(optInDelta >= 5e-2 && optInDelta <= 5e-1) )
      return -1;
   /* ceil( (K+1)*P / (6*delta) ) per level, in this order of operations: the
    * count is defined as this double expression, not as the real quotient.
    */
   count4 = (int)ceil((double)(11 * optInTimePeriod) / (6.0 * optInDelta));
   count8 = (int)ceil((double)(20 * optInTimePeriod) / (6.0 * optInDelta));
   /* No structural lookback: the two input slots and the two output slots are
    * seeded from the first bar rather than read from before it, and there is
    * no callee whose lookback could be inherited.
    */
   return TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_SWAK_BP,Swak_bp,count4,count8);
}
