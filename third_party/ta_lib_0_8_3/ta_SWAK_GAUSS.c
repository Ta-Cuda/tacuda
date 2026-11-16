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
TA_LIB_API TA_RetCode TA_SWAK_GAUSS( int    startIdx,
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
   double s;
   double b2p;
   double a2p;
   double om;
   double c0;
   double a1;
   double a2;
   double y;
   double y1;
   double y2;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 10000 )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   *outBegIdx= 0;
   *outNBElement= 0;
   lookbackTotal = TA_SWAK_GAUSS_Lookback(optInTimePeriod);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   /* The two-pole alpha of Ehlers' "Swiss Army Knife Indicator", Figure 5. The
    * paper's 360/P is a full turn, so 2*pi/P.
    *
    * Keep 1 - cos(w) as 2*sin(w/2)^2: the subtraction cancels more of b2p's
    * digits the longer the period, and the cutoff drifts with them.
    */
   s = sin(3.141592653589793 / (double)optInTimePeriod);
   b2p = 2.415 * (2.0 * s * s);
   a2p = -b2p + sqrt(fma(b2p, b2p, 2.0 * b2p));
   /* The Gaussian row of Figure 5: numerator a2p^2 on the bar alone, no
    * x[i-1] or x[i-2] term. Its DC gain is 1, so the line sits on price.
    */
   om = 1.0 - a2p;
   c0 = a2p * a2p;
   a1 = 2.0 * om;
   a2 = -(om * om);
   today = startIdx - lookbackTotal;
   /* Start in the steady state of a constant input equal to the first bar: a
    * DC-gain-1 row answers that constant, so both output slots hold it. The
    * input slots would hold it too, but this row's b1 and b2 are zero and
    * never reads them -- which is also why outReal may alias inReal here.
    */
   y1 = inReal[today];
   y2 = y1;
   /* Skip the unstable period: run the recurrence but publish nothing. */
   i = lookbackTotal;
   while( i != 0 )
   {
      y = fma(a1, y1, fma(a2, y2, c0 * inReal[today]));
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
      y = fma(a1, y1, fma(a2, y2, c0 * inReal[today]));
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

TA_NOINLINE TA_LIB_API int TA_SWAK_GAUSS_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 10000 )
      return -1;
   /* No structural lookback. Every term of the recurrence exists at the first
    * bar -- the two history slots are seeded from that bar rather than read
    * from before it -- and there is no callee whose lookback could be
    * inherited, so the function's own unstable period is the whole of it.
    */
   return TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_SWAK_GAUSS,Swak_gauss,(((10 + 3) * (optInTimePeriod + 2) + 8) / 9),(((19 + 3) * (optInTimePeriod + 2) + 8) / 9));
}
