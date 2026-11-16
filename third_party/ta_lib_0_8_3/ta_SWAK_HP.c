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
TA_LIB_API TA_RetCode TA_SWAK_HP( int    startIdx,
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
   double h;
   double sh;
   double ch;
   double a1p;
   double c0;
   double a1;
   double x0;
   double x1;
   double y;
   double y1;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 5 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   *outBegIdx= 0;
   *outNBElement= 0;
   lookbackTotal = TA_SWAK_HP_Lookback(optInTimePeriod);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   /* The one-pole alpha (Ehlers, "Swiss Army Knife Indicator", Figure 5).
    * The paper says this one "is computed exactly the same as it is for the
    * EMA" -- meaning the cutoff-period formula below, not TA_EMA's
    * 2/(n+1). The paper's 360/P is a full turn, so 2*pi/P.
    *
    * The period range starts at 5 because of what this expression does below
    * it, not for taste: at P = 4 it is 0/0, and what the doubles make of that
    * is no filter. At P = 2, cos(w) is -1 and c0 is 0, a dead filter. No
    * contiguous range below 5 avoids both.
    *
    * Keep it in the half angle h = w/2, where it is 2*sin h*(cos h - sin h)
    * over 1 - 2*sin(h)^2. Written in w, cos(w) - 1 cancels more of the
    * numerator's digits the longer the period, and the cutoff drifts with
    * them.
    */
   h = 3.141592653589793 / (double)optInTimePeriod;
   sh = sin(h);
   ch = cos(h);
   a1p = 2.0 * sh * (ch - sh) / (1.0 - 2.0 * sh * sh);
   /* The high-pass row: a (1, -1) numerator, so its DC gain is 0 and the line
    * is centred on zero rather than on price.
    */
   c0 = 1.0 - a1p / 2.0;
   a1 = 1.0 - a1p;
   today = startIdx - lookbackTotal;
   /* Start in the steady state of a constant input equal to the first bar: a
    * DC-gain-0 row answers 0 for a constant, so the output slot starts there
    * while the input slot holds the bar. The input slot is carried in a local
    * and never re-read from inReal, because outReal may alias it.
    */
   x1 = inReal[today];
   y1 = 0.0;
   /* Skip the unstable period: run the recurrence but publish nothing. */
   i = lookbackTotal;
   while( i != 0 )
   {
      x0 = inReal[today];
      y = fma(a1, y1, c0 * (x0 - x1));
      x1 = x0;
      y1 = y;
      today += 1;
      i -= 1;
   }
   outIdx = 0;
   while( today <= endIdx )
   {
      /* a1*y1 stays the outermost term. y1 is the newest output, so the bar
       * rate is the latency of whatever y1 crosses to become y: outermost, that
       * is one fused step. Nested inside, it is two, and every backend's last
       * bit moves with it.
       */
      x0 = inReal[today];
      y = fma(a1, y1, c0 * (x0 - x1));
      x1 = x0;
      y1 = y;
      outReal[outIdx] = y;
      outIdx = outIdx + 1;
      today += 1;
   }
   *outNBElement= outIdx;
   *outBegIdx= startIdx;
   return TA_SUCCESS;
}

TA_NOINLINE TA_LIB_API int TA_SWAK_HP_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 5 || (int)optInTimePeriod > 100000 )
      return -1;
   /* No structural lookback: the one input slot and the one output slot are
    * seeded from the first bar rather than read from before it, and there is
    * no callee whose lookback could be inherited.
    */
   return TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_SWAK_HP,Swak_hp,(10 * optInTimePeriod + 5) / 6,(19 * optInTimePeriod + 5) / 6);
}
