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

TA_LIB_API TA_RetCode TA_VORTEX( int    startIdx,
                                 int    endIdx,
                                 const double inHigh[],
                                 const double inLow[],
                                 const double inClose[],
                                 int optInTimePeriod,
                                 int          *outBegIdx,
                                 int          *outNBElement,
                                 double        outPlusVI[],
                                 double        outMinusVI[] )
{
   int outIdx;
   int today;
   int trailingIdx;
   int lookbackTotal;
   int i;
   int nullRun;
   int live;
   double sTR;
   double sVMP;
   double sVMM;
   double curTR;
   double curVMP;
   double curVMM;
   double trueRange;
   double tempDouble;
   double tempLT;
   double tempHT;
   double tempCY;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
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
   if( !outPlusVI )
      return TA_BAD_PARAM;
   if( !outMinusVI )
      return TA_BAD_PARAM;
   if( outPlusVI == outMinusVI )
      return TA_BAD_PARAM;

   /* Vortex Indicator (Botes & Siepman, TASC 28:1, Jan 2010): two lines,
    * each a rolling sum of "vortex movement" normalized by the rolling sum
    * of true range over the same optInTimePeriod bars.
    *
    *   TR[i]  = max( H[i]-L[i], |C[i-1]-H[i]|, |C[i-1]-L[i]| )   == TA_TRANGE
    *   VMP[i] = |H[i] - L[i-1]|
    *   VMM[i] = |L[i] - H[i-1]|
    *   +VI = SUM(VMP, n) / SUM(TR, n),  -VI = SUM(VMM, n) / SUM(TR, n)
    *
    * No smoothing, no recursion, nothing to seed. The TR expansion below is
    * TA_TRANGE's own operation order, bit for bit -- the differential test
    * composes TA_TRANGE + TA_SUM and asserts equality with memcmp.
    *
    * The trailing terms are recomputed from the inputs rather than carried
    * in a ring; the subtraction re-reads bars trailingIdx and trailingIdx-1,
    * both of which sit at or ahead of the output slot, which is why the
    * outputs are written LAST (see the loop comment).
    */
   lookbackTotal = TA_VORTEX_Lookback(optInTimePeriod);
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
   /* Prime the three window sums over the optInTimePeriod-1 terms before the
    * first output bar: [startIdx-optInTimePeriod+1, startIdx). Each term at
    * bar i reads bar i-1, so the earliest read is bar startIdx-optInTimePeriod
    * >= 0.
    */
   sTR = 0.0;
   sVMP = 0.0;
   sVMM = 0.0;
   nullRun = 0;
   for( i = startIdx - optInTimePeriod + 1; i < startIdx; i += 1 )
   {
      tempLT = inLow[i];
      tempHT = inHigh[i];
      tempCY = inClose[i - 1];
      trueRange = tempHT - tempLT;
      tempDouble = fabs(tempCY - tempHT);
      if( tempDouble > trueRange )
      {
         trueRange = tempDouble;
      }
      tempDouble = fabs(tempCY - tempLT);
      if( tempDouble > trueRange )
      {
         trueRange = tempDouble;
      }
      sTR += trueRange;
      sVMP += fabs(inHigh[i] - inLow[i - 1]);
      sVMM += fabs(inLow[i] - inHigh[i - 1]);
      live = (trueRange <= 0.0) ? 0 : 1;
      nullRun = (nullRun + 1) & (live - 1);
   }
   outIdx = 0;
   today = startIdx;
   trailingIdx = startIdx - optInTimePeriod + 1;
   while( today <= endIdx )
   {
      /* Add on today's terms. */
      tempLT = inLow[today];
      tempHT = inHigh[today];
      tempCY = inClose[today - 1];
      trueRange = tempHT - tempLT;
      tempDouble = fabs(tempCY - tempHT);
      if( tempDouble > trueRange )
      {
         trueRange = tempDouble;
      }
      tempDouble = fabs(tempCY - tempLT);
      if( tempDouble > trueRange )
      {
         trueRange = tempDouble;
      }
      sTR += trueRange;
      sVMP += fabs(inHigh[today] - inLow[today - 1]);
      sVMM += fabs(inLow[today] - inHigh[today - 1]);
      live = (trueRange <= 0.0) ? 0 : 1;
      nullRun = (nullRun + 1) & (live - 1);
      /* Once the whole window is flat, every TRUE-RANGE term in it is
       * provably zero (nullRun counts exactly that), so sTR's only content
       * is the running add/subtract's rounding residue -- purge it, and the
       * exact division gate below recognizes the case with no absolute
       * band. ONLY sTR: a zero true range (H == L == prevClose) does NOT
       * zero that bar's vortex terms, which read the PREVIOUS bar's
       * extremes -- a spread bar followed by a halt leaves |H - prevL| and
       * |L - prevH| alive inside the window, and zeroing the numerator sums
       * would poison both lines permanently (an unreachable negative -VI).
       * ULTOSC can reseed all its totals because its predicate covers both
       * of its per-bar terms; VORTEX's covers only the denominator's.
       */
      if( TA_UNLIKELY(nullRun >= optInTimePeriod) )
      {
         nullRun = optInTimePeriod;
         sTR = 0.0;
      }
      /* Record the current window sums, then retire the trailing bar's
       * terms so the sums are ready for the next iteration.
       */
      curTR = sTR;
      curVMP = sVMP;
      curVMM = sVMM;
      tempLT = inLow[trailingIdx];
      tempHT = inHigh[trailingIdx];
      tempCY = inClose[trailingIdx - 1];
      trueRange = tempHT - tempLT;
      tempDouble = fabs(tempCY - tempHT);
      if( tempDouble > trueRange )
      {
         trueRange = tempDouble;
      }
      tempDouble = fabs(tempCY - tempLT);
      if( tempDouble > trueRange )
      {
         trueRange = tempDouble;
      }
      sTR -= trueRange;
      sVMP -= fabs(inHigh[trailingIdx] - inLow[trailingIdx - 1]);
      sVMM -= fabs(inLow[trailingIdx] - inHigh[trailingIdx - 1]);
      trailingIdx += 1;
      /* Last operation is to write the outputs. Must be done after the
       * trailing bar has been fully consumed: the caller is allowed to pass
       * an output buffer aliasing any input, and the trailing reads above
       * touch bars trailingIdx-1 == outIdx and trailingIdx == outIdx+1 --
       * an emit-first order would have clobbered them (ULTOSC's own rule;
       * ACCBANDS' multi-output form).
       *
       * Zero-denominator gate, on the DENOMINATOR itself and exact: the
       * flat-run reseed above removes the running sums' residue, so
       * `curTR > 0.0` is a precise test -- ULTOSC gates its divisions the
       * same way after the same reseed. The flat-bar count alone is only a
       * proxy for sTR == 0 in exact arithmetic: floating-point absorption
       * can zero the running sum while the window still holds a live term
       * (a large spread swallows a 1-ULP one; the later subtract leaves
       * exactly 0.0 with nullRun far below n), and an ungated division
       * then emits NaN/Inf, which VORTEX does not declare. An absolute
       * TA_IS_ZERO band is no better: it zeroes legitimate ratios on any
       * instrument quoted below 1e-14, and the QUOTE-UNIT/SCALE gate
       * rejects it (VORTEX is homogeneous of degree 0).
       */
      if( curTR > 0.0 )
      {
         outPlusVI[outIdx] = curVMP / curTR;
         outMinusVI[outIdx] = curVMM / curTR;
      } else 
      {
         outPlusVI[outIdx] = 0.0;
         outMinusVI[outIdx] = 0.0;
      }
      outIdx += 1;
      today += 1;
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_VORTEX_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   /* Every per-bar term (TR, |H-prevL|, |L-prevH|) needs the prior bar, so
    * bar 0 is consumed exactly as TA_TRANGE consumes it, and the window then
    * needs optInTimePeriod terms: 1 + (optInTimePeriod - 1) = optInTimePeriod.
    * First valid output index is n, not n-1.
    */
   return optInTimePeriod;
}
