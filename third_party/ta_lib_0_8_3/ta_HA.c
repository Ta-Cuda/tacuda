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

TA_LIB_API TA_RetCode TA_HA( int    startIdx,
                             int    endIdx,
                             const double inOpen[],
                             const double inHigh[],
                             const double inLow[],
                             const double inClose[],
                             int          *outBegIdx,
                             int          *outNBElement,
                             double        outHAOpen[],
                             double        outHAHigh[],
                             double        outHALow[],
                             double        outHAClose[] )
{
   int i;
   int outIdx;
   int today;
   int lookbackTotal;
   double haOpen;
   double haClose;
   double haHigh;
   double haLow;
   double tempOpen;
   double tempHigh;
   double tempLow;
   double tempClose;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( !inOpen )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !inClose )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outHAOpen )
      return TA_BAD_PARAM;
   if( !outHAHigh )
      return TA_BAD_PARAM;
   if( !outHALow )
      return TA_BAD_PARAM;
   if( !outHAClose )
      return TA_BAD_PARAM;
   if( outHAOpen == outHAHigh || outHAOpen == outHALow || outHAOpen == outHAClose || outHAHigh == outHALow || outHAHigh == outHAClose || outHALow == outHAClose )
      return TA_BAD_PARAM;

   *outBegIdx= 0;
   *outNBElement= 0;
   lookbackTotal = TA_HA_Lookback();
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   /* The summation order ((O+H)+L)+C is the bit-exactness contract with every
    * external implementation of this indicator. TA_AVGPRICE is documented as
    * the same quantity but sums ((H+L)+C)+O, which is a different double on
    * some bars; the two are deliberately not unified.
    */
   today = startIdx - lookbackTotal;
   haOpen = (inOpen[today] + inClose[today]) / 2.0;
   haClose = (inOpen[today] + inHigh[today] + inLow[today] + inClose[today]) / 4.0;
   today += 1;
   /* Skip the unstable period. */
   i = lookbackTotal;
   while( i != 0 )
   {
      /* haOpen consumes the PREVIOUS candle on both sides of the midpoint, so
       * it must advance before haClose is overwritten. Swapping these two
       * still yields a plausible smoothed series.
       */
      haOpen = (haOpen + haClose) / 2.0;
      haClose = (inOpen[today] + inHigh[today] + inLow[today] + inClose[today]) / 4.0;
      today += 1;
      i -= 1;
   }
   tempHigh = inHigh[startIdx];
   tempLow = inLow[startIdx];
   /* The three-way extremum is spelled with plain comparisons, never the
    * max/min builtins: C's macros return their SECOND operand on a tie and
    * Rust's and Java's builtins do not, so a bar of signed zeros
    * emits different bytes per backend and the cross-language gate compares
    * bytes. Measured on (O,H,L,C) = (-0.0, +0.0, -0.0, -0.0).
    */
   haHigh = tempHigh;
   if( haOpen > haHigh )
   {
      haHigh = haOpen;
   }
   if( haClose > haHigh )
   {
      haHigh = haClose;
   }
   haLow = tempLow;
   if( haOpen < haLow )
   {
      haLow = haOpen;
   }
   if( haClose < haLow )
   {
      haLow = haClose;
   }
   outHAOpen[0] = haOpen;
   outHAHigh[0] = haHigh;
   outHALow[0] = haLow;
   outHAClose[0] = haClose;
   outIdx = 1;
   while( today <= endIdx )
   {
      /* Every price of the bar is read into a local before any output is
       * written, which is what lets an output alias any input: at startIdx 0
       * the store lands on the very slot the high and low were just read from.
       */
      tempOpen = inOpen[today];
      tempHigh = inHigh[today];
      tempLow = inLow[today];
      tempClose = inClose[today];
      haOpen = (haOpen + haClose) / 2.0;
      haClose = (tempOpen + tempHigh + tempLow + tempClose) / 4.0;
      haHigh = tempHigh;
      if( haOpen > haHigh )
      {
         haHigh = haOpen;
      }
      if( haClose > haHigh )
      {
         haHigh = haClose;
      }
      haLow = tempLow;
      if( haOpen < haLow )
      {
         haLow = haOpen;
      }
      if( haClose < haLow )
      {
         haLow = haClose;
      }
      outHAOpen[outIdx] = haOpen;
      outHAHigh[outIdx] = haHigh;
      outHALow[outIdx] = haLow;
      outHAClose[outIdx] = haClose;
      outIdx += 1;
      today += 1;
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_NOINLINE TA_LIB_API int TA_HA_Lookback( void )
{
   /* 13/9 is 1/ln(2) rounded up: the open is the only state, and it shows
    * the whole first difference at its first bar.
    */
   return TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_HA,Ha,(13 * 10 + 8) / 9,(13 * 19 + 8) / 9);
}
