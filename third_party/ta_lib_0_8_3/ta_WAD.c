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

TA_LIB_API TA_RetCode TA_WAD( int    startIdx,
                              int    endIdx,
                              const double inHigh[],
                              const double inLow[],
                              const double inClose[],
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outReal[] )
{
   double sum;
   double prevClose;
   double close;
   double trueExtreme;
   int i;
   int outIdx;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !inClose )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* Williams' Accumulation/Distribution, in the form Steven Achelis
    * published (Technical Analysis from A to Z, 2nd ed., p.368) and the form
    * every modern vendor ships: each bar's close is measured against the TRUE
    * range extreme -- the previous close when it lies outside today's bar --
    * and the results accumulate.
    *
    *    TRH = max( prevClose, high )      TRL = min( prevClose, low )
    *    AD  = close - TRL   if close > prevClose
    *        = close - TRH   if close < prevClose
    *        = 0             if close == prevClose
    *    WAD = running sum of AD
    *
    * NO VOLUME IS CONSUMED, despite the name. Larry Williams' original
    * multiplies the move by volume; Achelis' modification drops it, the
    * industry attached Williams' name to the modification anyway, and Tulip,
    * pandas-ta-classic, cTrader, TC2000, WealthCharts and MultiCharts all ship
    * the no-volume form. Shipping the volume form under this name would
    * surprise every user, so this is the one place the usual "the original
    * author wins" rule is set aside. The volume-weighted series is a different
    * indicator. What is left once the multiplier is gone is a signed
    * close-to-close move clipped by the true-range extreme and accumulated --
    * momentum measured on the true range -- which is why this is grouped with
    * the other no-volume directional lines (PLUS_DM, MINUS_DM, BOP, WILLR)
    * rather than with AD/OBV.
    *
    * The three-way branch is written with plain > and < rather than any
    * epsilon: the flat arm must fire on exactly-equal consecutive closes and
    * on nothing else, which also keeps -0.0 and NaN behaviour identical
    * across the C, Rust, Java and .NET backends.
    *
    * prevClose is carried in a scalar, so outReal may alias any input: every
    * read of bar i happens before the store at outIdx <= i, and no earlier
    * bar is ever re-read.
    */
   sum = 0.0;
   outIdx = 0;
   /* The first bar of the requested range is measured against itself, i.e. it
    * contributes exactly 0.0. The accumulator therefore restarts wherever the
    * caller starts, which is why this function is flagged path_dependent.
    */
   prevClose = inClose[startIdx];
   for( i = startIdx; i <= endIdx; i += 1 )
   {
      close = inClose[i];
      if( close > prevClose )
      {
         trueExtreme = inLow[i];
         if( prevClose < trueExtreme )
         {
            trueExtreme = prevClose;
         }
         sum += close - trueExtreme;
      } else if( close < prevClose )
      {
         trueExtreme = inHigh[i];
         if( prevClose > trueExtreme )
         {
            trueExtreme = prevClose;
         }
         sum += close - trueExtreme;
      }
      outReal[outIdx] = sum;
      outIdx = outIdx + 1;
      prevClose = close;
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_WAD_Lookback( void )
{
   /* The first bar has no previous close, so it accumulates nothing and the
    * line starts at 0.0 -- the same convention as the other cumulative
    * lines in the tree: OBV, AD, NVI, PVI and PVT all return 0 here and emit
    * a seed value at startIdx. Tulip's ti_wad_start() returns 1 instead, so its
    * series is this one without the leading zero.
    */
   return 0;
}
