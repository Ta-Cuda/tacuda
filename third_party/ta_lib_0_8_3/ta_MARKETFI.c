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

TA_LIB_API TA_RetCode TA_MARKETFI( int    startIdx,
                                   int    endIdx,
                                   const double inHigh[],
                                   const double inLow[],
                                   const double inVolume[],
                                   int          *outBegIdx,
                                   int          *outNBElement,
                                   double        outReal[] )
{
   int outIdx;
   int i;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !inVolume )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* Bill Williams' Market Facilitation Index: the price range a bar
    * travelled per unit of volume traded, i.e. how much movement the
    * market "facilitated" per tick.
    *
    *      MARKETFI = ( High - Low ) / Volume
    *
    * Stateless and per-bar: no seeding, no smoothing, no accumulator and
    * no unstable period, so the output for a bar never depends on where
    * the caller started the range.
    *
    * Retail material often abbreviates this "MFI" or "BW MFI". TA_MFI is
    * already the Money Flow Index, so this carries the name Tulip and
    * pandas-ta-classic use.
    */
   outIdx = 0;
   for( i = startIdx; i <= endIdx; i += 1 )
   {
      /* A zero-volume bar would divide by zero. Neither reference guards
       * it -- they emit +/-Inf, or NaN when the range is zero too -- but
       * issue #112 settled that a successful call never emits NaN or Inf,
       * so an untraded bar facilitated no movement and reports 0.
       *
       * The comparison is an exact != 0.0 rather than TA_IS_ZERO, whose
       * 1e-14 band is an absolute threshold and meaningless against an
       * unbounded volume scale. Same reasoning as the prevClose guard in
       * ta_codegen/input/nvi/nvi.c.
       */
      if( inVolume[i] != 0.0 )
      {
         outReal[outIdx++] = (inHigh[i] - inLow[i]) / inVolume[i];
      } else 
      {
         outReal[outIdx++] = 0.0;
      }
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_MARKETFI_Lookback( void )
{
   /* Each output depends only on its own bar, so nothing is consumed
    * before the first one can be produced.
    */
   return 0;
}
