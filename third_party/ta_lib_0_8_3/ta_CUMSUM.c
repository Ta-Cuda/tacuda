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

TA_LIB_API TA_RetCode TA_CUMSUM( int    startIdx,
                                 int    endIdx,
                                 const double inReal[],
                                 int          *outBegIdx,
                                 int          *outNBElement,
                                 double        outReal[] )
{
   double total;
   int i;
   int outIdx;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* Running total from the ANCHOR bar forward: the accumulator re-seeds at
    * startIdx, exactly as every shipped path-dependent accumulator does
    * (ad.c anchors at startIdx with ad = 0.0, obv.c seeds at
    * inVolume[startIdx]) -- TA_CUMSUM(3, 7, x)[0] is x[3], NOT sum(x[0..3]).
    * "From bar 0" could only mean "from the start of whatever buffer the
    * caller passed", under which a composed call handed inReal + off silently
    * changes answer; the path_dependent flag exists to declare this class
    * (issue #372). Left-to-right, one double, no compensation -- ad.c's own
    * plain += convention, and what both external oracles compute bit-exactly.
    *
    * Two statements, not outReal[outIdx] = (total += ...): a value-producing
    * compound assignment appears in zero input files, and Rust has no
    * assignment expression, so the one-liner cannot lower to a required
    * backend.
    */
   total = 0.0;
   for( i = startIdx, outIdx = 0; i <= endIdx; i += 1, outIdx += 1 )
   {
      total += inReal[i];
      outReal[outIdx] = total;
   }
   *outNBElement= outIdx;
   *outBegIdx= startIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_CUMSUM_Lookback( void )
{
   return 0;
}
