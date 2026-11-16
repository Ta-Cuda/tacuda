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

TA_LIB_API TA_RetCode TA_PVI( int    startIdx,
                              int    endIdx,
                              const double inClose[],
                              const double inVolume[],
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outReal[] )
{
   int i;
   int outIdx;
   double prevPVI;
   double prevClose;
   double prevVolume;
   double tempClose;
   double tempVolume;
   double tempPVI;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( !inClose )
      return TA_BAD_PARAM;
   if( !inVolume )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* The index is a running cumulative value seeded at 1000, updated only on
    * bars whose volume increased versus the prior bar (Positive Volume).
    */
   prevPVI = 1000.0;
   prevClose = inClose[startIdx];
   prevVolume = inVolume[startIdx];
   outIdx = 0;
   for( i = startIdx; i <= endIdx; i += 1 )
   {
      tempClose = inClose[i];
      tempVolume = inVolume[i];
      /* prevClose != 0 guards the percentage-change division: a zero previous
       * close is a degenerate input that would otherwise emit NaN/Inf; carry
       * the index forward unchanged instead. Never triggers on real prices.
       */
      if( tempVolume > prevVolume && prevClose != 0.0 )
      {
         /* The index is a running product, so it has no upper bound: enough
          * compounding gains push it past the largest double. Keep the last
          * representable value instead of writing +/-Inf, which no caller can
          * chart and which poisons every arithmetic downstream of it. Real
          * price series never come close.
          *
          * Written as a compound assignment on the copy, exactly as the update
          * was before the guard: spelling it `a + r*a` would match the FMA
          * fusion detector and silently re-round every bar, not just the
          * overflowing one.
          */
         tempPVI = prevPVI;
         tempPVI += (tempClose - prevClose) / prevClose * tempPVI;
         if( TA_IS_FINITE(tempPVI) )
         {
            prevPVI = tempPVI;
         }
      }
      outReal[outIdx++] = prevPVI;
      prevClose = tempClose;
      prevVolume = tempVolume;
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_PVI_Lookback( void )
{
   /* This function have no lookback needed. */
   return 0;
}
