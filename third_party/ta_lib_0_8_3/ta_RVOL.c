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

TA_LIB_API TA_RetCode TA_RVOL( int    startIdx,
                               int    endIdx,
                               const double inVolume[],
                               int optInTimePeriod,
                               int          *outBegIdx,
                               int          *outNBElement,
                               double        outReal[] )
{
   double periodTotal;
   double baseline;
   double todayVolume;
   double trailingVolume;
   int i;
   int outIdx;
   int trailingIdx;
   int lookbackTotal;
   int nullRun;
   int liveIn;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inVolume )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* One bar more than a moving average of the same period: today is excluded
    * from its own baseline.
    */
   lookbackTotal = (int)optInTimePeriod;
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   if( startIdx > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   periodTotal = 0.0;
   trailingIdx = startIdx - lookbackTotal;
   /* Consecutive zero-volume bars, ending at the last one added. Once they
    * fill a window the total is exactly zero, where add-then-subtract would
    * leave the rounding residue of the volumes that departed, of either sign.
    * The run advances through a mask: as a select or a product gcc compiles
    * it to a branch that mispredicts on scattered zero volumes. The test is
    * fabs(v) <= 0.0 rather than == 0.0: the same result, NaN included, from
    * one flag instead of two.
    */
   nullRun = 0;
   i = trailingIdx;
   while( i < startIdx )
   {
      periodTotal += (double)inVolume[i];
      liveIn = (fabs(inVolume[i]) <= 0.0) ? 0 : 1;
      nullRun = (nullRun + 1) & (liveIn - 1);
      i = i + 1;
   }
   outIdx = 0;
   while( i <= endIdx )
   {
      /* Drop the trailing bar BEFORE adding today's. Up to the first dead
       * window, that order makes each baseline bit-identical to the moving
       * average of the same period at the previous bar; the reverse order
       * differs only in the last ulp, so no tolerance can tell the two apart.
       */
      baseline = periodTotal / (double)optInTimePeriod;
      trailingVolume = (double)inVolume[trailingIdx];
      trailingIdx = trailingIdx + 1;
      todayVolume = (double)inVolume[i];
      i = i + 1;
      /* After both reads (input and output may be one buffer) and before the
       * total moves, or the divide keeps the old total live across the reset.
       */
      outReal[outIdx] = todayVolume / baseline;
      outIdx = outIdx + 1;
      periodTotal -= trailingVolume;
      periodTotal += todayVolume;
      liveIn = (fabs(todayVolume) <= 0.0) ? 0 : 1;
      nullRun = (nullRun + 1) & (liveIn - 1);
      if( TA_UNLIKELY(nullRun >= optInTimePeriod) )
      {
         periodTotal = 0.0;
      }
   }
   *outNBElement= outIdx;
   *outBegIdx= startIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_RVOL_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   return optInTimePeriod;
}
