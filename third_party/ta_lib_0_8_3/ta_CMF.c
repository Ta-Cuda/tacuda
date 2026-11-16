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

TA_LIB_API TA_RetCode TA_CMF( int    startIdx,
                              int    endIdx,
                              const double inHigh[],
                              const double inLow[],
                              const double inClose[],
                              const double inVolume[],
                              int optInTimePeriod,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outReal[] )
{
   double sumMFV;
   double sumVol;
   double high;
   double low;
   double close;
   double tmp;
   double mfv;
   int lookbackTotal;
   int outIdx;
   int i;
   int today;
   int nullRun;
   int liveVol;
   double local_mfv_flow[50];
   double *mfv_flow = &local_mfv_flow[0];
   double local_mfv_volume[50];
   double *mfv_volume = &local_mfv_volume[0];
   int mfv_Idx;
   int maxIdx_mfv;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !inClose )
      return TA_BAD_PARAM;
   if( !inVolume )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* Both the per-bar money flow volume and the volume that produced it are
    * carried in the circular buffer. Keeping the volume here rather than
    * re-reading inVolume[] at the trailing index is what makes outReal safe to
    * alias any input: once a bar has been consumed it is never read again.
    */
   /* Id, Type, Static Size */
   *outBegIdx= 0;
   *outNBElement= 0;
   /* Identify the minimum number of price bar needed
    * to calculate at least one output.
    */
   lookbackTotal = optInTimePeriod - 1;
   /* Move up the start index if there is not
    * enough initial data.
    */
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   if( optInTimePeriod < 1 ) return TA_INTERNAL_ERROR(314);
   if( (int)optInTimePeriod > (int)(sizeof(local_mfv_flow)/sizeof(double)) )
   {
      mfv_flow = TA_Malloc( sizeof(double)*optInTimePeriod );
      if( !mfv_flow )
      {
         return TA_ALLOC_ERR;
      }
      mfv_volume = TA_Malloc( sizeof(double)*optInTimePeriod );
      if( !mfv_volume )
      {
         TA_Free( mfv_flow );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      mfv_flow = &local_mfv_flow[0];
      mfv_volume = &local_mfv_volume[0];
   }
   maxIdx_mfv = (optInTimePeriod-1);
   mfv_Idx = 0;
   outIdx = 0;
   /* Accumulate the money flow volume and the volume over the first
    * complete window, filling the circular buffer as we go.
    *
    * The per-bar multiplier is written exactly as in ta_AD.c so that the
    * Chaikin money flow volume has one definition in the library.
    */
   today = startIdx - lookbackTotal;
   sumMFV = 0.0;
   sumVol = 0.0;
   /* Consecutive zero-volume bars. Once they fill a window both sums are
    * exactly zero, where add-then-subtract would leave the rounding residue of
    * the bars that departed, of either sign. The run advances through a mask:
    * as a select or a product gcc compiles it to a branch that mispredicts on
    * scattered zero volumes. The test is fabs(v) <= 0.0 rather than == 0.0:
    * the same result, NaN included, from one flag instead of two.
    */
   nullRun = 0;
   for( i = optInTimePeriod; i > 0; i -= 1 )
   {
      high = inHigh[today];
      low = inLow[today];
      close = inClose[today];
      tmp = high - low;
      if( tmp > 0.0 )
      {
         mfv = (close - low - (high - close)) / tmp * inVolume[today];
      } else 
      {
         mfv = 0.0;
      }
      mfv_flow[mfv_Idx] = mfv;
      mfv_volume[mfv_Idx] = inVolume[today];
      sumMFV += mfv;
      sumVol += inVolume[today];
      liveVol = (fabs(inVolume[today]) <= 0.0) ? 0 : 1;
      nullRun = (nullRun + 1) & (liveVol - 1);
      today += 1;
      mfv_Idx++;
      if( mfv_Idx > maxIdx_mfv ) mfv_Idx = 0;
   }
   /* The first full window is complete: emit its output for startIdx here,
    * then slide the window over the remaining bars below.
    *
    * A window whose volume is entirely zero has no money flow to distribute;
    * report 0.0 rather than propagating a division by zero (issue #112).
    */
   if( sumVol > 0.0 )
   {
      outReal[outIdx++] = sumMFV / sumVol;
   } else 
   {
      outReal[outIdx++] = 0.0;
   }
   /* Now continue processing the remaining bars. */
   while( today <= endIdx )
   {
      sumMFV -= mfv_flow[mfv_Idx];
      sumVol -= mfv_volume[mfv_Idx];
      high = inHigh[today];
      low = inLow[today];
      close = inClose[today];
      tmp = high - low;
      if( tmp > 0.0 )
      {
         mfv = (close - low - (high - close)) / tmp * inVolume[today];
      } else 
      {
         mfv = 0.0;
      }
      mfv_flow[mfv_Idx] = mfv;
      mfv_volume[mfv_Idx] = inVolume[today];
      sumMFV += mfv;
      sumVol += inVolume[today];
      liveVol = (fabs(inVolume[today]) <= 0.0) ? 0 : 1;
      nullRun = (nullRun + 1) & (liveVol - 1);
      today += 1;
      if( TA_UNLIKELY(nullRun >= optInTimePeriod) )
      {
         nullRun = optInTimePeriod;
         sumMFV = 0.0;
         sumVol = 0.0;
         outReal[outIdx++] = 0.0;
      } else if( sumVol > 0.0 )
      {
         outReal[outIdx++] = sumMFV / sumVol;
      } else 
      {
         outReal[outIdx++] = 0.0;
      }
      mfv_Idx++;
      if( mfv_Idx > maxIdx_mfv ) mfv_Idx = 0;
   }
   if( mfv_flow != &local_mfv_flow[0] ) { TA_Free( mfv_flow ); mfv_flow = &local_mfv_flow[0]; }
   if( mfv_volume != &local_mfv_volume[0] ) { TA_Free( mfv_volume ); mfv_volume = &local_mfv_volume[0]; }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_CMF_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   return optInTimePeriod - 1;
}
