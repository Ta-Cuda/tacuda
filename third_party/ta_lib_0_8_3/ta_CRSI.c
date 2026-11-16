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

TA_LIB_API TA_RetCode TA_CRSI( int    startIdx,
                               int    endIdx,
                               const double inReal[],
                               int optInTimePeriod,
                               int optInStreakPeriod,
                               int optInRankPeriod,
                               int          *outBegIdx,
                               int          *outNBElement,
                               double        outReal[] )
{
   double *tempRSI;
   double *tempStreak;
   double *tempStreakRSI;
   TA_RetCode retCode;
   int lookbackTotal;
   int anchorIdx;
   int today;
   int outIdx;
   int i;
   int offsetRSI;
   int offsetStreak;
   int tempBegIdx;
   int rsiNb;
   int streakNb;
   int rocNb;
   double prevClose;
   double close;
   double streak;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 3;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInStreakPeriod == TA_INTEGER_DEFAULT )
      optInStreakPeriod = 2;
   else if( (int)optInStreakPeriod < 2 || (int)optInStreakPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInRankPeriod == TA_INTEGER_DEFAULT )
      optInRankPeriod = 100;
   else if( (int)optInRankPeriod < 2 || (int)optInRankPeriod > 10000 )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   *outBegIdx= 0;
   *outNBElement= 0;
   lookbackTotal = TA_CRSI_Lookback(optInTimePeriod,optInStreakPeriod,optInRankPeriod);
   if( lookbackTotal > endIdx )
   {
      return TA_SUCCESS;
   }
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   /* Both RSI legs start on the bar lookbackTotal before startIdx (the streak,
    * a difference, one bar later), so the leg with the shorter lookback warms
    * up over the spare bars instead of starting cold at its own lookback.
    */
   anchorIdx = startIdx - lookbackTotal;
   tempStreak = malloc((endIdx - anchorIdx) * sizeof(double));
   if( !tempStreak )
   {
      return TA_ALLOC_ERR;
   }
   tempRSI = malloc((endIdx - anchorIdx - TA_RSI_Lookback(optInTimePeriod) + 1) * sizeof(double));
   if( !tempRSI )
   {
      free(tempStreak);
      return TA_ALLOC_ERR;
   }
   tempStreakRSI = malloc((endIdx - anchorIdx - TA_RSI_Lookback(optInStreakPeriod)) * sizeof(double));
   if( !tempStreakRSI )
   {
      free(tempStreak);
      free(tempRSI);
      return TA_ALLOC_ERR;
   }
   streak = 0.0;
   prevClose = inReal[anchorIdx];
   outIdx = 0;
   today = anchorIdx + 1;
   while( today <= endIdx )
   {
      close = inReal[today];
      if( close > prevClose )
      {
         streak = (streak > 0.0) ? streak + 1.0 : 1.0;
      } else if( close < prevClose )
      {
         streak = (streak < 0.0) ? streak - 1.0 : -1.0;
      } else 
      {
         streak = 0.0;
      }
      tempStreak[outIdx++] = streak;
      prevClose = close;
      today += 1;
   }
   retCode = TA_RSI(0,outIdx - 1,tempStreak,optInStreakPeriod,&tempBegIdx,&streakNb,tempStreakRSI);
   if( retCode != TA_SUCCESS )
   {
      free(tempStreak);
      free(tempRSI);
      free(tempStreakRSI);
      return retCode;
   }
   retCode = TA_RSI(anchorIdx + TA_RSI_Lookback(optInTimePeriod),endIdx,inReal,optInTimePeriod,&tempBegIdx,&rsiNb,tempRSI);
   if( retCode != TA_SUCCESS )
   {
      free(tempStreak);
      free(tempRSI);
      free(tempStreakRSI);
      return retCode;
   }
   /* The streak is consumed: its buffer holds the returns. */
   retCode = TA_ROCP(startIdx - optInRankPeriod,endIdx,inReal,1,&tempBegIdx,&rocNb,tempStreak);
   if( retCode != TA_SUCCESS )
   {
      free(tempStreak);
      free(tempRSI);
      free(tempStreakRSI);
      return retCode;
   }
   retCode = TA_PERCENTRANK(0,rocNb - 1,tempStreak,optInRankPeriod,outBegIdx,outNBElement,outReal);
   if( retCode != TA_SUCCESS )
   {
      free(tempStreak);
      free(tempRSI);
      free(tempStreakRSI);
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   offsetRSI = rsiNb - *outNBElement;
   offsetStreak = streakNb - *outNBElement;
   for( i = 0; i < (int)*outNBElement; i += 1 )
   {
      outReal[i] = (tempRSI[i + offsetRSI] + tempStreakRSI[i + offsetStreak] + outReal[i]) / 3.0;
   }
   free(tempStreak);
   free(tempRSI);
   free(tempStreakRSI);
   *outBegIdx= startIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_CRSI_Lookback( int optInTimePeriod, int optInStreakPeriod, int optInRankPeriod )
{
   int retValue;
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 3;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   if( (int)optInStreakPeriod == TA_INTEGER_DEFAULT )
      optInStreakPeriod = 2;
   else if( (int)optInStreakPeriod < 2 || (int)optInStreakPeriod > 100000 )
      return -1;
   if( (int)optInRankPeriod == TA_INTEGER_DEFAULT )
      optInRankPeriod = 100;
   else if( (int)optInRankPeriod < 2 || (int)optInRankPeriod > 10000 )
      return -1;
   retValue = TA_RSI_Lookback(optInTimePeriod);
   retValue = max(retValue,TA_RSI_Lookback(optInStreakPeriod) + 1);
   retValue = max(retValue,TA_PERCENTRANK_Lookback(optInRankPeriod) + TA_ROCP_Lookback(1));
   return retValue;
}
