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
TA_LIB_API TA_RetCode TA_VIDYA( int    startIdx,
                                int    endIdx,
                                const double inReal[],
                                int optInTimePeriod,
                                int optInCMOPeriod,
                                int          *outBegIdx,
                                int          *outNBElement,
                                double        outReal[] )
{
   int outIdx;
   int today;
   int trailingIdx;
   int lookbackTotal;
   int i;
   int nullRun;
   int live;
   double upSum;
   double downSum;
   double sum;
   double diff;
   double up;
   double tempReal;
   double prevValue;
   double trailingValue;
   double alpha;
   double k;
   double prevVIDYA;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 12;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInCMOPeriod == TA_INTEGER_DEFAULT )
      optInCMOPeriod = 9;
   else if( (int)optInCMOPeriod < 2 || (int)optInCMOPeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   *outBegIdx= 0;
   *outNBElement= 0;
   /* No smoothing at period 1: the output is a copy of the input, as MA gives
    * for every MAType. The unstable period still delays the first output.
    */
   if( optInTimePeriod == 1 )
   {
      lookbackTotal = TA_VIDYA_Lookback(optInTimePeriod,optInCMOPeriod);
      if( startIdx < lookbackTotal )
      {
         startIdx = lookbackTotal;
      }
      if( startIdx > endIdx )
      {
         return TA_SUCCESS;
      }
      *outBegIdx= startIdx;
      outIdx = 0;
      today = startIdx;
      while( today <= endIdx )
      {
         outReal[outIdx++] = inReal[today++];
      }
      *outNBElement= outIdx;
      return TA_SUCCESS;
   }
   lookbackTotal = TA_VIDYA_Lookback(optInTimePeriod,optInCMOPeriod);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   alpha = 2.0 / (double)(optInTimePeriod + 1);
   /* The CMO below is TA_CMOU's loop, spelling and nullRun reset included, so
    * that from the first full window on VIDYA equals the composite of TA_CMOU
    * bit for bit. The recursion is seeded on the window's first price and
    * steps through the warm-up with the CMO of the changes seen so far. The
    * unstable period enters it that many bars before startIdx.
    */
   today = startIdx - lookbackTotal;
   trailingIdx = today + 1;
   prevValue = inReal[today];
   trailingValue = prevValue;
   prevVIDYA = prevValue;
   upSum = 0.0;
   downSum = 0.0;
   nullRun = 0;
   for( i = 0; i < optInCMOPeriod; i += 1 )
   {
      today += 1;
      tempReal = inReal[today];
      diff = tempReal - prevValue;
      prevValue = tempReal;
      up = (diff + fabs(diff)) * 0.5;
      upSum += up;
      downSum += up - diff;
      live = (fabs(diff) <= 0.0) ? 0 : 1;
      nullRun = (nullRun + 1) & (live - 1);
      sum = upSum + downSum;
      if( sum > 0.0 )
      {
         k = alpha * (fabs(100.0 * (upSum - downSum) / sum) / 100.0);
      } else 
      {
         k = 0.0;
      }
      prevVIDYA = fma(prevValue - prevVIDYA, k, prevVIDYA);
   }
   /* Each bar below drops the change that leaves the window after its own
    * update, ready for the next bar, as in TA_CMOU.
    */
   tempReal = inReal[trailingIdx];
   diff = tempReal - trailingValue;
   trailingValue = tempReal;
   trailingIdx += 1;
   up = (diff + fabs(diff)) * 0.5;
   upSum -= up;
   downSum -= up - diff;
   today += 1;
   /* Skip the unstable period: the whole computation, nothing written. */
   while( today <= startIdx )
   {
      tempReal = inReal[today];
      diff = tempReal - prevValue;
      prevValue = tempReal;
      up = (diff + fabs(diff)) * 0.5;
      upSum += up;
      downSum += up - diff;
      live = (fabs(diff) <= 0.0) ? 0 : 1;
      nullRun = (nullRun + 1) & (live - 1);
      if( TA_UNLIKELY(nullRun >= optInCMOPeriod) )
      {
         nullRun = optInCMOPeriod;
         upSum = 0.0;
         downSum = 0.0;
         k = 0.0;
      } else 
      {
         sum = upSum + downSum;
         if( sum > 0.0 )
         {
            k = alpha * (fabs(100.0 * (upSum - downSum) / sum) / 100.0);
         } else 
         {
            k = 0.0;
         }
      }
      prevVIDYA = fma(prevValue - prevVIDYA, k, prevVIDYA);
      tempReal = inReal[trailingIdx];
      diff = tempReal - trailingValue;
      trailingValue = tempReal;
      trailingIdx += 1;
      up = (diff + fabs(diff)) * 0.5;
      upSum -= up;
      downSum -= up - diff;
      today += 1;
   }
   outReal[0] = prevVIDYA;
   outIdx = 1;
   while( today <= endIdx )
   {
      tempReal = inReal[today];
      diff = tempReal - prevValue;
      prevValue = tempReal;
      up = (diff + fabs(diff)) * 0.5;
      upSum += up;
      downSum += up - diff;
      live = (fabs(diff) <= 0.0) ? 0 : 1;
      nullRun = (nullRun + 1) & (live - 1);
      if( TA_UNLIKELY(nullRun >= optInCMOPeriod) )
      {
         nullRun = optInCMOPeriod;
         upSum = 0.0;
         downSum = 0.0;
         k = 0.0;
      } else 
      {
         sum = upSum + downSum;
         if( sum > 0.0 )
         {
            k = alpha * (fabs(100.0 * (upSum - downSum) / sum) / 100.0);
         } else 
         {
            k = 0.0;
         }
      }
      prevVIDYA = fma(prevValue - prevVIDYA, k, prevVIDYA);
      outReal[outIdx++] = prevVIDYA;
      tempReal = inReal[trailingIdx];
      diff = tempReal - trailingValue;
      trailingValue = tempReal;
      trailingIdx += 1;
      up = (diff + fabs(diff)) * 0.5;
      upSum -= up;
      downSum -= up - diff;
      today += 1;
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_NOINLINE TA_LIB_API int TA_VIDYA_Lookback( int optInTimePeriod, int optInCMOPeriod )
{
   int root;
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 12;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   if( (int)optInCMOPeriod == TA_INTEGER_DEFAULT )
      optInCMOPeriod = 9;
   else if( (int)optInCMOPeriod < 2 || (int)optInCMOPeriod > 100000 )
      return -1;
   root = (int)sqrt((double)optInCMOPeriod);
   if( optInTimePeriod == 1 )
   {
      return TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_VIDYA,Vidya,0,0);
   }
   return optInCMOPeriod + TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_VIDYA,Vidya,((2 * 4 * (optInTimePeriod + 1) * root > 100000000) ? 100000000 : 2 * 4 * (optInTimePeriod + 1) * root),((2 * 8 * (optInTimePeriod + 1) * root > 100000000) ? 100000000 : 2 * 8 * (optInTimePeriod + 1) * root));
}
