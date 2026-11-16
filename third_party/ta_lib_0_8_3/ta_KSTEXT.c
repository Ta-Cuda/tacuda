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
TA_LIB_API TA_RetCode TA_KSTEXT( int    startIdx,
                                 int    endIdx,
                                 const double inReal[],
                                 int optInROC1Period,
                                 int optInROC2Period,
                                 int optInROC3Period,
                                 int optInROC4Period,
                                 int optInMA1Period,
                                 int optInMA2Period,
                                 int optInMA3Period,
                                 int optInMA4Period,
                                 int optInSignalPeriod,
                                 TA_MAType optInROCMAType,
                                 TA_MAType optInSignalMAType,
                                 int          *outBegIdx,
                                 int          *outNBElement,
                                 double        outKST[],
                                 double        outKSTSignal[] )
{
   double *kstBuffer;
   double *tempBuffer;
   TA_RetCode retCode;
   int lookbackTotal;
   int lookbackSignal;
   int lookbackMA;
   int sigStart;
   int lookbackMA1;
   int lookbackMA2;
   int lookbackMA3;
   int lookbackMA4;
   int tempInteger;
   int tempBegIdx;
   int rocNb;
   int kstNb;
   int legNb;
   int sigNb;
   int i;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInROC1Period == TA_INTEGER_DEFAULT )
      optInROC1Period = 10;
   else if( (int)optInROC1Period < 1 || (int)optInROC1Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInROC2Period == TA_INTEGER_DEFAULT )
      optInROC2Period = 15;
   else if( (int)optInROC2Period < 1 || (int)optInROC2Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInROC3Period == TA_INTEGER_DEFAULT )
      optInROC3Period = 20;
   else if( (int)optInROC3Period < 1 || (int)optInROC3Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInROC4Period == TA_INTEGER_DEFAULT )
      optInROC4Period = 30;
   else if( (int)optInROC4Period < 1 || (int)optInROC4Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInMA1Period == TA_INTEGER_DEFAULT )
      optInMA1Period = 10;
   else if( (int)optInMA1Period < 1 || (int)optInMA1Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInMA2Period == TA_INTEGER_DEFAULT )
      optInMA2Period = 10;
   else if( (int)optInMA2Period < 1 || (int)optInMA2Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInMA3Period == TA_INTEGER_DEFAULT )
      optInMA3Period = 10;
   else if( (int)optInMA3Period < 1 || (int)optInMA3Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInMA4Period == TA_INTEGER_DEFAULT )
      optInMA4Period = 15;
   else if( (int)optInMA4Period < 1 || (int)optInMA4Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSignalPeriod == TA_INTEGER_DEFAULT )
      optInSignalPeriod = 9;
   else if( (int)optInSignalPeriod < 1 || (int)optInSignalPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInROCMAType == TA_INTEGER_DEFAULT || optInROCMAType == TA_MAType_DEFAULT )
      optInROCMAType = 0;
   else if( (int)optInROCMAType < TA_MATYPE_MIN || (int)optInROCMAType > TA_MATYPE_MAX )
      return TA_BAD_PARAM;
   if( (int)optInSignalMAType == TA_INTEGER_DEFAULT || optInSignalMAType == TA_MAType_DEFAULT )
      optInSignalMAType = 0;
   else if( (int)optInSignalMAType < TA_MATYPE_MIN || (int)optInSignalMAType > TA_MATYPE_MAX )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outKST )
      return TA_BAD_PARAM;
   if( !outKSTSignal )
      return TA_BAD_PARAM;
   if( outKST == outKSTSignal )
      return TA_BAD_PARAM;

   /* With every type SMA this is bit-exact with kst(): each leg's average
    * starts on the first bar the signal consumes, and the line accumulates
    * its legs left to right. Changing either breaks the equality.
    */
   *outBegIdx= 0;
   *outNBElement= 0;
   /* lookbackTotal must stay kstext_lookback() of the same parameters. */
   lookbackSignal = TA_MA_Lookback(optInSignalPeriod,optInSignalMAType);
   lookbackMA1 = TA_MA_Lookback(optInMA1Period,optInROCMAType);
   lookbackMA2 = TA_MA_Lookback(optInMA2Period,optInROCMAType);
   lookbackMA3 = TA_MA_Lookback(optInMA3Period,optInROCMAType);
   lookbackMA4 = TA_MA_Lookback(optInMA4Period,optInROCMAType);
   lookbackTotal = optInROC1Period + lookbackMA1;
   tempInteger = optInROC2Period + lookbackMA2;
   if( tempInteger > lookbackTotal )
   {
      lookbackTotal = tempInteger;
   }
   tempInteger = optInROC3Period + lookbackMA3;
   if( tempInteger > lookbackTotal )
   {
      lookbackTotal = tempInteger;
   }
   tempInteger = optInROC4Period + lookbackMA4;
   if( tempInteger > lookbackTotal )
   {
      lookbackTotal = tempInteger;
   }
   lookbackTotal += lookbackSignal;
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   sigStart = startIdx - lookbackSignal;
   /* Both buffers are sized by the longest leg average: a rate of change
    * starts its average's lookback before sigStart.
    */
   lookbackMA = lookbackMA1;
   if( lookbackMA2 > lookbackMA )
   {
      lookbackMA = lookbackMA2;
   }
   if( lookbackMA3 > lookbackMA )
   {
      lookbackMA = lookbackMA3;
   }
   if( lookbackMA4 > lookbackMA )
   {
      lookbackMA = lookbackMA4;
   }
   tempInteger = endIdx - sigStart + 1 + lookbackMA;
   kstBuffer = malloc(tempInteger * sizeof(double));
   if( !kstBuffer )
   {
      return TA_ALLOC_ERR;
   }
   tempBuffer = malloc(tempInteger * sizeof(double));
   if( !tempBuffer )
   {
      free(kstBuffer);
      return TA_ALLOC_ERR;
   }
   retCode = TA_ROC(sigStart - lookbackMA1,endIdx,inReal,optInROC1Period,&tempBegIdx,&rocNb,kstBuffer);
   if( retCode != TA_SUCCESS )
   {
      free(kstBuffer);
      free(tempBuffer);
      return retCode;
   }
   retCode = TA_MA(0,rocNb - 1,kstBuffer,optInMA1Period,optInROCMAType,&tempBegIdx,&kstNb,kstBuffer);
   if( retCode != TA_SUCCESS )
   {
      free(kstBuffer);
      free(tempBuffer);
      return retCode;
   }
   retCode = TA_ROC(sigStart - lookbackMA2,endIdx,inReal,optInROC2Period,&tempBegIdx,&rocNb,tempBuffer);
   if( retCode != TA_SUCCESS )
   {
      free(kstBuffer);
      free(tempBuffer);
      return retCode;
   }
   retCode = TA_MA(0,rocNb - 1,tempBuffer,optInMA2Period,optInROCMAType,&tempBegIdx,&legNb,tempBuffer);
   if( retCode != TA_SUCCESS )
   {
      free(kstBuffer);
      free(tempBuffer);
      return retCode;
   }
   for( i = 0; i < kstNb; i += 1 )
   {
      kstBuffer[i] = fma(2.0, tempBuffer[i], kstBuffer[i]);
   }
   retCode = TA_ROC(sigStart - lookbackMA3,endIdx,inReal,optInROC3Period,&tempBegIdx,&rocNb,tempBuffer);
   if( retCode != TA_SUCCESS )
   {
      free(kstBuffer);
      free(tempBuffer);
      return retCode;
   }
   retCode = TA_MA(0,rocNb - 1,tempBuffer,optInMA3Period,optInROCMAType,&tempBegIdx,&legNb,tempBuffer);
   if( retCode != TA_SUCCESS )
   {
      free(kstBuffer);
      free(tempBuffer);
      return retCode;
   }
   for( i = 0; i < kstNb; i += 1 )
   {
      kstBuffer[i] = fma(3.0, tempBuffer[i], kstBuffer[i]);
   }
   retCode = TA_ROC(sigStart - lookbackMA4,endIdx,inReal,optInROC4Period,&tempBegIdx,&rocNb,tempBuffer);
   if( retCode != TA_SUCCESS )
   {
      free(kstBuffer);
      free(tempBuffer);
      return retCode;
   }
   retCode = TA_MA(0,rocNb - 1,tempBuffer,optInMA4Period,optInROCMAType,&tempBegIdx,&legNb,tempBuffer);
   if( retCode != TA_SUCCESS )
   {
      free(kstBuffer);
      free(tempBuffer);
      return retCode;
   }
   for( i = 0; i < kstNb; i += 1 )
   {
      kstBuffer[i] = fma(4.0, tempBuffer[i], kstBuffer[i]);
   }
   /* Every read of inReal is done: an output may alias it. */
   memmove(outKST,&kstBuffer[lookbackSignal],(endIdx - startIdx + 1) * sizeof(double));
   retCode = TA_MA(0,kstNb - 1,kstBuffer,optInSignalPeriod,optInSignalMAType,&tempBegIdx,&sigNb,outKSTSignal);
   free(kstBuffer);
   free(tempBuffer);
   if( retCode != TA_SUCCESS )
   {
      return retCode;
   }
   *outBegIdx= startIdx;
   *outNBElement= sigNb;
   return TA_SUCCESS;
}

TA_LIB_API int TA_KSTEXT_Lookback( int optInROC1Period, int optInROC2Period, int optInROC3Period, int optInROC4Period, int optInMA1Period, int optInMA2Period, int optInMA3Period, int optInMA4Period, int optInSignalPeriod, TA_MAType optInROCMAType, TA_MAType optInSignalMAType )
{
   int legMax;
   int leg;
   if( (int)optInROC1Period == TA_INTEGER_DEFAULT )
      optInROC1Period = 10;
   else if( (int)optInROC1Period < 1 || (int)optInROC1Period > 100000 )
      return -1;
   if( (int)optInROC2Period == TA_INTEGER_DEFAULT )
      optInROC2Period = 15;
   else if( (int)optInROC2Period < 1 || (int)optInROC2Period > 100000 )
      return -1;
   if( (int)optInROC3Period == TA_INTEGER_DEFAULT )
      optInROC3Period = 20;
   else if( (int)optInROC3Period < 1 || (int)optInROC3Period > 100000 )
      return -1;
   if( (int)optInROC4Period == TA_INTEGER_DEFAULT )
      optInROC4Period = 30;
   else if( (int)optInROC4Period < 1 || (int)optInROC4Period > 100000 )
      return -1;
   if( (int)optInMA1Period == TA_INTEGER_DEFAULT )
      optInMA1Period = 10;
   else if( (int)optInMA1Period < 1 || (int)optInMA1Period > 100000 )
      return -1;
   if( (int)optInMA2Period == TA_INTEGER_DEFAULT )
      optInMA2Period = 10;
   else if( (int)optInMA2Period < 1 || (int)optInMA2Period > 100000 )
      return -1;
   if( (int)optInMA3Period == TA_INTEGER_DEFAULT )
      optInMA3Period = 10;
   else if( (int)optInMA3Period < 1 || (int)optInMA3Period > 100000 )
      return -1;
   if( (int)optInMA4Period == TA_INTEGER_DEFAULT )
      optInMA4Period = 15;
   else if( (int)optInMA4Period < 1 || (int)optInMA4Period > 100000 )
      return -1;
   if( (int)optInSignalPeriod == TA_INTEGER_DEFAULT )
      optInSignalPeriod = 9;
   else if( (int)optInSignalPeriod < 1 || (int)optInSignalPeriod > 100000 )
      return -1;
   if( (int)optInROCMAType == TA_INTEGER_DEFAULT || optInROCMAType == TA_MAType_DEFAULT )
      optInROCMAType = 0;
   else if( (int)optInROCMAType < TA_MATYPE_MIN || (int)optInROCMAType > TA_MATYPE_MAX )
      return -1;
   if( (int)optInSignalMAType == TA_INTEGER_DEFAULT || optInSignalMAType == TA_MAType_DEFAULT )
      optInSignalMAType = 0;
   else if( (int)optInSignalMAType < TA_MATYPE_MIN || (int)optInSignalMAType > TA_MATYPE_MAX )
      return -1;
   legMax = optInROC1Period + TA_MA_Lookback(optInMA1Period,optInROCMAType);
   leg = optInROC2Period + TA_MA_Lookback(optInMA2Period,optInROCMAType);
   if( leg > legMax )
   {
      legMax = leg;
   }
   leg = optInROC3Period + TA_MA_Lookback(optInMA3Period,optInROCMAType);
   if( leg > legMax )
   {
      legMax = leg;
   }
   leg = optInROC4Period + TA_MA_Lookback(optInMA4Period,optInROCMAType);
   if( leg > legMax )
   {
      legMax = leg;
   }
   return legMax + TA_MA_Lookback(optInSignalPeriod,optInSignalMAType);
}
