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

TA_LIB_API TA_RetCode TA_STOCHRSI( int    startIdx,
                                   int    endIdx,
                                   const double inReal[],
                                   int optInTimePeriod,
                                   int optInFastK_Period,
                                   int optInFastD_Period,
                                   TA_MAType optInFastD_MAType,
                                   int          *outBegIdx,
                                   int          *outNBElement,
                                   double        outFastK[],
                                   double        outFastD[] )
{
   double *tempRSIBuffer;
   TA_RetCode retCode;
   int lookbackTotal;
   int lookbackSTOCHF;
   int tempArraySize;
   int outBegIdx1;
   int outBegIdx2;
   int outNbElement1;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInFastK_Period == TA_INTEGER_DEFAULT )
      optInFastK_Period = 5;
   else if( (int)optInFastK_Period < 1 || (int)optInFastK_Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInFastD_Period == TA_INTEGER_DEFAULT )
      optInFastD_Period = 3;
   else if( (int)optInFastD_Period < 1 || (int)optInFastD_Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInFastD_MAType == TA_INTEGER_DEFAULT || optInFastD_MAType == TA_MAType_DEFAULT )
      optInFastD_MAType = 0;
   else if( (int)optInFastD_MAType < TA_MATYPE_MIN || (int)optInFastD_MAType > TA_MATYPE_MAX )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outFastK )
      return TA_BAD_PARAM;
   if( !outFastD )
      return TA_BAD_PARAM;
   if( outFastK == outFastD )
      return TA_BAD_PARAM;

   /* Stochastic RSI
    *
    * Reference: "Stochastic RSI and Dynamic Momentum Index"
    *            by Tushar Chande and Stanley Kroll
    *            Stock&Commodities V.11:5 (189-199)
    *
    * The TA-Lib version offer flexibility beyond what is explain
    * in the Stock&Commodities article.
    *
    * To calculate the "Unsmoothed stochastic RSI" with symetry like
    * explain in the article, keep the optInTimePeriod and optInFastK_Period
    * equal. Example:
    *
    *    unsmoothed stoch RSI 14 : optInTimePeriod   = 14
    *                              optInFastK_Period = 14
    *                              optInFastD_Period = 'x'
    *
    * The outFastK is the unsmoothed RSI discuss in the article.
    *
    * You can set the optInFastD_Period to smooth the RSI. The smooth
    * version will be found in outFastD. The outFastK will still contain
    * the unsmoothed stoch RSI. If you do not care about the smoothing of
    * the StochRSI, just leave optInFastD_Period to 1 and ignore outFastD.
    */
   *outBegIdx= 0;
   *outNBElement= 0;
   /* Adjust startIdx to account for the lookback period. */
   lookbackSTOCHF = TA_STOCHF_Lookback(optInFastK_Period,optInFastD_Period,optInFastD_MAType);
   lookbackTotal = TA_RSI_Lookback(optInTimePeriod) + lookbackSTOCHF;
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   *outBegIdx= startIdx;
   tempArraySize = endIdx - startIdx + 1 + lookbackSTOCHF;
   tempRSIBuffer = malloc(tempArraySize * sizeof(double));
   if( !tempRSIBuffer )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_ALLOC_ERR;
   }
   retCode = TA_RSI(startIdx - lookbackSTOCHF,endIdx,inReal,optInTimePeriod,&outBegIdx1,&outNbElement1,tempRSIBuffer);
   if( retCode != TA_SUCCESS || outNbElement1 == 0 )
   {
      free(tempRSIBuffer);
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   retCode = TA_STOCHF(0,tempArraySize - 1,tempRSIBuffer,tempRSIBuffer,tempRSIBuffer,optInFastK_Period,optInFastD_Period,optInFastD_MAType,&outBegIdx2,outNBElement,outFastK,outFastD);
   free(tempRSIBuffer);
   if( retCode != TA_SUCCESS || (int)*outNBElement == 0 )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   return TA_SUCCESS;
}

TA_LIB_API int TA_STOCHRSI_Lookback( int optInTimePeriod, int optInFastK_Period, int optInFastD_Period, TA_MAType optInFastD_MAType )
{
   int retValue;
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   if( (int)optInFastK_Period == TA_INTEGER_DEFAULT )
      optInFastK_Period = 5;
   else if( (int)optInFastK_Period < 1 || (int)optInFastK_Period > 100000 )
      return -1;
   if( (int)optInFastD_Period == TA_INTEGER_DEFAULT )
      optInFastD_Period = 3;
   else if( (int)optInFastD_Period < 1 || (int)optInFastD_Period > 100000 )
      return -1;
   if( (int)optInFastD_MAType == TA_INTEGER_DEFAULT || optInFastD_MAType == TA_MAType_DEFAULT )
      optInFastD_MAType = 0;
   else if( (int)optInFastD_MAType < TA_MATYPE_MIN || (int)optInFastD_MAType > TA_MATYPE_MAX )
      return -1;
   retValue = TA_RSI_Lookback(optInTimePeriod) + TA_STOCHF_Lookback(optInFastK_Period,optInFastD_Period,optInFastD_MAType);
   return retValue;
}
