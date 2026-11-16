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

TA_LIB_API TA_RetCode TA_MA( int    startIdx,
                             int    endIdx,
                             const double inReal[],
                             int optInTimePeriod,
                             TA_MAType optInMAType,
                             int          *outBegIdx,
                             int          *outNBElement,
                             double        outReal[] )
{
   double *famaScratch;
   TA_RetCode retCode;
   int nbElement;
   int outIdx;
   int todayIdx;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 30;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInMAType == TA_INTEGER_DEFAULT || optInMAType == TA_MAType_DEFAULT )
      optInMAType = 0;
   else if( (int)optInMAType < TA_MATYPE_MIN || (int)optInMAType > TA_MATYPE_MAX )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* Nothing to produce: the range ends before the lookback. Answer here
    * rather than forwarding.
    *
    * The VALUE is the same either way: ma_lookback returns exactly the lookback
    * the arm's callee computes for itself, from the same arguments the arm
    * passes it, and every callee clamps startIdx to that lookback and yields
    * 0,0 without reading. What changes is whose frame answers, and that is
    * visible to one caller only - the zero-length no-I/O probe. It hands the
    * body empty arrays on this range to check that nothing is read; forwarding
    * makes the callee's PUBLIC input bound (endIdx + 1 elements, no
    * sub-lookback escape) answer TA_BAD_PARAM before any array is reached, so
    * the probe cannot tell "read nothing" from "never ran". ma was the last
    * core withheld from that sweep for exactly this reason; apo, bbands, ppo,
    * pvo and stddev already carried this guard. No legitimate caller is
    * affected: ma's own public tier already requires endIdx + 1 input elements,
    * and on this range the callee's OUTPUT bound is 0, so a caller sizing by
    * the published formula was never rejected by it.
    *
    * It cannot mask a TA_BAD_PARAM. An optInMAType outside the enum is refused
    * by the generated entry point above this guard. An in-range member with no
    * arm here would reach ma_lookback's own default and get 0, and 0 > endIdx
    * is false for every endIdx the entry point admits, so control still reaches
    * the switch and still answers TA_BAD_PARAM. The identity path below is out
    * of the guard's reach for the same reason - its lookback is 0.
    */
   if( TA_MA_Lookback(optInTimePeriod,optInMAType) > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   /* No-smoothing identity: period 1 (every MA type) or the explicit
    * TA_MAType_DISABLED (any period, issue #93). One copy path, lookback 0.
    */
   if( optInTimePeriod == 1 || optInMAType == TA_MAType_DISABLED )
   {
      nbElement = endIdx - startIdx + 1;
      *outNBElement= nbElement;
      for( todayIdx = startIdx, outIdx = 0; outIdx < nbElement; outIdx += 1, todayIdx += 1 )
      {
         outReal[outIdx] = inReal[todayIdx];
      }
      *outBegIdx= startIdx;
      return TA_SUCCESS;
   }
   /* Simply forward the job to the corresponding TA function. */
   switch( optInMAType )
   {
   case TA_MAType_SMA:
      retCode = TA_SMA(startIdx,endIdx,inReal,optInTimePeriod,outBegIdx,outNBElement,outReal);
      break;
   case TA_MAType_EMA:
      retCode = TA_EMA(startIdx,endIdx,inReal,optInTimePeriod,outBegIdx,outNBElement,outReal);
      break;
   case TA_MAType_WMA:
      retCode = TA_WMA(startIdx,endIdx,inReal,optInTimePeriod,outBegIdx,outNBElement,outReal);
      break;
   case TA_MAType_DEMA:
      retCode = TA_DEMA(startIdx,endIdx,inReal,optInTimePeriod,outBegIdx,outNBElement,outReal);
      break;
   case TA_MAType_TEMA:
      retCode = TA_TEMA(startIdx,endIdx,inReal,optInTimePeriod,outBegIdx,outNBElement,outReal);
      break;
   case TA_MAType_TRIMA:
      retCode = TA_TRIMA(startIdx,endIdx,inReal,optInTimePeriod,outBegIdx,outNBElement,outReal);
      break;
   case TA_MAType_KAMA:
      retCode = TA_KAMA(startIdx,endIdx,inReal,optInTimePeriod,outBegIdx,outNBElement,outReal);
      break;
   case TA_MAType_MAMA:
      /* The optInTimePeriod is ignored. */
      famaScratch = malloc((endIdx - startIdx + 1) * sizeof(double));
      if( !famaScratch )
      {
         *outBegIdx= 0;
         *outNBElement= 0;
         return TA_ALLOC_ERR;
      }
      retCode = TA_MAMA(startIdx,endIdx,inReal,0.5,0.05,outBegIdx,outNBElement,outReal,famaScratch);
      free(famaScratch);
      break;
   case TA_MAType_T3:
      retCode = TA_T3(startIdx,endIdx,inReal,optInTimePeriod,0.7,outBegIdx,outNBElement,outReal);
      break;
   case TA_MAType_HMA:
      retCode = TA_HMA(startIdx,endIdx,inReal,optInTimePeriod,outBegIdx,outNBElement,outReal);
      break;
   case TA_MAType_ZLEMA:
      retCode = TA_ZLEMA(startIdx,endIdx,inReal,optInTimePeriod,outBegIdx,outNBElement,outReal);
      break;
   case TA_MAType_RMA:
      retCode = TA_RMA(startIdx,endIdx,inReal,optInTimePeriod,outBegIdx,outNBElement,outReal);
      break;
   case TA_MAType_VIDYA:
      /* The one period is the EMA length; the CMO period is round(3n/4),
       * Chande's 12:9 ratio.
       */
      retCode = TA_VIDYA(startIdx,endIdx,inReal,optInTimePeriod,(3 * optInTimePeriod + 2) / 4,outBegIdx,outNBElement,outReal);
      break;
   case TA_MAType_ALMA:
      retCode = TA_ALMA(startIdx,endIdx,inReal,optInTimePeriod,6.0,0.85,outBegIdx,outNBElement,outReal);
      break;
   default:
      retCode = TA_BAD_PARAM;
      break;
   }
   return retCode;
}

TA_LIB_API int TA_MA_Lookback( int optInTimePeriod, TA_MAType optInMAType )
{
   int retValue;
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 30;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   if( (int)optInMAType == TA_INTEGER_DEFAULT || optInMAType == TA_MAType_DEFAULT )
      optInMAType = 0;
   else if( (int)optInMAType < TA_MATYPE_MIN || (int)optInMAType > TA_MATYPE_MAX )
      return -1;
   if( optInTimePeriod <= 1 || optInMAType == TA_MAType_DISABLED )
   {
      return 0;
   }
   switch( optInMAType )
   {
   case TA_MAType_SMA:
      retValue = TA_SMA_Lookback(optInTimePeriod);
      break;
   case TA_MAType_EMA:
      retValue = TA_EMA_Lookback(optInTimePeriod);
      break;
   case TA_MAType_WMA:
      retValue = TA_WMA_Lookback(optInTimePeriod);
      break;
   case TA_MAType_DEMA:
      retValue = TA_DEMA_Lookback(optInTimePeriod);
      break;
   case TA_MAType_TEMA:
      retValue = TA_TEMA_Lookback(optInTimePeriod);
      break;
   case TA_MAType_TRIMA:
      retValue = TA_TRIMA_Lookback(optInTimePeriod);
      break;
   case TA_MAType_KAMA:
      retValue = TA_KAMA_Lookback(optInTimePeriod);
      break;
   case TA_MAType_MAMA:
      retValue = TA_MAMA_Lookback(0.5,0.05);
      break;
   case TA_MAType_T3:
      retValue = TA_T3_Lookback(optInTimePeriod,0.7);
      break;
   case TA_MAType_HMA:
      retValue = TA_HMA_Lookback(optInTimePeriod);
      break;
   case TA_MAType_ZLEMA:
      retValue = TA_ZLEMA_Lookback(optInTimePeriod);
      break;
   case TA_MAType_RMA:
      retValue = TA_RMA_Lookback(optInTimePeriod);
      break;
   case TA_MAType_VIDYA:
      retValue = TA_VIDYA_Lookback(optInTimePeriod,(3 * optInTimePeriod + 2) / 4);
      break;
   case TA_MAType_ALMA:
      retValue = TA_ALMA_Lookback(optInTimePeriod,6.0,0.85);
      break;
   default:
      retValue = 0;
      break;
   }
   return retValue;
}
