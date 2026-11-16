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

TA_LIB_API TA_RetCode TA_MACDEXT( int    startIdx,
                                  int    endIdx,
                                  const double inReal[],
                                  int optInFastPeriod,
                                  TA_MAType optInFastMAType,
                                  int optInSlowPeriod,
                                  TA_MAType optInSlowMAType,
                                  int optInSignalPeriod,
                                  TA_MAType optInSignalMAType,
                                  int          *outBegIdx,
                                  int          *outNBElement,
                                  double        outMACD[],
                                  double        outMACDSignal[],
                                  double        outMACDHist[] )
{
   double *slowMABuffer;
   double *fastMABuffer;
   TA_RetCode retCode;
   int tempInteger;
   int outBegIdx1;
   int outNbElement1;
   int outBegIdx2;
   int outNbElement2;
   int lookbackTotal;
   int lookbackSignal;
   int lookbackLargest;
   int i;
   int tempMAType;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInFastPeriod == TA_INTEGER_DEFAULT )
      optInFastPeriod = 12;
   else if( (int)optInFastPeriod < 2 || (int)optInFastPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInFastMAType == TA_INTEGER_DEFAULT || optInFastMAType == TA_MAType_DEFAULT )
      optInFastMAType = 0;
   else if( (int)optInFastMAType < TA_MATYPE_MIN || (int)optInFastMAType > TA_MATYPE_MAX )
      return TA_BAD_PARAM;
   if( (int)optInSlowPeriod == TA_INTEGER_DEFAULT )
      optInSlowPeriod = 26;
   else if( (int)optInSlowPeriod < 2 || (int)optInSlowPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSlowMAType == TA_INTEGER_DEFAULT || optInSlowMAType == TA_MAType_DEFAULT )
      optInSlowMAType = 0;
   else if( (int)optInSlowMAType < TA_MATYPE_MIN || (int)optInSlowMAType > TA_MATYPE_MAX )
      return TA_BAD_PARAM;
   if( (int)optInSignalPeriod == TA_INTEGER_DEFAULT )
      optInSignalPeriod = 9;
   else if( (int)optInSignalPeriod < 1 || (int)optInSignalPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSignalMAType == TA_INTEGER_DEFAULT || optInSignalMAType == TA_MAType_DEFAULT )
      optInSignalMAType = 0;
   else if( (int)optInSignalMAType < TA_MATYPE_MIN || (int)optInSignalMAType > TA_MATYPE_MAX )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outMACD )
      return TA_BAD_PARAM;
   if( !outMACDSignal )
      return TA_BAD_PARAM;
   if( !outMACDHist )
      return TA_BAD_PARAM;
   if( outMACD == outMACDSignal || outMACD == outMACDHist || outMACDSignal == outMACDHist )
      return TA_BAD_PARAM;

   if( optInFastMAType == TA_MAType_EMA && optInSlowMAType == TA_MAType_EMA && optInSignalMAType == TA_MAType_EMA && optInFastPeriod >= 2 && optInSlowPeriod >= 2 && optInSignalPeriod >= 2 )
   {
      /* An all-EMA MACDEXT computes exactly what MACD computes. Delegate
       * to its single-pass implementation. Period 1 stays on the generic
       * path: ma() copies the input for it instead of running an EMA
       * recursion.
       *
       * This block is a batch-only specialization: the generator strips it
       * from the streaming tier, which composes the general three-MA path for
       * every parameter value. The two agreeing bit for bit is not assumed --
       * stream_verify's multi-enum diagonal selects all-EMA and holds this
       * block to the composed path (issue #181). Keep the comment INSIDE the
       * block: above it, the stream inherits it and reads as if it delegated.
       */
      return TA_MACD(startIdx,endIdx,inReal,optInFastPeriod,optInSlowPeriod,optInSignalPeriod,outBegIdx,outNBElement,outMACD,outMACDSignal,outMACDHist);
   }
   /* Make sure slow is really slower than
    * the fast period! if not, swap...
    */
   if( optInSlowPeriod < optInFastPeriod )
   {
      /* swap period */
      tempInteger = optInSlowPeriod;
      optInSlowPeriod = optInFastPeriod;
      optInFastPeriod = tempInteger;
      /* swap type */
      tempMAType = optInSlowMAType;
      optInSlowMAType = optInFastMAType;
      optInFastMAType = tempMAType;
   }
   /* Find the MA with the largest lookback */
   lookbackLargest = TA_MA_Lookback(optInFastPeriod,optInFastMAType);
   tempInteger = TA_MA_Lookback(optInSlowPeriod,optInSlowMAType);
   if( tempInteger > lookbackLargest )
   {
      lookbackLargest = tempInteger;
   }
   /* Add the lookback needed for the signal line */
   lookbackSignal = TA_MA_Lookback(optInSignalPeriod,optInSignalMAType);
   lookbackTotal = lookbackSignal + lookbackLargest;
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
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   /* Allocate intermediate buffer for fast/slow MA. */
   tempInteger = endIdx - startIdx + 1 + lookbackSignal;
   fastMABuffer = malloc(tempInteger * sizeof(double));
   if( !fastMABuffer )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_ALLOC_ERR;
   }
   slowMABuffer = malloc(tempInteger * sizeof(double));
   if( !slowMABuffer )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      free(fastMABuffer);
      return TA_ALLOC_ERR;
   }
   /* Calculate the slow MA.
    *
    * Move back the startIdx to get enough data
    * for the signal period. That way, once the
    * signal calculation is done, all the output
    * will start at the requested 'startIdx'.
    */
   tempInteger = startIdx - lookbackSignal;
   retCode = TA_MA(tempInteger,endIdx,inReal,optInSlowPeriod,optInSlowMAType,&outBegIdx1,&outNbElement1,slowMABuffer);
   if( retCode != TA_SUCCESS )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      free(fastMABuffer);
      free(slowMABuffer);
      return retCode;
   }
   /* Calculate the fast MA. */
   retCode = TA_MA(tempInteger,endIdx,inReal,optInFastPeriod,optInFastMAType,&outBegIdx2,&outNbElement2,fastMABuffer);
   if( retCode != TA_SUCCESS )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      free(fastMABuffer);
      free(slowMABuffer);
      return retCode;
   }
   /* Parano tests. Will be removed eventually. */
   if( outBegIdx1 != tempInteger || outBegIdx2 != tempInteger || outNbElement1 != outNbElement2 || outNbElement1 != endIdx - startIdx + 1 + lookbackSignal )
   {
      free(fastMABuffer);
      free(slowMABuffer);
      return TA_BAD_PARAM;
   }
   /* Calculate (fast MA) - (slow MA). */
   for( i = 0; i < outNbElement1; i += 1 )
   {
      fastMABuffer[i] = fastMABuffer[i] - slowMABuffer[i];
   }
   /* Copy the result into the output for the caller. */
   /* memmove, not memcpy: fastMABuffer aliases outMACD when the caller buffer is
    * reused as scratch, so source and destination overlap (issue #94).
    */
   memmove(outMACD,&fastMABuffer[lookbackSignal],(endIdx - startIdx + 1) * sizeof(double));
   /* Calculate the signal/trigger line. */
   retCode = TA_MA(0,outNbElement1 - 1,fastMABuffer,optInSignalPeriod,optInSignalMAType,&outBegIdx2,&outNbElement2,outMACDSignal);
   free(fastMABuffer);
   free(slowMABuffer);
   if( retCode != TA_SUCCESS )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   /* Calculate the histogram. */
   for( i = 0; i < outNbElement2; i += 1 )
   {
      outMACDHist[i] = outMACD[i] - outMACDSignal[i];
   }
   /* All done! Indicate the output limits and return success. */
   *outBegIdx= startIdx;
   *outNBElement= outNbElement2;
   return TA_SUCCESS;
}

TA_LIB_API int TA_MACDEXT_Lookback( int optInFastPeriod, TA_MAType optInFastMAType, int optInSlowPeriod, TA_MAType optInSlowMAType, int optInSignalPeriod, TA_MAType optInSignalMAType )
{
   int tempInteger;
   int lookbackLargest;
   if( (int)optInFastPeriod == TA_INTEGER_DEFAULT )
      optInFastPeriod = 12;
   else if( (int)optInFastPeriod < 2 || (int)optInFastPeriod > 100000 )
      return -1;
   if( (int)optInFastMAType == TA_INTEGER_DEFAULT || optInFastMAType == TA_MAType_DEFAULT )
      optInFastMAType = 0;
   else if( (int)optInFastMAType < TA_MATYPE_MIN || (int)optInFastMAType > TA_MATYPE_MAX )
      return -1;
   if( (int)optInSlowPeriod == TA_INTEGER_DEFAULT )
      optInSlowPeriod = 26;
   else if( (int)optInSlowPeriod < 2 || (int)optInSlowPeriod > 100000 )
      return -1;
   if( (int)optInSlowMAType == TA_INTEGER_DEFAULT || optInSlowMAType == TA_MAType_DEFAULT )
      optInSlowMAType = 0;
   else if( (int)optInSlowMAType < TA_MATYPE_MIN || (int)optInSlowMAType > TA_MATYPE_MAX )
      return -1;
   if( (int)optInSignalPeriod == TA_INTEGER_DEFAULT )
      optInSignalPeriod = 9;
   else if( (int)optInSignalPeriod < 1 || (int)optInSignalPeriod > 100000 )
      return -1;
   if( (int)optInSignalMAType == TA_INTEGER_DEFAULT || optInSignalMAType == TA_MAType_DEFAULT )
      optInSignalMAType = 0;
   else if( (int)optInSignalMAType < TA_MATYPE_MIN || (int)optInSignalMAType > TA_MATYPE_MAX )
      return -1;
   /* Find the MA with the largest lookback */
   lookbackLargest = TA_MA_Lookback(optInFastPeriod,optInFastMAType);
   tempInteger = TA_MA_Lookback(optInSlowPeriod,optInSlowMAType);
   if( tempInteger > lookbackLargest )
   {
      lookbackLargest = tempInteger;
   }
   /* Add to the largest MA lookback the signal line lookback */
   return lookbackLargest + TA_MA_Lookback(optInSignalPeriod,optInSignalMAType);
}
