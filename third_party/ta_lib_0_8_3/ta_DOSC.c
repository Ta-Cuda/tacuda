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
TA_LIB_API TA_RetCode TA_DOSC( int    startIdx,
                               int    endIdx,
                               const double inReal[],
                               int optInTimePeriod,
                               int optInFirstPeriod,
                               int optInSecondPeriod,
                               int optInSignalPeriod,
                               int          *outBegIdx,
                               int          *outNBElement,
                               double        outReal[] )
{
   double k1;
   double beta1;
   double k2;
   double beta2;
   double invPeriod;
   double prevGain;
   double prevLoss;
   double prevValue;
   double gainDelta;
   double tempValue1;
   double tempValue2;
   double rsiValue;
   double ema1;
   double ema2;
   double sum1;
   double sum2;
   double sumSignal;
   int lookbackTotal;
   int lookbackRSI;
   int lookbackEMA1;
   int lookbackEMA2;
   int skipRSI;
   int today;
   int i;
   int outIdx;
   int rsiBar;
   int nRsi;
   int n1;
   int n2;
   double local_dsBuffer[32];
   double *dsBuffer = &local_dsBuffer[0];
   int dsBuffer_Idx;
   int maxIdx_dsBuffer;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInFirstPeriod == TA_INTEGER_DEFAULT )
      optInFirstPeriod = 5;
   else if( (int)optInFirstPeriod < 2 || (int)optInFirstPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSecondPeriod == TA_INTEGER_DEFAULT )
      optInSecondPeriod = 3;
   else if( (int)optInSecondPeriod < 2 || (int)optInSecondPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSignalPeriod == TA_INTEGER_DEFAULT )
      optInSignalPeriod = 9;
   else if( (int)optInSignalPeriod < 2 || (int)optInSignalPeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* Constance Brown's "triple smoothed derivative of RSI plotted as a
    * histogram" (MTA Journal, 1994): MACD's histogram construction applied to
    * a double-smoothed RSI.
    *
    *    S1_t   = EMA(RSI(x, t), f)_t
    *    DS_t   = EMA(S1, s)_t
    *    DOSC_t = DS_t - SMA(DS, g)_t
    *
    * The output must stay bit-identical to
    * TA_RSI -> TA_EMA -> TA_EMA -> TA_SMA -> TA_SUB, so every stage keeps its
    * callee's operation order, and every stage boundary is the callee's
    * LOOKBACK, not (period-1): a warm unstable period then folds in.
    *
    * Compare the counters BEFORE subtracting them: the Rust backend renders
    * them usize, where `n = nRsi - skipRSI; if( n >= 0 )` underflows.
    */
   /* This ptr will point on a circular buffer of at least
    * "optInSignalPeriod" element.
    */
   lookbackTotal = TA_DOSC_Lookback(optInTimePeriod,optInFirstPeriod,optInSecondPeriod,optInSignalPeriod);
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
   if( optInSignalPeriod < 1 ) return TA_INTERNAL_ERROR(495);
   if( (int)optInSignalPeriod > (int)(sizeof(local_dsBuffer)/sizeof(double)) )
   {
      dsBuffer = TA_Malloc( sizeof(double)*optInSignalPeriod );
      if( !dsBuffer )
      {
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      dsBuffer = &local_dsBuffer[0];
   }
   maxIdx_dsBuffer = (optInSignalPeriod-1);
   dsBuffer_Idx = 0;
   lookbackRSI = TA_RSI_Lookback(optInTimePeriod);
   lookbackEMA1 = TA_EMA_Lookback(optInFirstPeriod);
   lookbackEMA2 = TA_EMA_Lookback(optInSecondPeriod);
   /* The RSI values the composed chain never publishes. Derive the count
    * from the lookback, never from TA_GetUnstablePeriod: under the Auto
    * levels it is a function of the period.
    */
   skipRSI = lookbackRSI - optInTimePeriod;
   /* k and beta must sum to exactly 1.0, or a flat input drifts off its
    * level. Each subtraction is exact only from an operand in [0.5,1): at a
    * period of 2 that is k, above it beta. Above it beta stays as the divide
    * wrote it: a register last written by a subtraction costs each FMA
    * reading it one more cycle on Intel P-cores.
    */
   beta1 = (double)(optInFirstPeriod - 1) / (double)(optInFirstPeriod + 1);
   k1 = 1.0 - beta1;
   if( beta1 < 0.5 )
   {
      beta1 = 1.0 - k1;
   }
   beta2 = (double)(optInSecondPeriod - 1) / (double)(optInSecondPeriod + 1);
   k2 = 1.0 - beta2;
   if( beta2 < 0.5 )
   {
      beta2 = 1.0 - k2;
   }
   ema1 = 0.0;
   ema2 = 0.0;
   sum1 = 0.0;
   sum2 = 0.0;
   sumSignal = 0.0;
   nRsi = 0;
   invPeriod = 1.0 / (double)optInTimePeriod;
   today = startIdx - lookbackTotal;
   rsiBar = today + optInTimePeriod;
   prevValue = inReal[today];
   prevGain = 0.0;
   prevLoss = 0.0;
   today = today + 1;
   for( i = optInTimePeriod; i > 0; i -= 1 )
   {
      tempValue1 = inReal[today];
      today = today + 1;
      tempValue2 = tempValue1 - prevValue;
      prevValue = tempValue1;
      gainDelta = (tempValue2 > 0.0) ? tempValue2 : 0.0;
      prevGain += gainDelta;
      prevLoss += gainDelta - tempValue2;
   }
   prevLoss *= invPeriod;
   prevGain *= invPeriod;
   tempValue1 = prevGain + prevLoss;
   if( tempValue1 > 0.0 )
   {
      rsiValue = 100.0 * (prevGain / tempValue1);
   } else 
   {
      rsiValue = 50.0;
   }
   /* Warm-up. Feeds every RSI value before startIdx through the chain and
    * leaves rsiValue holding startIdx's own. Nothing is emitted here: the
    * first bar whose signal window is full is startIdx, by construction of
    * the lookback.
    */
   while( rsiBar < startIdx )
   {
      if( nRsi >= skipRSI )
      {
         n1 = nRsi - skipRSI;
         if( n1 < optInFirstPeriod )
         {
            sum1 = sum1 + rsiValue;
            if( n1 == optInFirstPeriod - 1 )
            {
               ema1 = sum1 / optInFirstPeriod;
            }
         } else 
         {
            ema1 = fma(beta1, ema1, k1 * rsiValue);
         }
         if( n1 >= lookbackEMA1 )
         {
            n2 = n1 - lookbackEMA1;
            if( n2 < optInSecondPeriod )
            {
               sum2 = sum2 + ema1;
               if( n2 == optInSecondPeriod - 1 )
               {
                  ema2 = sum2 / optInSecondPeriod;
               }
            } else 
            {
               ema2 = fma(beta2, ema2, k2 * ema1);
            }
            if( n2 >= lookbackEMA2 )
            {
               dsBuffer[dsBuffer_Idx] = ema2;
               sumSignal = sumSignal + ema2;
               dsBuffer_Idx++;
               if( dsBuffer_Idx > maxIdx_dsBuffer ) dsBuffer_Idx = 0;
            }
         }
      }
      nRsi = nRsi + 1;
      tempValue1 = inReal[today];
      today = today + 1;
      tempValue2 = tempValue1 - prevValue;
      prevValue = tempValue1;
      prevLoss *= (double)(optInTimePeriod - 1);
      prevGain *= (double)(optInTimePeriod - 1);
      gainDelta = (tempValue2 > 0.0) ? tempValue2 : 0.0;
      prevGain += gainDelta;
      prevLoss += gainDelta - tempValue2;
      prevLoss *= invPeriod;
      prevGain *= invPeriod;
      tempValue1 = prevGain + prevLoss;
      if( tempValue1 > 0.0 )
      {
         rsiValue = 100.0 * (prevGain / tempValue1);
      } else 
      {
         rsiValue = 50.0;
      }
      rsiBar = rsiBar + 1;
   }
   /* The first output. Every stage is past its seed here, even at periods
    * 2/2/2/2, so the store below and the one in the stable loop are
    * unconditional. Keep them so: under a guard, the managed peek frames
    * seed a dead output local.
    */
   ema1 = fma(beta1, ema1, k1 * rsiValue);
   ema2 = fma(beta2, ema2, k2 * ema1);
   dsBuffer[dsBuffer_Idx] = ema2;
   sumSignal = sumSignal + ema2;
   outReal[0] = ema2 - sumSignal / (double)optInSignalPeriod;
   outIdx = 1;
   dsBuffer_Idx++;
   if( dsBuffer_Idx > maxIdx_dsBuffer ) dsBuffer_Idx = 0;
   sumSignal = sumSignal - dsBuffer[dsBuffer_Idx];
   /* Stable zone. */
   while( today <= endIdx )
   {
      tempValue1 = inReal[today];
      today = today + 1;
      tempValue2 = tempValue1 - prevValue;
      prevValue = tempValue1;
      prevLoss *= (double)(optInTimePeriod - 1);
      prevGain *= (double)(optInTimePeriod - 1);
      gainDelta = (tempValue2 > 0.0) ? tempValue2 : 0.0;
      prevGain += gainDelta;
      prevLoss += gainDelta - tempValue2;
      prevLoss *= invPeriod;
      prevGain *= invPeriod;
      tempValue1 = prevGain + prevLoss;
      if( tempValue1 > 0.0 )
      {
         rsiValue = 100.0 * (prevGain / tempValue1);
      } else 
      {
         rsiValue = 50.0;
      }
      ema1 = fma(beta1, ema1, k1 * rsiValue);
      ema2 = fma(beta2, ema2, k2 * ema1);
      dsBuffer[dsBuffer_Idx] = ema2;
      sumSignal = sumSignal + ema2;
      outReal[outIdx] = ema2 - sumSignal / (double)optInSignalPeriod;
      outIdx = outIdx + 1;
      dsBuffer_Idx++;
      if( dsBuffer_Idx > maxIdx_dsBuffer ) dsBuffer_Idx = 0;
      sumSignal = sumSignal - dsBuffer[dsBuffer_Idx];
   }
   if( dsBuffer != &local_dsBuffer[0] ) { TA_Free( dsBuffer ); dsBuffer = &local_dsBuffer[0]; }
   *outNBElement= outIdx;
   *outBegIdx= startIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_DOSC_Lookback( int optInTimePeriod, int optInFirstPeriod, int optInSecondPeriod, int optInSignalPeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   if( (int)optInFirstPeriod == TA_INTEGER_DEFAULT )
      optInFirstPeriod = 5;
   else if( (int)optInFirstPeriod < 2 || (int)optInFirstPeriod > 100000 )
      return -1;
   if( (int)optInSecondPeriod == TA_INTEGER_DEFAULT )
      optInSecondPeriod = 3;
   else if( (int)optInSecondPeriod < 2 || (int)optInSecondPeriod > 100000 )
      return -1;
   if( (int)optInSignalPeriod == TA_INTEGER_DEFAULT )
      optInSignalPeriod = 9;
   else if( (int)optInSignalPeriod < 2 || (int)optInSignalPeriod > 100000 )
      return -1;
   /* Keep every term a callee's lookback call: that is how DOSC inherits
    * TA_FUNC_UNST_RSI and TA_FUNC_UNST_EMA with no id of its own. The EMA
    * term counts twice, once per smoothing stage.
    */
   return TA_RSI_Lookback(optInTimePeriod) + TA_EMA_Lookback(optInFirstPeriod) + TA_EMA_Lookback(optInSecondPeriod) + TA_SMA_Lookback(optInSignalPeriod);
}
