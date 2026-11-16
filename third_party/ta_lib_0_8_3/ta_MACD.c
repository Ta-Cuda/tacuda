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
TA_LIB_API TA_RetCode TA_MACD( int    startIdx,
                               int    endIdx,
                               const double inReal[],
                               int optInFastPeriod,
                               int optInSlowPeriod,
                               int optInSignalPeriod,
                               int          *outBegIdx,
                               int          *outNBElement,
                               double        outMACD[],
                               double        outMACDSignal[],
                               double        outMACDHist[] )
{
   double prevFast;
   double prevSlow;
   double prevSignal;
   double macdValue;
   double tempReal;
   double slowK;
   double fastK;
   double signalK;
   double slowBeta;
   double fastBeta;
   double signalBeta;
   int i;
   int today;
   int outIdx;
   int tempInteger;
   int lookbackTotal;
   int lookbackSignal;
   int lookbackSlow;
   int fastToday;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInFastPeriod == TA_INTEGER_DEFAULT )
      optInFastPeriod = 12;
   else if( (int)optInFastPeriod < 2 || (int)optInFastPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSlowPeriod == TA_INTEGER_DEFAULT )
      optInSlowPeriod = 26;
   else if( (int)optInSlowPeriod < 2 || (int)optInSlowPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSignalPeriod == TA_INTEGER_DEFAULT )
      optInSignalPeriod = 9;
   else if( (int)optInSignalPeriod < 1 || (int)optInSignalPeriod > 100000 )
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

   /* Make sure slow is really slower than
    * the fast period! if not, swap...
    */
   if( optInSlowPeriod < optInFastPeriod )
   {
      /* swap */
      tempInteger = optInSlowPeriod;
      optInSlowPeriod = optInFastPeriod;
      optInFastPeriod = tempInteger;
   }
   /* The fixed 26/12 MACD: k of 0.075 and 0.15, as near as a pair summing
    * to exactly 1.0 comes.
    */
   if( optInSlowPeriod == 0 )
   {
      /* Fix 26 */
      optInSlowPeriod = 26;
      slowBeta = 1.0 - 0.075;
   } else 
   {
      slowBeta = (double)(optInSlowPeriod - 1) / (double)(optInSlowPeriod + 1);
   }
   slowK = 1.0 - slowBeta;
   if( slowBeta < 0.5 )
   {
      slowBeta = 1.0 - slowK;
   }
   if( optInFastPeriod == 0 )
   {
      /* Fix 12 */
      optInFastPeriod = 12;
      fastBeta = 1.0 - 0.15;
   } else 
   {
      fastBeta = (double)(optInFastPeriod - 1) / (double)(optInFastPeriod + 1);
   }
   fastK = 1.0 - fastBeta;
   if( fastBeta < 0.5 )
   {
      fastBeta = 1.0 - fastK;
   }
   /* A signal period of 1 disables signal-line smoothing: the signal IS the
    * MACD line and the histogram is exactly zero. The recursion
    * below, at a k of 1.0 and a beta of 0.0, does not keep the sign of a
    * -0.0 line value; hence the explicit arm at each step.
    */
   signalBeta = (double)(optInSignalPeriod - 1) / (double)(optInSignalPeriod + 1);
   signalK = 1.0 - signalBeta;
   if( signalBeta < 0.5 )
   {
      signalBeta = 1.0 - signalK;
   }
   lookbackSignal = TA_EMA_Lookback(optInSignalPeriod);
   /* Move up the start index if there is not
    * enough initial data.
    */
   lookbackSlow = TA_EMA_Lookback(optInSlowPeriod);
   lookbackTotal = lookbackSignal + lookbackSlow;
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
   /* Everything is computed in a single lockstep pass: each bar
    * advances the fast and slow EMA (two independent recursions),
    * their difference is the MACD line, and each MACD-line value
    * is immediately fed into the signal EMA. No temporary buffers.
    *
    * The arithmetic order below is the bit-exactness contract
    * (do not reorder or fuse operations):
    *  - EMA recursion: k*x + beta*prev, with ema.c's k and beta.
    *  - Each EMA is seeded with the sum of its first 'period'
    *    inputs, accumulated from 0.0 in input order, divided by
    *    the period. The signal EMA is seeded the same way from the
    *    first 'signal period' MACD-line values.
    *
    * In-place (an output == inReal) is supported: outputs at
    * [outIdx] are written only after inReal[startIdx+outIdx] was
    * read.
    */
   /* Seed each price EMA with a simple average of its first
    * 'period' price bars, each window placed by that EMA's own
    * lookback, so that the line is TA_EMA(fast) - TA_EMA(slow) bit
    * for bit. The slow EMA then runs alone to the end of the fast
    * window.
    *
    * ema_lookback(n) - n must never decrease as n grows: a fast
    * window ending before the slow one would skip bars of the fast
    * EMA, and one starting before it would read below the lookback.
    */
   today = startIdx - lookbackTotal;
   tempReal = 0.0;
   i = optInSlowPeriod;
   while( i-- > 0 )
   {
      tempReal += inReal[today++];
   }
   prevSlow = tempReal / optInSlowPeriod;
   fastToday = startIdx - lookbackTotal + (lookbackSlow - TA_EMA_Lookback(optInFastPeriod));
   prevFast = 0.0;
   i = optInFastPeriod;
   while( i-- > 0 )
   {
      prevFast += inReal[fastToday++];
   }
   prevFast = prevFast / optInFastPeriod;
   while( today < fastToday )
   {
      tempReal = inReal[today++];
      prevSlow = fma(slowBeta, prevSlow, slowK * tempReal);
   }
   /* Advance both EMA through their unstable period, up to the
    * first MACD-line bar.
    */
   while( today <= startIdx - lookbackSignal )
   {
      tempReal = inReal[today++];
      prevFast = fma(fastBeta, prevFast, fastK * tempReal);
      prevSlow = fma(slowBeta, prevSlow, slowK * tempReal);
   }
   macdValue = prevFast - prevSlow;
   /* Seed the signal EMA with a simple average of the first
    * 'signal period' MACD-line values, accumulated as they are
    * produced.
    */
   prevSignal = 0.0;
   prevSignal += macdValue;
   i = optInSignalPeriod - 1;
   while( i-- > 0 )
   {
      tempReal = inReal[today++];
      prevFast = fma(fastBeta, prevFast, fastK * tempReal);
      prevSlow = fma(slowBeta, prevSlow, slowK * tempReal);
      macdValue = prevFast - prevSlow;
      prevSignal += macdValue;
   }
   prevSignal = prevSignal / optInSignalPeriod;
   /* Advance everything in lockstep through the unstable period
    * of the signal EMA, up to the first output bar.
    */
   while( today <= startIdx )
   {
      tempReal = inReal[today++];
      prevFast = fma(fastBeta, prevFast, fastK * tempReal);
      prevSlow = fma(slowBeta, prevSlow, slowK * tempReal);
      macdValue = prevFast - prevSlow;
      if( optInSignalPeriod == 1 )
      {
         prevSignal = macdValue;
      } else 
      {
         prevSignal = fma(signalBeta, prevSignal, signalK * macdValue);
      }
   }
   /* Stable zone: keep advancing in lockstep and write the three
    * outputs.
    */
   outMACD[0] = macdValue;
   outMACDSignal[0] = prevSignal;
   outMACDHist[0] = macdValue - prevSignal;
   outIdx = 1;
   while( today <= endIdx )
   {
      tempReal = inReal[today++];
      prevFast = fma(fastBeta, prevFast, fastK * tempReal);
      prevSlow = fma(slowBeta, prevSlow, slowK * tempReal);
      macdValue = prevFast - prevSlow;
      if( optInSignalPeriod == 1 )
      {
         prevSignal = macdValue;
      } else 
      {
         prevSignal = fma(signalBeta, prevSignal, signalK * macdValue);
      }
      outMACD[outIdx] = macdValue;
      outMACDSignal[outIdx] = prevSignal;
      outMACDHist[outIdx] = macdValue - prevSignal;
      outIdx += 1;
   }
   /* All done! Indicate the output limits and return success. */
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}
