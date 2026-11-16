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
TA_LIB_API TA_RetCode TA_FOSC( int    startIdx,
                               int    endIdx,
                               const double inReal[],
                               int optInTimePeriod,
                               int          *outBegIdx,
                               int          *outNBElement,
                               double        outReal[] )
{
   int outIdx;
   int today;
   int lookbackTotal;
   int trailingIdx;
   double SumX;
   double SumXY;
   double SumY;
   double SumXSqr;
   double Divisor;
   double m;
   double b;
   double closeValue;
   int i;
   int j;
   int windowStart;
   int barsSinceReseed;
   double tempValue1;
   double tempValue2;
   double trailingValue;
   double weightedTrailing;
   double sumAbs;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 5;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* The forecast compared against the close is the one made one bar EARLIER:
    * FOSC[t] = 100*(inReal[t] - TSF[t-1])/inReal[t]. The regression window
    * ends at t-1, so the lookback is period, not period-1.
    *
    * The window arithmetic below is TA_TSF's verbatim -- priming scan, O(1)
    * recurrence, both re-anchor triggers (#254) and the fused
    * `b + m*(double)optInTimePeriod` -- which is what makes FOSC bit-identical
    * to a TA_TSF call anchored one bar earlier. Reshaping any of it breaks
    * that silently.
    *
    * trailingValue -- the value the NEXT bar's window drops -- is read before
    * the output write because with outReal==inReal (#130) that write lands on
    * exactly that cell whenever startIdx is the clamped minimum. closeValue
    * carries no such constraint: it sits startIdx bars ahead of the cursor.
    */
   lookbackTotal = TA_FOSC_Lookback(optInTimePeriod);
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
   outIdx = 0;
   today = startIdx;
   trailingIdx = startIdx - lookbackTotal;
   SumX = (double)optInTimePeriod * (optInTimePeriod - 1) * 0.5;
   SumXSqr = (double)optInTimePeriod * (optInTimePeriod - 1) * (2 * optInTimePeriod - 1) / 6.0;
   Divisor = SumX * SumX - optInTimePeriod * SumXSqr;
   SumXY = 0;
   SumY = 0;
   sumAbs = 0;
   for( i = optInTimePeriod; i-- != 0;  )
   {
      tempValue1 = inReal[today - 1 - i];
      SumY += tempValue1;
      SumXY += (double)i * tempValue1;
      sumAbs += fabs(tempValue1);
   }
   m = (optInTimePeriod * SumXY - SumX * SumY) / Divisor;
   b = (SumY - m * SumX) / (double)optInTimePeriod;
   barsSinceReseed = 32 * optInTimePeriod;
   trailingValue = inReal[trailingIdx];
   trailingIdx += 1;
   closeValue = inReal[today];
   if( closeValue != 0.0 )
   {
      outReal[outIdx++] = 100.0 * (closeValue - (fma(m, (double)optInTimePeriod, b))) / closeValue;
   } else 
   {
      outReal[outIdx++] = 0.0;
   }
   today += 1;
   while( today <= endIdx )
   {
      weightedTrailing = (double)optInTimePeriod * trailingValue;
      SumXY = SumXY + SumY - weightedTrailing;
      SumY = SumY - trailingValue + inReal[today - 1];
      sumAbs = sumAbs - fabs(trailingValue) + fabs(inReal[today - 1]);
      barsSinceReseed -= 1;
      if( barsSinceReseed <= 0 || fabs(weightedTrailing) > 100.0 * sumAbs )
      {
         barsSinceReseed = 32 * optInTimePeriod;
         windowStart = today - lookbackTotal;
         SumY = 0;
         SumXY = 0;
         sumAbs = 0;
         tempValue2 = (double)(optInTimePeriod - 1);
         for( j = windowStart; j < today; j += 1 )
         {
            tempValue1 = inReal[j];
            SumY += tempValue1;
            SumXY += tempValue2 * tempValue1;
            sumAbs += fabs(tempValue1);
            tempValue2 -= 1.0;
         }
      }
      m = (optInTimePeriod * SumXY - SumX * SumY) / Divisor;
      b = (SumY - m * SumX) / (double)optInTimePeriod;
      trailingValue = inReal[trailingIdx];
      trailingIdx += 1;
      closeValue = inReal[today];
      if( closeValue != 0.0 )
      {
         outReal[outIdx++] = 100.0 * (closeValue - (fma(m, (double)optInTimePeriod, b))) / closeValue;
      } else 
      {
         outReal[outIdx++] = 0.0;
      }
      today += 1;
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_FOSC_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 5;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   return optInTimePeriod;
}
