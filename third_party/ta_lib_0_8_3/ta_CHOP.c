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

TA_LIB_API TA_RetCode TA_CHOP( int    startIdx,
                               int    endIdx,
                               const double inHigh[],
                               const double inLow[],
                               const double inClose[],
                               int optInTimePeriod,
                               int          *outBegIdx,
                               int          *outNBElement,
                               double        outReal[] )
{
   int today;
   int outIdx;
   int lookbackTotal;
   int i;
   double highest;
   double lowest;
   double sumTR;
   double logPeriod;
   double tempHT;
   double tempLT;
   double prevClose;
   double trueHigh;
   double trueLow;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !inClose )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   *outBegIdx= 0;
   *outNBElement= 0;
   lookbackTotal = TA_CHOP_Lookback(optInTimePeriod);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   logPeriod = log10((double)optInTimePeriod);
   outIdx = 0;
   today = startIdx;
   while( today <= endIdx )
   {
      /* Re-summed oldest to newest every bar, never a running total: a
       * running total leaves a residue on a window of zero true ranges, which
       * the exact guard below would then read as a trend.
       */
      highest = inHigh[today];
      lowest = inLow[today];
      sumTR = 0.0;
      prevClose = 0.0;
      for( i = optInTimePeriod; i >= 0; i -= 1 )
      {
         if( i < optInTimePeriod )
         {
            tempHT = inHigh[today - i];
            tempLT = inLow[today - i];
            trueHigh = tempHT;
            if( prevClose > trueHigh )
            {
               trueHigh = prevClose;
            }
            trueLow = tempLT;
            if( prevClose < trueLow )
            {
               trueLow = prevClose;
            }
            sumTR += trueHigh - trueLow;
            if( tempHT > highest )
            {
               highest = tempHT;
            }
            if( tempLT < lowest )
            {
               lowest = tempLT;
            }
         }
         prevClose = inClose[today - i];
      }
      /* Exact tests, never an epsilon band (issue #253). Keep the "<=" form:
       * a NaN box, or a NaN sum over a non-empty box, must reach the log
       * rather than read as 100.
       * Keep the parenthesised quotient: it returns exactly 100 whenever the
       * computed ratio is exactly the period.
       */
      if( highest - lowest <= 0.0 || sumTR <= 0.0 )
      {
         outReal[outIdx] = 100.0;
      } else 
      {
         outReal[outIdx] = 100.0 * (log10(sumTR / (highest - lowest)) / logPeriod);
      }
      outIdx += 1;
      today += 1;
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_CHOP_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   return optInTimePeriod;
}
