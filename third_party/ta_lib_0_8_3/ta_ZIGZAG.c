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

TA_LIB_API TA_RetCode TA_ZIGZAG( int    startIdx,
                                 int    endIdx,
                                 const double inHigh[],
                                 const double inLow[],
                                 double optInSensitivity,
                                 int optInMinTrendLength,
                                 int          *outBegIdx,
                                 int          *outNBElement,
                                 double        outZigZag[],
                                 int        outTrend[],
                                 int        outPivotIdx[] )
{
   int lookbackTotal;
   int today;
   int outIdx;
   int trend;
   int barsSincePivot;
   int barIdx;
   int isExt;
   double s;
   double up;
   double dn;
   double pivot;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( optInSensitivity == TA_REAL_DEFAULT )
      optInSensitivity = 5;
   else if( !(optInSensitivity >= 0e0 && optInSensitivity <= 1e2) )
      return TA_BAD_PARAM;
   if( (int)optInMinTrendLength == TA_INTEGER_DEFAULT )
      optInMinTrendLength = 1;
   else if( (int)optInMinTrendLength < 1 || (int)optInMinTrendLength > 100000 )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outZigZag )
      return TA_BAD_PARAM;
   if( !outTrend )
      return TA_BAD_PARAM;
   if( !outPivotIdx )
      return TA_BAD_PARAM;
   if( (const void *)outZigZag == (const void *)outTrend || (const void *)outZigZag == (const void *)outPivotIdx || outTrend == outPivotIdx )
      return TA_BAD_PARAM;

   lookbackTotal = TA_ZIGZAG_Lookback(optInSensitivity,optInMinTrendLength);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   if( startIdx > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   /* Percent in, fraction inside. Each factor is rounded once, here, rather
    * than rebuilt per bar: the threshold is `pivot * up` in binary64, which is
    * the expression order the oracle uses.
    */
   s = optInSensitivity / 100.0;
   up = 1.0 + s;
   dn = 1.0 - s;
   /* The seed: a down leg whose extreme is the low of the bar
    * optInMinTrendLength before the first output. Because the gate counts the
    * seed bar like every other pivot, the first reversal can fire exactly at
    * the first output bar, not one later.
    */
   today = startIdx - lookbackTotal;
   trend = -1;
   pivot = inLow[today];
   barIdx = today;
   barsSincePivot = 0;
   today = today + 1;
   outIdx = 0;
   while( today <= endIdx )
   {
      /* The pivot's bar is barIdx - barsSincePivot. Both are carried: a stream
       * has no cursor to read a bar number from.
       */
      barsSincePivot = barsSincePivot + 1;
      barIdx = barIdx + 1;
      /* Reversal is tested BEFORE extension. On an outside bar that both makes
       * a new extreme and clears the threshold, the leg reverses and the old
       * pivot stays where it was; testing extension first would move the pivot
       * and lose the reversal, which is a whole leg of difference rather than a
       * rounding.
       */
      if( trend == -1 )
      {
         if( inHigh[today] >= pivot * up && barsSincePivot >= optInMinTrendLength )
         {
            trend = 1;
            pivot = inHigh[today];
            barsSincePivot = 0;
         } else 
         {
            /* Ties extend: at an equal price the pivot moves to the later bar,
             * which only outPivotIdx shows.
             *
             * A select, not a branch: on moving prices the extension is taken
             * on a share of the bars no predictor follows.
             */
            isExt = (inLow[today] <= pivot) ? 1 : 0;
            pivot = TA_MIN_LT(pivot, inLow[today]);
            barsSincePivot = barsSincePivot - isExt * barsSincePivot;
         }
      } else if( inLow[today] <= pivot * dn && barsSincePivot >= optInMinTrendLength )
      {
         trend = -1;
         pivot = inLow[today];
         barsSincePivot = 0;
      } else 
      {
         isExt = (inHigh[today] >= pivot) ? 1 : 0;
         pivot = TA_MAX_GT(pivot, inHigh[today]);
         barsSincePivot = barsSincePivot - isExt * barsSincePivot;
      }
      if( today >= startIdx )
      {
         outZigZag[outIdx] = pivot;
         outTrend[outIdx] = trend;
         outPivotIdx[outIdx] = barIdx - barsSincePivot;
         outIdx = outIdx + 1;
      }
      today = today + 1;
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_ZIGZAG_Lookback( double optInSensitivity, int optInMinTrendLength )
{
   if( optInSensitivity == TA_REAL_DEFAULT )
      optInSensitivity = 5;
   else if( !(optInSensitivity >= 0e0 && optInSensitivity <= 1e2) )
      return -1;
   if( (int)optInMinTrendLength == TA_INTEGER_DEFAULT )
      optInMinTrendLength = 1;
   else if( (int)optInMinTrendLength < 1 || (int)optInMinTrendLength > 100000 )
      return -1;
   /* The first bar at which a reversal off the seed can fire. The gate counts
    * from the pivot's OWN bar and the seed is a pivot like any other, so the
    * earliest reversal is optInMinTrendLength bars after the seed, which sits
    * that many bars before the first output.
    *
    * Independent of the sensitivity: the threshold decides WHETHER a reversal
    * fires, never how early it may.
    */
   return optInMinTrendLength;
}
