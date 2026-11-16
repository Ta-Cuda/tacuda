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

TA_LIB_API TA_RetCode TA_DONCHIAN( int    startIdx,
                                   int    endIdx,
                                   const double inHigh[],
                                   const double inLow[],
                                   int optInTimePeriod,
                                   int          *outBegIdx,
                                   int          *outNBElement,
                                   double        outRealUpperBand[],
                                   double        outRealMiddleBand[],
                                   double        outRealLowerBand[] )
{
   double lowest;
   double highest;
   double tmpLow;
   double tmpHigh;
   int outIdx;
   int nbInitialElementNeeded;
   int trailingIdx;
   int lowestIdx;
   int highestIdx;
   int today;
   int i;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outRealUpperBand )
      return TA_BAD_PARAM;
   if( !outRealMiddleBand )
      return TA_BAD_PARAM;
   if( !outRealLowerBand )
      return TA_BAD_PARAM;
   if( outRealUpperBand == outRealMiddleBand || outRealUpperBand == outRealLowerBand || outRealMiddleBand == outRealLowerBand )
      return TA_BAD_PARAM;

   /* Donchian Channels over the optInTimePeriod bars ending at the current
    * bar:
    *
    *    Upper  = Highest High of the window
    *    Lower  = Lowest  Low  of the window
    *    Middle = (Upper + Lower) / 2
    *
    * The window includes the current bar, matching every other library and
    * charting platform. A breakout rule compares the current bar against the
    * PREVIOUS bar's band, which is where the one-bar offset belongs.
    *
    * Upper/Middle/Lower are bit-identical to MAX(high,N)/MIDPRICE(N)/MIN(low,N).
    */
   /* Identify the minimum number of price bar needed
    * to identify at least one output over the specified
    * period.
    */
   nbInitialElementNeeded = optInTimePeriod - 1;
   /* Move up the start index if there is not
    * enough initial data.
    */
   if( startIdx < nbInitialElementNeeded )
   {
      startIdx = nbInitialElementNeeded;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   /* Proceed with the calculation for the requested range.
    * Note that this algorithm allows the input and
    * output to be the same buffer: every position written (outIdx) sits
    * at or below trailingIdx, the oldest position any later bar reads.
    *
    * The highest high and lowest low of the window are cached with their
    * indices; the window is rescanned only when a cached extremum drops
    * out of it (same approach as MIN/MAX/WILLR and MIDPRICE).
    */
   outIdx = 0;
   today = startIdx;
   trailingIdx = startIdx - nbInitialElementNeeded;
   highestIdx = -1;
   highest = 0.0;
   lowestIdx = -1;
   lowest = 0.0;
   while( today <= endIdx )
   {
      tmpHigh = inHigh[today];
      tmpLow = inLow[today];
      if( highestIdx < trailingIdx )
      {
         highestIdx = trailingIdx;
         highest = inHigh[highestIdx];
         i = highestIdx;
         TA_UNROLL(4)
         while( ++i <= today )
         {
            tmpHigh = inHigh[i];
            if( tmpHigh > highest )
            {
               highestIdx = i;
               highest = tmpHigh;
            }
         }
      } else if( tmpHigh >= highest )
      {
         highestIdx = today;
         highest = tmpHigh;
      }
      if( lowestIdx < trailingIdx )
      {
         lowestIdx = trailingIdx;
         lowest = inLow[lowestIdx];
         i = lowestIdx;
         TA_UNROLL(4)
         while( ++i <= today )
         {
            tmpLow = inLow[i];
            if( tmpLow < lowest )
            {
               lowestIdx = i;
               lowest = tmpLow;
            }
         }
      } else if( tmpLow <= lowest )
      {
         lowestIdx = today;
         lowest = tmpLow;
      }
      outRealUpperBand[outIdx] = highest;
      outRealLowerBand[outIdx] = lowest;
      outRealMiddleBand[outIdx] = (highest + lowest) / 2.0;
      outIdx += 1;
      trailingIdx += 1;
      today += 1;
   }
   /* Keep the outBegIdx relative to the
    * caller input before returning.
    */
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_DONCHIAN_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   return optInTimePeriod - 1;
}
