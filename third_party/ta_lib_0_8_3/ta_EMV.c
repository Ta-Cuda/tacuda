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

TA_LIB_API TA_RetCode TA_EMV( int    startIdx,
                              int    endIdx,
                              const double inHigh[],
                              const double inLow[],
                              const double inVolume[],
                              int optInTimePeriod,
                              double optInVolumeDivisor,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outReal[] )
{
   double periodTotal;
   double prevMid;
   double mid;
   double range;
   double boxRatio;
   double raw;
   double tempReal;
   int lookbackTotal;
   int outIdx;
   int i;
   double local_rawBuffer[50];
   double *rawBuffer = &local_rawBuffer[0];
   int rawBuffer_Idx;
   int maxIdx_rawBuffer;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( optInVolumeDivisor == TA_REAL_DEFAULT )
      optInVolumeDivisor = 10000;
   else if( !(optInVolumeDivisor >= 1e0 && optInVolumeDivisor <= TA_REAL_MAX) )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !inVolume )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* The one-bar values are kept in a ring rather than recomputed at the
    * trailing index: outReal may alias an input, and those bars may already
    * hold outputs.
    */
   lookbackTotal = TA_EMV_Lookback(optInTimePeriod,optInVolumeDivisor);
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
   if( optInTimePeriod < 1 ) return TA_INTERNAL_ERROR(440);
   if( (int)optInTimePeriod > (int)(sizeof(local_rawBuffer)/sizeof(double)) )
   {
      rawBuffer = TA_Malloc( sizeof(double)*optInTimePeriod );
      if( !rawBuffer )
      {
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      rawBuffer = &local_rawBuffer[0];
   }
   maxIdx_rawBuffer = (optInTimePeriod-1);
   rawBuffer_Idx = 0;
   /* Keep the operand order and sma.c's add / snapshot / subtract order: the
    * composite gate rebuilds this from MEDPRICE, MOM, SUB, DIV and SMA and
    * compares bitwise.
    *
    * The zero guards are exact != 0.0 tests, never TA_IS_ZERO: the output
    * scale follows the volume and the divisor, so no absolute band fits. A
    * nonzero range with a box ratio that underflows to 0 takes the guard too.
    */
   i = startIdx - lookbackTotal;
   prevMid = (inHigh[i] + inLow[i]) / 2.0;
   i = i + 1;
   periodTotal = 0.0;
   while( i < startIdx )
   {
      mid = (inHigh[i] + inLow[i]) / 2.0;
      range = inHigh[i] - inLow[i];
      raw = 0.0;
      if( range != 0.0 )
      {
         boxRatio = inVolume[i] / optInVolumeDivisor / range;
         if( boxRatio != 0.0 )
         {
            raw = (mid - prevMid) / boxRatio;
         }
      }
      prevMid = mid;
      i = i + 1;
      rawBuffer[rawBuffer_Idx] = raw;
      periodTotal += raw;
      rawBuffer_Idx++;
      if( rawBuffer_Idx > maxIdx_rawBuffer ) rawBuffer_Idx = 0;
   }
   outIdx = 0;
   while( i <= endIdx )
   {
      mid = (inHigh[i] + inLow[i]) / 2.0;
      range = inHigh[i] - inLow[i];
      raw = 0.0;
      if( range != 0.0 )
      {
         boxRatio = inVolume[i] / optInVolumeDivisor / range;
         if( boxRatio != 0.0 )
         {
            raw = (mid - prevMid) / boxRatio;
         }
      }
      prevMid = mid;
      i = i + 1;
      rawBuffer[rawBuffer_Idx] = raw;
      periodTotal += raw;
      tempReal = periodTotal;
      rawBuffer_Idx++;
      if( rawBuffer_Idx > maxIdx_rawBuffer ) rawBuffer_Idx = 0;
      periodTotal -= rawBuffer[rawBuffer_Idx];
      outReal[outIdx] = tempReal / (double)optInTimePeriod;
      outIdx = outIdx + 1;
   }
   if( rawBuffer != &local_rawBuffer[0] ) { TA_Free( rawBuffer ); rawBuffer = &local_rawBuffer[0]; }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_EMV_Lookback( int optInTimePeriod, double optInVolumeDivisor )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   if( optInVolumeDivisor == TA_REAL_DEFAULT )
      optInVolumeDivisor = 10000;
   else if( !(optInVolumeDivisor >= 1e0 && optInVolumeDivisor <= TA_REAL_MAX) )
      return -1;
   return 1 + TA_SMA_Lookback(optInTimePeriod);
}
