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
TA_LIB_API TA_RetCode TA_CKSP( int    startIdx,
                               int    endIdx,
                               const double inHigh[],
                               const double inLow[],
                               const double inClose[],
                               int optInTimePeriod,
                               double optInMultiplier,
                               int optInStopPeriod,
                               int          *outBegIdx,
                               int          *outNBElement,
                               double        outHighStop[],
                               double        outLowStop[] )
{
   double local_hhBuf[30];
   double *hhBuf = &local_hhBuf[0];
   int hhBuf_Idx;
   int maxIdx_hhBuf;
   double local_llBuf[30];
   double *llBuf = &local_llBuf[0];
   double local_hsBuf[30];
   double *hsBuf = &local_hsBuf[0];
   int hsBuf_Idx;
   int maxIdx_hsBuf;
   double local_lsBuf[30];
   double *lsBuf = &local_lsBuf[0];
   int i;
   int ip;
   int iq;
   int today;
   int outIdx;
   int lookbackTotal;
   int lastP;
   int lastQ;
   double prevATR;
   double periodTotal;
   double wAlpha;
   double wBeta;
   double val2;
   double val3;
   double greatest;
   double tempCY;
   double tempLT;
   double tempHT;
   double tempReal;
   double hhPre;
   double llPre;
   double hsPre;
   double lsPre;
   double sufHi;
   double sufLo;
   double highest;
   double lowest;
   double band;
   double highStop;
   double lowStop;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 10;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( optInMultiplier == TA_REAL_DEFAULT )
      optInMultiplier = 1;
   else if( !(optInMultiplier >= 0e0 && optInMultiplier <= TA_REAL_MAX) )
      return TA_BAD_PARAM;
   if( (int)optInStopPeriod == TA_INTEGER_DEFAULT )
      optInStopPeriod = 9;
   else if( (int)optInStopPeriod < 1 || (int)optInStopPeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !inClose )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outHighStop )
      return TA_BAD_PARAM;
   if( !outLowStop )
      return TA_BAD_PARAM;
   if( outHighStop == outLowStop )
      return TA_BAD_PARAM;

   *outBegIdx= 0;
   *outNBElement= 0;
   lookbackTotal = TA_CKSP_Lookback(optInTimePeriod,optInMultiplier,optInStopPeriod);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   /* Four rolling extrema, van Herk / Gil-Werman, one buffer each: the window
    * ending in slot j is the current block's prefix extremum joined with the
    * previous block's suffix extremum from slot j+1. A slot holds the raw value
    * while its block is filling and is turned into the suffix extremum, in
    * place, when the block completes; slot j+1 is always still the previous
    * block's when slot j is filled. The high and low sides share an index.
    *
    * The extrema are exact, so both outputs must stay bit-identical to
    * TA_MAX( TA_MAX(high) - x*TA_ATR ) and its mirror, whatever the block
    * phase; only the sign of a zero tied between +0.0 and -0.0 is free.
    */
   if( optInTimePeriod < 1 ) return TA_INTERNAL_ERROR(474);
   if( (int)optInTimePeriod > (int)(sizeof(local_hhBuf)/sizeof(double)) )
   {
      hhBuf = TA_Malloc( sizeof(double)*optInTimePeriod );
      if( !hhBuf )
      {
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      hhBuf = &local_hhBuf[0];
   }
   maxIdx_hhBuf = (optInTimePeriod-1);
   hhBuf_Idx = 0;
   if( optInTimePeriod < 1 ) return TA_INTERNAL_ERROR(475);
   if( (int)optInTimePeriod > (int)(sizeof(local_llBuf)/sizeof(double)) )
   {
      llBuf = TA_Malloc( sizeof(double)*optInTimePeriod );
      if( !llBuf )
      {
         if( hhBuf != &local_hhBuf[0] ) TA_Free( hhBuf );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      llBuf = &local_llBuf[0];
   }
   if( optInStopPeriod < 1 ) return TA_INTERNAL_ERROR(476);
   if( (int)optInStopPeriod > (int)(sizeof(local_hsBuf)/sizeof(double)) )
   {
      hsBuf = TA_Malloc( sizeof(double)*optInStopPeriod );
      if( !hsBuf )
      {
         if( hhBuf != &local_hhBuf[0] ) TA_Free( hhBuf );
         if( llBuf != &local_llBuf[0] ) TA_Free( llBuf );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      hsBuf = &local_hsBuf[0];
   }
   maxIdx_hsBuf = (optInStopPeriod-1);
   hsBuf_Idx = 0;
   if( optInStopPeriod < 1 ) return TA_INTERNAL_ERROR(477);
   if( (int)optInStopPeriod > (int)(sizeof(local_lsBuf)/sizeof(double)) )
   {
      lsBuf = TA_Malloc( sizeof(double)*optInStopPeriod );
      if( !lsBuf )
      {
         if( hhBuf != &local_hhBuf[0] ) TA_Free( hhBuf );
         if( llBuf != &local_llBuf[0] ) TA_Free( llBuf );
         if( hsBuf != &local_hsBuf[0] ) TA_Free( hsBuf );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      lsBuf = &local_lsBuf[0];
   }
   lastP = optInTimePeriod - 1;
   lastQ = optInStopPeriod - 1;
   /* The Average True Range is carried inline rather than taken from a call,
    * because the two stages advance together one bar at a time and a
    * whole-range buffer between them would not stream.
    *
    * The arithmetic order below is the bit-exactness contract with TA_ATR (do
    * not reorder): True Range from high-low, then the two previous-close
    * distances in that order; the seed summed from 0.0 over the first 'period'
    * True Ranges and divided once; the same two Wilder coefficients, wBeta
    * rounded and wAlpha derived from it, in one fused statement.
    */
   wBeta = (double)(optInTimePeriod - 1) / (double)optInTimePeriod;
   wAlpha = 1.0 - wBeta;
   today = startIdx - lookbackTotal + 1;
   periodTotal = 0.0;
   i = optInTimePeriod;
   while( i-- > 0 )
   {
      tempLT = inLow[today];
      tempHT = inHigh[today];
      tempCY = inClose[today - 1];
      greatest = tempHT - tempLT;
      /* val1 */
      val2 = fabs(tempCY - tempHT);
      if( val2 > greatest )
      {
         greatest = val2;
      }
      val3 = fabs(tempCY - tempLT);
      if( val3 > greatest )
      {
         greatest = val3;
      }
      periodTotal += greatest;
      today += 1;
   }
   prevATR = periodTotal / optInTimePeriod;
   /* Skip the bars the lookback adds for the unstable period. Taking the count
    * from the lookback rather than naming the setting keeps the two from
    * disagreeing.
    */
   i = lookbackTotal - lastQ - optInTimePeriod;
   while( i != 0 )
   {
      tempLT = inLow[today];
      tempHT = inHigh[today];
      tempCY = inClose[today - 1];
      greatest = tempHT - tempLT;
      /* val1 */
      val2 = fabs(tempCY - tempHT);
      if( val2 > greatest )
      {
         greatest = val2;
      }
      val3 = fabs(tempCY - tempLT);
      if( val3 > greatest )
      {
         greatest = val3;
      }
      prevATR = fma(wBeta, prevATR, wAlpha * greatest);
      today += 1;
      i -= 1;
   }
   /* prevATR is now the Average True Range of bar today-1, the first bar with
    * a first stop. Its extreme window is one whole block, taken straight from
    * the input.
    */
   highest = inHigh[today - 1];
   lowest = inLow[today - 1];
   hhBuf[lastP] = highest;
   llBuf[lastP] = lowest;
   ip = lastP;
   while( ip > 0 )
   {
      ip -= 1;
      tempHT = inHigh[today - 1 - lastP + ip];
      tempLT = inLow[today - 1 - lastP + ip];
      if( tempHT > highest )
      {
         highest = tempHT;
      }
      if( tempLT < lowest )
      {
         lowest = tempLT;
      }
      hhBuf[ip] = highest;
      llBuf[ip] = lowest;
   }
   hhPre = highest;
   llPre = lowest;
   /* The multiple of the ATR is formed on its own, never fused into the
    * offset, so that each first stop is TA_MAX - x*TA_ATR (or its mirror) as a
    * caller composing the three functions would compute it.
    */
   band = optInMultiplier * prevATR;
   highStop = highest - band;
   lowStop = lowest + band;
   hsPre = highStop;
   lsPre = lowStop;
   hsBuf[hsBuf_Idx] = highStop;
   lsBuf[hsBuf_Idx] = lowStop;
   hsBuf_Idx++;
   if( hsBuf_Idx > maxIdx_hsBuf ) hsBuf_Idx = 0;
   /* Fill the first stop window, through startIdx inclusive. */
   while( today <= startIdx )
   {
      tempLT = inLow[today];
      tempHT = inHigh[today];
      tempCY = inClose[today - 1];
      greatest = tempHT - tempLT;
      /* val1 */
      val2 = fabs(tempCY - tempHT);
      if( val2 > greatest )
      {
         greatest = val2;
      }
      val3 = fabs(tempCY - tempLT);
      if( val3 > greatest )
      {
         greatest = val3;
      }
      prevATR = fma(wBeta, prevATR, wAlpha * greatest);
      /* Keep this shape: the prefix extreme selected in a local, the
       * block-start override after it, one store. A store made only when the
       * bar sets a new extreme, or the override ahead of the select, compiles
       * to a data-dependent branch in the stream step, where the prefix lives
       * in the handle.
       */
      highest = hhPre;
      lowest = llPre;
      if( tempHT > highest )
      {
         highest = tempHT;
      }
      if( tempLT < lowest )
      {
         lowest = tempLT;
      }
      if( hhBuf_Idx == 0 )
      {
         highest = tempHT;
         lowest = tempLT;
      }
      hhPre = highest;
      llPre = lowest;
      if( hhBuf_Idx < lastP )
      {
         tempReal = hhBuf[hhBuf_Idx + 1];
         if( tempReal > highest )
         {
            highest = tempReal;
         }
         tempReal = llBuf[hhBuf_Idx + 1];
         if( tempReal < lowest )
         {
            lowest = tempReal;
         }
      }
      hhBuf[hhBuf_Idx] = tempHT;
      llBuf[hhBuf_Idx] = tempLT;
      hhBuf_Idx++;
      if( hhBuf_Idx > maxIdx_hhBuf ) hhBuf_Idx = 0;
      if( hhBuf_Idx == 0 )
      {
         sufHi = hhBuf[lastP];
         sufLo = llBuf[lastP];
         ip = lastP;
         while( ip > 1 )
         {
            ip -= 1;
            tempReal = hhBuf[ip];
            if( tempReal > sufHi )
            {
               sufHi = tempReal;
            }
            hhBuf[ip] = sufHi;
            tempReal = llBuf[ip];
            if( tempReal < sufLo )
            {
               sufLo = tempReal;
            }
            llBuf[ip] = sufLo;
         }
      }
      band = optInMultiplier * prevATR;
      highStop = highest - band;
      lowStop = lowest + band;
      if( highStop > hsPre )
      {
         hsPre = highStop;
      }
      if( lowStop < lsPre )
      {
         lsPre = lowStop;
      }
      hsBuf[hsBuf_Idx] = highStop;
      lsBuf[hsBuf_Idx] = lowStop;
      hsBuf_Idx++;
      if( hsBuf_Idx > maxIdx_hsBuf ) hsBuf_Idx = 0;
      today += 1;
   }
   /* The first stop window is one whole block as well: its extremes are the
    * prefix extremes, and the block is complete.
    */
   sufHi = hsBuf[lastQ];
   sufLo = lsBuf[lastQ];
   iq = lastQ;
   while( iq > 1 )
   {
      iq -= 1;
      tempReal = hsBuf[iq];
      if( tempReal > sufHi )
      {
         sufHi = tempReal;
      }
      hsBuf[iq] = sufHi;
      tempReal = lsBuf[iq];
      if( tempReal < sufLo )
      {
         sufLo = tempReal;
      }
      lsBuf[iq] = sufLo;
   }
   outHighStop[0] = hsPre;
   outLowStop[0] = lsPre;
   outIdx = 1;
   while( today <= endIdx )
   {
      tempLT = inLow[today];
      tempHT = inHigh[today];
      tempCY = inClose[today - 1];
      greatest = tempHT - tempLT;
      /* val1 */
      val2 = fabs(tempCY - tempHT);
      if( val2 > greatest )
      {
         greatest = val2;
      }
      val3 = fabs(tempCY - tempLT);
      if( val3 > greatest )
      {
         greatest = val3;
      }
      prevATR = fma(wBeta, prevATR, wAlpha * greatest);
      highest = hhPre;
      lowest = llPre;
      if( tempHT > highest )
      {
         highest = tempHT;
      }
      if( tempLT < lowest )
      {
         lowest = tempLT;
      }
      if( hhBuf_Idx == 0 )
      {
         highest = tempHT;
         lowest = tempLT;
      }
      hhPre = highest;
      llPre = lowest;
      if( hhBuf_Idx < lastP )
      {
         tempReal = hhBuf[hhBuf_Idx + 1];
         if( tempReal > highest )
         {
            highest = tempReal;
         }
         tempReal = llBuf[hhBuf_Idx + 1];
         if( tempReal < lowest )
         {
            lowest = tempReal;
         }
      }
      hhBuf[hhBuf_Idx] = tempHT;
      llBuf[hhBuf_Idx] = tempLT;
      hhBuf_Idx++;
      if( hhBuf_Idx > maxIdx_hhBuf ) hhBuf_Idx = 0;
      band = optInMultiplier * prevATR;
      highStop = highest - band;
      lowStop = lowest + band;
      highest = hsPre;
      lowest = lsPre;
      if( highStop > highest )
      {
         highest = highStop;
      }
      if( lowStop < lowest )
      {
         lowest = lowStop;
      }
      if( hsBuf_Idx == 0 )
      {
         highest = highStop;
         lowest = lowStop;
      }
      hsPre = highest;
      lsPre = lowest;
      if( hsBuf_Idx < lastQ )
      {
         tempReal = hsBuf[hsBuf_Idx + 1];
         if( tempReal > highest )
         {
            highest = tempReal;
         }
         tempReal = lsBuf[hsBuf_Idx + 1];
         if( tempReal < lowest )
         {
            lowest = tempReal;
         }
      }
      hsBuf[hsBuf_Idx] = highStop;
      lsBuf[hsBuf_Idx] = lowStop;
      hsBuf_Idx++;
      if( hsBuf_Idx > maxIdx_hsBuf ) hsBuf_Idx = 0;
      outHighStop[outIdx] = highest;
      outLowStop[outIdx] = lowest;
      /* A completed block becomes its suffix extrema. Nothing this bar reads
       * them, so both passes stay below the output stores, where the peek
       * frame never runs them.
       */
      if( hhBuf_Idx == 0 )
      {
         sufHi = hhBuf[lastP];
         sufLo = llBuf[lastP];
         ip = lastP;
         while( ip > 1 )
         {
            ip -= 1;
            tempReal = hhBuf[ip];
            if( tempReal > sufHi )
            {
               sufHi = tempReal;
            }
            hhBuf[ip] = sufHi;
            tempReal = llBuf[ip];
            if( tempReal < sufLo )
            {
               sufLo = tempReal;
            }
            llBuf[ip] = sufLo;
         }
      }
      if( hsBuf_Idx == 0 )
      {
         sufHi = hsBuf[lastQ];
         sufLo = lsBuf[lastQ];
         iq = lastQ;
         while( iq > 1 )
         {
            iq -= 1;
            tempReal = hsBuf[iq];
            if( tempReal > sufHi )
            {
               sufHi = tempReal;
            }
            hsBuf[iq] = sufHi;
            tempReal = lsBuf[iq];
            if( tempReal < sufLo )
            {
               sufLo = tempReal;
            }
            lsBuf[iq] = sufLo;
         }
      }
      outIdx += 1;
      today += 1;
   }
   if( hhBuf != &local_hhBuf[0] ) { TA_Free( hhBuf ); hhBuf = &local_hhBuf[0]; }
   if( llBuf != &local_llBuf[0] ) { TA_Free( llBuf ); llBuf = &local_llBuf[0]; }
   if( hsBuf != &local_hsBuf[0] ) { TA_Free( hsBuf ); hsBuf = &local_hsBuf[0]; }
   if( lsBuf != &local_lsBuf[0] ) { TA_Free( lsBuf ); lsBuf = &local_lsBuf[0]; }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_CKSP_Lookback( int optInTimePeriod, double optInMultiplier, int optInStopPeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 10;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   if( optInMultiplier == TA_REAL_DEFAULT )
      optInMultiplier = 1;
   else if( !(optInMultiplier >= 0e0 && optInMultiplier <= TA_REAL_MAX) )
      return -1;
   if( (int)optInStopPeriod == TA_INTEGER_DEFAULT )
      optInStopPeriod = 9;
   else if( (int)optInStopPeriod < 1 || (int)optInStopPeriod > 100000 )
      return -1;
   /* The first stops need the Average True Range at their own bar, and the
    * stop window reaches optInStopPeriod-1 first stops further back. The ATR
    * term is never restated here, which is what makes CKSP inherit
    * TA_FUNC_UNST_ATR.
    *
    * A stop can rest on a first stop optInStopPeriod-1 bars old, whose ATR
    * was that much closer to its seed: the first difference two starts show
    * is the smaller for it, and an Auto level is held against that one.
    */
   return TA_ATR_Lookback(optInTimePeriod) + optInStopPeriod - 1 + TA_GLOBALS_UNSTABLE_OFFSET(TA_FUNC_UNST_ATR,cksp_auto_offset(optInTimePeriod, optInMultiplier, optInStopPeriod));
}
