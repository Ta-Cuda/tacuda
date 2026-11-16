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
TA_LIB_API TA_RetCode TA_CVI( int    startIdx,
                              int    endIdx,
                              const double inHigh[],
                              const double inLow[],
                              int optInTimePeriod,
                              int optInROCPeriod,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outReal[] )
{
   double prevEMA;
   double laggedEMA;
   double tempReal;
   double optInK_1;
   double emaBeta;
   int i;
   int today;
   int outIdx;
   int lookbackTotal;
   double local_emaRing[32];
   double *emaRing = &local_emaRing[0];
   int emaRing_Idx;
   int maxIdx_emaRing;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 10;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInROCPeriod == TA_INTEGER_DEFAULT )
      optInROCPeriod = 10;
   else if( (int)optInROCPeriod < 1 || (int)optInROCPeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* CVI[t] = 100 * (E[t] - E[t-optInROCPeriod]) / E[t-optInROCPeriod], with E
    * an EMA of the high-low spread. The spread is never materialised and the
    * EMA is anchored optInROCPeriod bars behind startIdx.
    *
    * The arithmetic below is TA_EMA's and TA_ROCP's verbatim -- seed sum
    * accumulated from 0.0 in ascending bar order, k*x + beta*prev, and
    * 100*((a-b)/b) under an exact zero test. That is what makes this fused pass
    * bit-identical to composing TA_SUB, TA_EMA and TA_ROCP, which test_cvi.c
    * holds it to memcmp-exact; reshaping any of it breaks that silently. The
    * guard stays an exact `!= 0.0` and never TA_IS_ZERO -- an epsilon band
    * carries the quote unit and would zero the indicator for anything priced
    * under it (issue #253).
    */
   *outBegIdx= 0;
   *outNBElement= 0;
   lookbackTotal = TA_EMA_Lookback(optInTimePeriod) + TA_ROCP_Lookback(optInROCPeriod);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   if( optInROCPeriod < 1 ) return TA_INTERNAL_ERROR(416);
   if( (int)optInROCPeriod > (int)(sizeof(local_emaRing)/sizeof(double)) )
   {
      emaRing = TA_Malloc( sizeof(double)*optInROCPeriod );
      if( !emaRing )
      {
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      emaRing = &local_emaRing[0];
   }
   maxIdx_emaRing = (optInROCPeriod-1);
   emaRing_Idx = 0;
   emaBeta = (double)(optInTimePeriod - 1) / (double)(optInTimePeriod + 1);
   optInK_1 = 1.0 - emaBeta;
   if( emaBeta < 0.5 )
   {
      emaBeta = 1.0 - optInK_1;
   }
   today = startIdx - lookbackTotal;
   i = optInTimePeriod;
   tempReal = 0.0;
   while( i-- > 0 )
   {
      tempReal += inHigh[today] - inLow[today];
      today += 1;
   }
   prevEMA = tempReal / optInTimePeriod;
   /* The ring keeps only the newest optInROCPeriod values, so pushing every EMA
    * value from the seed bar on leaves exactly the lagged terms the output
    * loop reads.
    */
   emaRing[emaRing_Idx] = prevEMA;
   emaRing_Idx++;
   if( emaRing_Idx > maxIdx_emaRing ) emaRing_Idx = 0;
   while( today < startIdx )
   {
      tempReal = inHigh[today] - inLow[today];
      prevEMA = fma(emaBeta, prevEMA, optInK_1 * tempReal);
      today += 1;
      emaRing[emaRing_Idx] = prevEMA;
      emaRing_Idx++;
      if( emaRing_Idx > maxIdx_emaRing ) emaRing_Idx = 0;
   }
   /* Read the expiring slot before overwriting it: that is what makes the lag
    * exactly optInROCPeriod rather than one less.
    */
   outIdx = 0;
   while( today <= endIdx )
   {
      tempReal = inHigh[today] - inLow[today];
      prevEMA = fma(emaBeta, prevEMA, optInK_1 * tempReal);
      today += 1;
      laggedEMA = emaRing[emaRing_Idx];
      emaRing[emaRing_Idx] = prevEMA;
      emaRing_Idx++;
      if( emaRing_Idx > maxIdx_emaRing ) emaRing_Idx = 0;
      if( laggedEMA != 0.0 )
      {
         outReal[outIdx++] = 100.0 * ((prevEMA - laggedEMA) / laggedEMA);
      } else 
      {
         outReal[outIdx++] = 0.0;
      }
   }
   if( emaRing != &local_emaRing[0] ) { TA_Free( emaRing ); emaRing = &local_emaRing[0]; }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_CVI_Lookback( int optInTimePeriod, int optInROCPeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 10;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   if( (int)optInROCPeriod == TA_INTEGER_DEFAULT )
      optInROCPeriod = 10;
   else if( (int)optInROCPeriod < 1 || (int)optInROCPeriod > 100000 )
      return -1;
   return TA_EMA_Lookback(optInTimePeriod) + TA_ROCP_Lookback(optInROCPeriod);
}
