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
TA_LIB_API TA_RetCode TA_MASSI( int    startIdx,
                                int    endIdx,
                                const double inHigh[],
                                const double inLow[],
                                int optInFastPeriod,
                                int optInSlowPeriod,
                                int          *outBegIdx,
                                int          *outNBElement,
                                double        outReal[] )
{
   double optInK_1;
   double emaBeta;
   double hl;
   double ema1;
   double ema2;
   double sum1;
   double sum2;
   double ratio;
   double total;
   double tempReal;
   int lookbackTotal;
   int lookbackEma;
   int lookbackEma2;
   int today;
   int outIdx;
   int nBar;
   int n2;
   double local_ratioRing[32];
   double *ratioRing = &local_ratioRing[0];
   int ratioRing_Idx;
   int maxIdx_ratioRing;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInFastPeriod == TA_INTEGER_DEFAULT )
      optInFastPeriod = 9;
   else if( (int)optInFastPeriod < 2 || (int)optInFastPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSlowPeriod == TA_INTEGER_DEFAULT )
      optInSlowPeriod = 25;
   else if( (int)optInSlowPeriod < 2 || (int)optInSlowPeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   lookbackEma = TA_EMA_Lookback(optInFastPeriod);
   lookbackEma2 = lookbackEma * 2;
   lookbackTotal = lookbackEma2 + (optInSlowPeriod - 1);
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
   if( optInSlowPeriod < 1 ) return TA_INTERNAL_ERROR(418);
   if( (int)optInSlowPeriod > (int)(sizeof(local_ratioRing)/sizeof(double)) )
   {
      ratioRing = TA_Malloc( sizeof(double)*optInSlowPeriod );
      if( !ratioRing )
      {
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      ratioRing = &local_ratioRing[0];
   }
   maxIdx_ratioRing = (optInSlowPeriod-1);
   ratioRing_Idx = 0;
   *outBegIdx= startIdx;
   /* Dorsey's pipeline in one pass: EMA of the high-low range, EMA of that,
    * their ratio summed over a trailing window.
    *
    * Each stage seeds the way ema.c does -- a simple average of that stage's
    * first optInFastPeriod inputs -- and its boundary below is the callee
    * LOOKBACK, never (optInFastPeriod - 1). The two coincide exactly at
    * unstable period 0, which is where every cross-language gate runs, so
    * confusing them is invisible until TA_SetUnstablePeriod(TA_FUNC_UNST_EMA)
    * is warmed. The seed sums accumulate from 0.0 in production order; do not
    * reorder or fuse them (0.0+x is not x for x=-0.0).
    */
   emaBeta = (double)(optInFastPeriod - 1) / (double)(optInFastPeriod + 1);
   optInK_1 = 1.0 - emaBeta;
   if( emaBeta < 0.5 )
   {
      emaBeta = 1.0 - optInK_1;
   }
   ema1 = 0.0;
   ema2 = 0.0;
   sum1 = 0.0;
   sum2 = 0.0;
   total = 0.0;
   tempReal = 0.0;
   today = startIdx - lookbackTotal;
   nBar = 0;
   /* Runs through startIdx inclusive: that last pass fills the summation
    * window and so produces the first output.
    */
   while( today <= startIdx )
   {
      hl = inHigh[today] - inLow[today];
      if( nBar < optInFastPeriod )
      {
         sum1 = sum1 + hl;
         if( nBar == optInFastPeriod - 1 )
         {
            ema1 = sum1 / optInFastPeriod;
         }
      } else 
      {
         ema1 = fma(emaBeta, ema1, optInK_1 * hl);
      }
      /* The stage counter is compared BEFORE it is subtracted, never after.
       * `n2 = nBar - lookbackEma; if( n2 >= 0 )` is correct in C and broken
       * everywhere else: the Rust backend renders these as usize, so the
       * subtraction underflows for the first lookbackEma bars.
       */
      if( nBar >= lookbackEma )
      {
         n2 = nBar - lookbackEma;
         if( n2 < optInFastPeriod )
         {
            sum2 = sum2 + ema1;
            if( n2 == optInFastPeriod - 1 )
            {
               ema2 = sum2 / optInFastPeriod;
            }
         } else 
         {
            ema2 = fma(emaBeta, ema2, optInK_1 * ema1);
         }
      }
      if( nBar >= lookbackEma2 )
      {
         /* A flat market is the ratio's continuous limit, 1.0, not the zero
          * that an oscillator centred on zero would report: MASSI's own
          * neutral is optInSlowPeriod. Test ema2 exactly and never through an
          * epsilon band -- a smoothed price range carries the quote unit, so a
          * fixed band would pin the whole index at optInSlowPeriod for any
          * instrument quoted under it (issue #253).
          */
         if( ema2 == 0.0 )
         {
            ratio = 1.0;
         } else 
         {
            ratio = ema1 / ema2;
         }
         /* TA_SUM's accumulation order -- add, publish, subtract -- reproduced
          * over a ring: the slot written here was emptied out of `total` at the
          * end of the previous bar, so nothing has to be read before the store.
          */
         ratioRing[ratioRing_Idx] = ratio;
         total = total + ratio;
         ratioRing_Idx++;
         if( ratioRing_Idx > maxIdx_ratioRing ) ratioRing_Idx = 0;
         if( nBar == lookbackTotal )
         {
            tempReal = total;
            total = total - ratioRing[ratioRing_Idx];
         }
      }
      nBar = nBar + 1;
      today = today + 1;
   }
   /* In-place safe: this store lands lookbackTotal bars behind the input
    * cursor, so no bar is written under a read still owed to it.
    */
   outReal[0] = tempReal;
   outIdx = 1;
   while( today <= endIdx )
   {
      hl = inHigh[today] - inLow[today];
      ema1 = fma(emaBeta, ema1, optInK_1 * hl);
      ema2 = fma(emaBeta, ema2, optInK_1 * ema1);
      if( ema2 == 0.0 )
      {
         ratio = 1.0;
      } else 
      {
         ratio = ema1 / ema2;
      }
      ratioRing[ratioRing_Idx] = ratio;
      total = total + ratio;
      ratioRing_Idx++;
      if( ratioRing_Idx > maxIdx_ratioRing ) ratioRing_Idx = 0;
      tempReal = total;
      total = total - ratioRing[ratioRing_Idx];
      outReal[outIdx] = tempReal;
      outIdx = outIdx + 1;
      today = today + 1;
   }
   *outNBElement= outIdx;
   if( ratioRing != &local_ratioRing[0] ) { TA_Free( ratioRing ); ratioRing = &local_ratioRing[0]; }
   return TA_SUCCESS;
}

TA_LIB_API int TA_MASSI_Lookback( int optInFastPeriod, int optInSlowPeriod )
{
   if( (int)optInFastPeriod == TA_INTEGER_DEFAULT )
      optInFastPeriod = 9;
   else if( (int)optInFastPeriod < 2 || (int)optInFastPeriod > 100000 )
      return -1;
   if( (int)optInSlowPeriod == TA_INTEGER_DEFAULT )
      optInSlowPeriod = 25;
   else if( (int)optInSlowPeriod < 2 || (int)optInSlowPeriod > 100000 )
      return -1;
   /* Two stacked EMA warm-ups over the high-low range, then the summation
    * window. The EMA term is exactly the callee's own lookback, which is what
    * makes MASSI inherit TA_FUNC_UNST_EMA -- and it shifts by 2u, not u.
    */
   return TA_EMA_Lookback(optInFastPeriod) * 2 + (optInSlowPeriod - 1);
}
