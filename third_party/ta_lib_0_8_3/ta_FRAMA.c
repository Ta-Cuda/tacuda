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
TA_LIB_API TA_RetCode TA_FRAMA( int    startIdx,
                                int    endIdx,
                                const double inHigh[],
                                const double inLow[],
                                int optInTimePeriod,
                                int          *outBegIdx,
                                int          *outNBElement,
                                double        outReal[] )
{
   double expScale;
   double local_slot_sufHigh[32];
   double *slot_sufHigh = &local_slot_sufHigh[0];
   double local_slot_sufLow[32];
   double *slot_sufLow = &local_slot_sufLow[0];
   double local_slot_oldHigh[32];
   double *slot_oldHigh = &local_slot_oldHigh[0];
   double local_slot_oldLow[32];
   double *slot_oldLow = &local_slot_oldLow[0];
   int slot_Idx;
   int maxIdx_slot;
   double tmpHigh;
   double tmpLow;
   double preHigh;
   double preLow;
   double hi1;
   double lo1;
   double hi2;
   double lo2;
   double r1;
   double r2;
   double r;
   double price;
   double alpha;
   double prevFRAMA;
   int i;
   int today;
   int outIdx;
   int lookbackTotal;
   int half;
   int lastSlot;
   int seedIdx;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 16;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   expScale = -4.6 / 0.6931471805599453;
   /* -4.6/ln(2): alpha = exp(-4.6*(D-1)) with D-1 = log2((R1+R2)/R). */
   /* Id, Type, Static Size */
   if( optInTimePeriod % 2 != 0 )
   {
      return TA_BAD_PARAM;
   }
   *outBegIdx= 0;
   *outNBElement= 0;
   lookbackTotal = TA_FRAMA_Lookback(optInTimePeriod);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   half = optInTimePeriod / 2;
   lastSlot = half - 1;
   /* van Herk / Gil-Werman over blocks of `half` bars. Slot j holds the current
    * block's bar j (or, past the running position, the previous block's suffix
    * extremes), and in oldHigh/oldLow the newer half's extremes from `half` bars
    * ago, which are the older half's extremes now: R2[t] = R1[t-half].
    *
    * Every slot store must stay below the output store: that is what keeps the
    * stream's Peek read-only and O(1). The slots hold copies, so outReal may
    * alias an input.
    */
   if( half < 1 ) return TA_INTERNAL_ERROR(438);
   if( (int)half > (int)(sizeof(local_slot_sufHigh)/sizeof(double)) )
   {
      slot_sufHigh = TA_Malloc( sizeof(double)*half );
      if( !slot_sufHigh )
      {
         return TA_ALLOC_ERR;
      }
      slot_sufLow = TA_Malloc( sizeof(double)*half );
      if( !slot_sufLow )
      {
         TA_Free( slot_sufHigh );
         return TA_ALLOC_ERR;
      }
      slot_oldHigh = TA_Malloc( sizeof(double)*half );
      if( !slot_oldHigh )
      {
         TA_Free( slot_sufHigh );
         TA_Free( slot_sufLow );
         return TA_ALLOC_ERR;
      }
      slot_oldLow = TA_Malloc( sizeof(double)*half );
      if( !slot_oldLow )
      {
         TA_Free( slot_sufHigh );
         TA_Free( slot_sufLow );
         TA_Free( slot_oldHigh );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      slot_sufHigh = &local_slot_sufHigh[0];
      slot_sufLow = &local_slot_sufLow[0];
      slot_oldHigh = &local_slot_oldHigh[0];
      slot_oldLow = &local_slot_oldLow[0];
   }
   maxIdx_slot = (half-1);
   slot_Idx = 0;
   today = startIdx - lookbackTotal + 1;
   seedIdx = startIdx - (lookbackTotal - optInTimePeriod) - 1;
   /* The first block's suffix reads must see a bar inside the window. */
   i = 0;
   while( i < half )
   {
      slot_sufHigh[i] = inHigh[today];
      slot_sufLow[i] = inLow[today];
      slot_oldHigh[i] = inHigh[today];
      slot_oldLow[i] = inLow[today];
      i += 1;
   }
   preHigh = 0.0;
   preLow = 0.0;
   /* Through seedIdx only the blocks are built: the seed discards every value
    * before it, so no alpha is computed there.
    */
   while( today <= seedIdx )
   {
      tmpHigh = inHigh[today];
      tmpLow = inLow[today];
      if( slot_Idx == 0 )
      {
         preHigh = tmpHigh;
         preLow = tmpLow;
      } else 
      {
         if( tmpHigh > preHigh )
         {
            preHigh = tmpHigh;
         }
         if( tmpLow < preLow )
         {
            preLow = tmpLow;
         }
      }
      if( slot_Idx == lastSlot )
      {
         hi1 = preHigh;
         lo1 = preLow;
      } else 
      {
         hi1 = slot_sufHigh[slot_Idx + 1];
         if( preHigh > hi1 )
         {
            hi1 = preHigh;
         }
         lo1 = slot_sufLow[slot_Idx + 1];
         if( preLow < lo1 )
         {
            lo1 = preLow;
         }
      }
      slot_oldHigh[slot_Idx] = hi1;
      slot_oldLow[slot_Idx] = lo1;
      slot_sufHigh[slot_Idx] = tmpHigh;
      slot_sufLow[slot_Idx] = tmpLow;
      if( slot_Idx == lastSlot )
      {
         i = lastSlot;
         while( i > 0 )
         {
            i -= 1;
            if( slot_sufHigh[i + 1] > slot_sufHigh[i] )
            {
               slot_sufHigh[i] = slot_sufHigh[i + 1];
            }
            if( slot_sufLow[i + 1] < slot_sufLow[i] )
            {
               slot_sufLow[i] = slot_sufLow[i + 1];
            }
         }
      }
      slot_Idx++;
      if( slot_Idx > maxIdx_slot ) slot_Idx = 0;
      today += 1;
   }
   prevFRAMA = (inHigh[seedIdx] + inLow[seedIdx]) / 2.0;
   outIdx = 0;
   while( today <= endIdx )
   {
      tmpHigh = inHigh[today];
      tmpLow = inLow[today];
      price = (tmpHigh + tmpLow) / 2.0;
      if( slot_Idx == 0 )
      {
         preHigh = tmpHigh;
         preLow = tmpLow;
      } else 
      {
         if( tmpHigh > preHigh )
         {
            preHigh = tmpHigh;
         }
         if( tmpLow < preLow )
         {
            preLow = tmpLow;
         }
      }
      if( slot_Idx == lastSlot )
      {
         hi1 = preHigh;
         lo1 = preLow;
      } else 
      {
         hi1 = slot_sufHigh[slot_Idx + 1];
         if( preHigh > hi1 )
         {
            hi1 = preHigh;
         }
         lo1 = slot_sufLow[slot_Idx + 1];
         if( preLow < lo1 )
         {
            lo1 = preLow;
         }
      }
      hi2 = slot_oldHigh[slot_Idx];
      lo2 = slot_oldLow[slot_Idx];
      r1 = hi1 - lo1;
      r2 = hi2 - lo2;
      r = ((hi1 > hi2) ? hi1 : hi2) - ((lo1 < lo2) ? lo1 : lo2);
      /* R >= max(R1,R2) makes R1+R2 <= R exactly the alpha >= 1 clamp, so a
       * clamped or flat bar returns the price with no transcendental. Keep the
       * recursion as alpha*P + (1-alpha)*prev: a computed alpha of 1.0 then
       * returns the price exactly too.
       */
      if( r1 > 0.0 && r2 > 0.0 && r1 + r2 > r )
      {
         alpha = exp(expScale * log((r1 + r2) / r));
         prevFRAMA = fma(1.0 - alpha, prevFRAMA, alpha * price);
      } else 
      {
         prevFRAMA = price;
      }
      if( today >= startIdx )
      {
         outReal[outIdx++] = prevFRAMA;
      }
      slot_oldHigh[slot_Idx] = hi1;
      slot_oldLow[slot_Idx] = lo1;
      slot_sufHigh[slot_Idx] = tmpHigh;
      slot_sufLow[slot_Idx] = tmpLow;
      if( slot_Idx == lastSlot )
      {
         i = lastSlot;
         while( i > 0 )
         {
            i -= 1;
            if( slot_sufHigh[i + 1] > slot_sufHigh[i] )
            {
               slot_sufHigh[i] = slot_sufHigh[i + 1];
            }
            if( slot_sufLow[i + 1] < slot_sufLow[i] )
            {
               slot_sufLow[i] = slot_sufLow[i + 1];
            }
         }
      }
      slot_Idx++;
      if( slot_Idx > maxIdx_slot ) slot_Idx = 0;
      today += 1;
   }
   if( slot_sufHigh != &local_slot_sufHigh[0] ) { TA_Free( slot_sufHigh ); slot_sufHigh = &local_slot_sufHigh[0]; }
   if( slot_sufLow != &local_slot_sufLow[0] ) { TA_Free( slot_sufLow ); slot_sufLow = &local_slot_sufLow[0]; }
   if( slot_oldHigh != &local_slot_oldHigh[0] ) { TA_Free( slot_oldHigh ); slot_oldHigh = &local_slot_oldHigh[0]; }
   if( slot_oldLow != &local_slot_oldLow[0] ) { TA_Free( slot_oldLow ); slot_oldLow = &local_slot_oldLow[0]; }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_NOINLINE TA_LIB_API int TA_FRAMA_Lookback( int optInTimePeriod )
{
   int root;
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 16;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   root = (int)sqrt((double)optInTimePeriod);
   /* The range check cannot demand an even period; without this the lookback
    * answers a usable number for a call that cannot run.
    */
   if( optInTimePeriod % 2 != 0 )
   {
      return -1;
   }
   return optInTimePeriod + TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_FRAMA,Frama,((9 * (4 + 4) * (root + 2) / 2 < 99 * 10) ? 9 * (4 + 4) * (root + 2) / 2 : 99 * 10),((9 * (8 + 4) * (root + 2) / 2 < 99 * 19) ? 9 * (8 + 4) * (root + 2) / 2 : 99 * 19));
}
