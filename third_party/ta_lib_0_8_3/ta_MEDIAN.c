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

TA_LIB_API TA_RetCode TA_MEDIAN( int    startIdx,
                                 int    endIdx,
                                 const double inReal[],
                                 int optInTimePeriod,
                                 int          *outBegIdx,
                                 int          *outNBElement,
                                 double        outReal[] )
{
   double newValue;
   double oldValue;
   double result;
   double lower;
   double upper;
   int lookbackTotal;
   int outIdx;
   int i;
   int j;
   int pos;
   int nbSorted;
   int lowerIdx;
   int upperIdx;
   int lo;
   int hi;
   int mid;
   double local_ring[30];
   double *ring = &local_ring[0];
   int ring_Idx;
   int maxIdx_ring;
   double local_sorted[30];
   double *sorted = &local_sorted[0];

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 30;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 10000 )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* The window is carried twice: "ring" by age, "sorted" by value. */
   lookbackTotal = optInTimePeriod - 1;
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
   if( optInTimePeriod < 1 ) return TA_INTERNAL_ERROR(431);
   if( (int)optInTimePeriod > (int)(sizeof(local_ring)/sizeof(double)) )
   {
      ring = TA_Malloc( sizeof(double)*optInTimePeriod );
      if( !ring )
      {
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      ring = &local_ring[0];
   }
   maxIdx_ring = (optInTimePeriod-1);
   ring_Idx = 0;
   if( optInTimePeriod < 1 ) return TA_INTERNAL_ERROR(432);
   if( (int)optInTimePeriod > (int)(sizeof(local_sorted)/sizeof(double)) )
   {
      sorted = TA_Malloc( sizeof(double)*optInTimePeriod );
      if( !sorted )
      {
         if( ring != &local_ring[0] ) TA_Free( ring );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      sorted = &local_sorted[0];
   }
   /* Never read: set so two handles opened over the same bars hold the same
    * state.
    */
   sorted[lookbackTotal] = 0.0;
   /* The two central ordinals, zero-based; the same slot at odd n. */
   lowerIdx = (optInTimePeriod - 1) / 2;
   upperIdx = optInTimePeriod / 2;
   nbSorted = 0;
   i = startIdx - lookbackTotal;
   while( i < startIdx )
   {
      newValue = inReal[i];
      if( lookbackTotal < 128 )
      {
         j = nbSorted;
         while( j > 0 && sorted[j - 1] > newValue )
         {
            sorted[j] = sorted[j - 1];
            j -= 1;
         }
      } else 
      {
         lo = 0;
         hi = nbSorted;
         while( lo < hi )
         {
            mid = (lo + hi) / 2;
            if( sorted[mid] <= newValue )
            {
               lo = mid + 1;
            } else 
            {
               hi = mid;
            }
         }
         j = nbSorted;
         while( j > lo )
         {
            sorted[j] = sorted[j - 1];
            j -= 1;
         }
      }
      sorted[j] = newValue;
      nbSorted += 1;
      ring[ring_Idx] = newValue;
      i += 1;
      ring_Idx++;
      if( ring_Idx > maxIdx_ring ) ring_Idx = 0;
   }
   /* Both scratch buffers hold copies and inReal is never read below i, so
    * inReal and outReal may be the same buffer.
    *
    * Every buffer store sits BELOW the output store: deriving the whole answer
    * read-only above it is what lets the streaming peek frame drop the state
    * update.
    *
    * Below 128 values a linear scan beats a binary search: one mispredicted
    * loop exit costs less than log2(n) unpredictable halvings.
    */
   outIdx = 0;
   do
   {
      newValue = inReal[i];
      /* pos counts the retained values <= newValue, so the full window is
       * sorted[0..pos-1], newValue, sorted[pos..].
       */
      if( lookbackTotal < 128 )
      {
         pos = 0;
         while( pos < lookbackTotal && sorted[pos] <= newValue )
         {
            pos += 1;
         }
      } else 
      {
         lo = 0;
         hi = lookbackTotal;
         while( lo < hi )
         {
            mid = (lo + hi) / 2;
            if( sorted[mid] <= newValue )
            {
               lo = mid + 1;
            } else 
            {
               hi = mid;
            }
         }
         pos = lo;
      }
      if( lowerIdx < pos )
      {
         lower = sorted[lowerIdx];
      } else if( lowerIdx == pos )
      {
         lower = newValue;
      } else 
      {
         lower = sorted[lowerIdx - 1];
      }
      if( upperIdx < pos )
      {
         upper = sorted[upperIdx];
      } else if( upperIdx == pos )
      {
         upper = newValue;
      } else 
      {
         upper = sorted[upperIdx - 1];
      }
      /* At odd n both reads are one value; averaging it would overflow above
       * DBL_MAX/2.
       */
      if( lowerIdx == upperIdx )
      {
         result = lower;
      } else 
      {
         result = (lower + upper) / 2.0;
      }
      outReal[outIdx] = result;
      outIdx += 1;
      ring[ring_Idx] = newValue;
      ring_Idx++;
      if( ring_Idx > maxIdx_ring ) ring_Idx = 0;
      oldValue = ring[ring_Idx];
      /* j is the first retained value >= oldValue. Keep every run of equal
       * values in age order (newValue goes after its equals, as above): the
       * oldest of a run is then the departing value bit for bit, which is what
       * keeps -0.0 and 0.0 apart. Inserting before the equals instead changes
       * no value but flips the sign of some zero outputs.
       */
      if( lookbackTotal < 128 )
      {
         j = 0;
         while( j < lookbackTotal && sorted[j] < oldValue )
         {
            j += 1;
         }
      } else 
      {
         lo = 0;
         hi = lookbackTotal;
         while( lo < hi )
         {
            mid = (lo + hi) / 2;
            if( sorted[mid] < oldValue )
            {
               lo = mid + 1;
            } else 
            {
               hi = mid;
            }
         }
         j = lo;
      }
      /* Evict oldValue and place newValue with one shift of the slots between
       * them.
       */
      if( j < pos )
      {
         while( j < pos - 1 )
         {
            sorted[j] = sorted[j + 1];
            j += 1;
         }
         sorted[pos - 1] = newValue;
      } else 
      {
         while( j > pos )
         {
            sorted[j] = sorted[j - 1];
            j -= 1;
         }
         sorted[pos] = newValue;
      }
      i += 1;
   } while( i <= endIdx );
   *outNBElement= outIdx;
   *outBegIdx= startIdx;
   if( ring != &local_ring[0] ) { TA_Free( ring ); ring = &local_ring[0]; }
   if( sorted != &local_sorted[0] ) { TA_Free( sorted ); sorted = &local_sorted[0]; }
   return TA_SUCCESS;
}

TA_LIB_API int TA_MEDIAN_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 30;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 10000 )
      return -1;
   return optInTimePeriod - 1;
}
