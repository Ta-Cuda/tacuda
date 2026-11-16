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

TA_LIB_API TA_RetCode TA_ROGERSSATCHELL( int    startIdx,
                                         int    endIdx,
                                         const double inOpen[],
                                         const double inHigh[],
                                         const double inLow[],
                                         const double inClose[],
                                         int optInTimePeriod,
                                         double optInAnnualization,
                                         int          *outBegIdx,
                                         int          *outNBElement,
                                         double        outReal[] )
{
   double o;
   double h;
   double l;
   double c;
   double p1;
   double p2;
   double term;
   double periodTotal;
   double windowTotal;
   double windowMagnitude;
   double peakTotal;
   double sqrtA;
   int i;
   int j;
   int outIdx;
   int nbInitialElementNeeded;
   int barsSinceRebuild;
   double local_termRing[32];
   double *termRing = &local_termRing[0];
   int termRing_Idx;
   int maxIdx_termRing;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 10;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( optInAnnualization == TA_REAL_DEFAULT )
      optInAnnualization = 252;
   else if( !(optInAnnualization >= 0e0 && optInAnnualization <= TA_REAL_MAX) )
      return TA_BAD_PARAM;
   if( !inOpen )
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

   /* Each bar's term costs four logarithms, so it is computed once and kept
    * until it leaves the window. That also makes outReal safe to alias any
    * input: a bar is never read again once it has been consumed.
    */
   nbInitialElementNeeded = optInTimePeriod - 1;
   if( startIdx < nbInitialElementNeeded )
   {
      startIdx = nbInitialElementNeeded;
   }
   if( startIdx > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   if( optInTimePeriod < 1 ) return TA_INTERNAL_ERROR(483);
   if( (int)optInTimePeriod > (int)(sizeof(local_termRing)/sizeof(double)) )
   {
      termRing = TA_Malloc( sizeof(double)*optInTimePeriod );
      if( !termRing )
      {
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      termRing = &local_termRing[0];
   }
   maxIdx_termRing = (optInTimePeriod-1);
   termRing_Idx = 0;
   /* Rogers and Satchell, The Annals of Applied Probability 1(4):504-512 (1991),
    * eq. (2): with the log price measured from the bar's open, S1 = ln(H/O),
    * I1 = ln(L/O) and X1 = ln(C/O), one bar's estimate of its variance is
    * S1(S1 - X1) + I1(I1 - X1) = ln(H/C)ln(H/O) + ln(L/C)ln(L/O), unbiased
    * whatever the drift. The window mean, the root and the annual scale are
    * convention, not the paper's.
    *
    * Keep sqrt(A) a separate factor applied last: A = 1.0 is then an exact
    * identity and the annualised output is exactly sqrt(A) times the per-bar
    * one. Folding A under the root moves both by an ulp.
    */
   sqrtA = sqrt(optInAnnualization);
   periodTotal = 0.0;
   for( j = startIdx - nbInitialElementNeeded; j < startIdx; j += 1 )
   {
      /* Keep the two products separate statements. As one expression the
       * first product is fused into the add, which changes the values and
       * breaks bit equality with the same estimator composed from LN, DIV,
       * MULT, ADD and SUM.
       *
       * A bar with any price at or below zero contributes a 0.0 term and
       * still counts toward the window. The test is exact, not a band, so a
       * small-unit quote is not zeroed.
       */
      o = inOpen[j];
      h = inHigh[j];
      l = inLow[j];
      c = inClose[j];
      if( o > 0.0 && h > 0.0 && l > 0.0 && c > 0.0 )
      {
         p1 = log(h / c) * log(h / o);
         p2 = log(l / c) * log(l / o);
         term = p1 + p2;
      } else 
      {
         term = 0.0;
      }
      termRing[termRing_Idx] = term;
      periodTotal += term;
      termRing_Idx++;
      if( termRing_Idx > maxIdx_termRing ) termRing_Idx = 0;
   }
   i = startIdx;
   outIdx = 0;
   barsSinceRebuild = 32 * optInTimePeriod;
   peakTotal = fabs(periodTotal);
   do
   {
      o = inOpen[i];
      h = inHigh[i];
      l = inLow[i];
      c = inClose[i];
      if( o > 0.0 && h > 0.0 && l > 0.0 && c > 0.0 )
      {
         p1 = log(h / c) * log(h / o);
         p2 = log(l / c) * log(l / o);
         term = p1 + p2;
      } else 
      {
         term = 0.0;
      }
      /* Add, publish, subtract: the slot written here was taken out of the
       * sum at the end of the previous bar, and the advance lands on the
       * oldest term, the one that leaves next.
       */
      termRing[termRing_Idx] = term;
      periodTotal += term;
      windowTotal = periodTotal;
      windowMagnitude = fabs(windowTotal);
      peakTotal = (windowMagnitude > peakTotal) ? windowMagnitude : peakTotal;
      termRing_Idx++;
      if( termRing_Idx > maxIdx_termRing ) termRing_Idx = 0;
      periodTotal -= termRing[termRing_Idx];
      /* A running sum carries rounding at the scale of the largest window it
       * has held, so it is rebuilt as a fresh sum once it falls below 1e-6 of
       * that peak, and at least every 32 windows. Compare against the peak,
       * not the current sum: after a quiet stretch arrives the current sum
       * can be nothing but that rounding, of either sign, where a fresh sum
       * of an all-flat window is exactly 0.0.
       *
       * Compare magnitudes. A window holding a bar whose high or low sits
       * inside its open and close can sum below zero, and a signed test
       * would then rebuild on every bar for as long as it does.
       *
       * Sum oldest first, so the rebuilt value is the one a fresh pass over
       * the bars gives.
       */
      barsSinceRebuild -= 1;
      if( windowMagnitude < 0.000001 * peakTotal || barsSinceRebuild <= 0 )
      {
         barsSinceRebuild = 32 * optInTimePeriod;
         windowTotal = 0.0;
         for( j = termRing_Idx; j < optInTimePeriod; j += 1 )
         {
            windowTotal += termRing[j];
         }
         for( j = 0; j < termRing_Idx; j += 1 )
         {
            windowTotal += termRing[j];
         }
         peakTotal = fabs(windowTotal);
         periodTotal = windowTotal;
         periodTotal -= termRing[termRing_Idx];
      }
      /* Divide, then root, then scale. A sum at or below zero answers 0.0
       * instead of reaching the root: only a bar whose high or low sits
       * strictly inside its open and close can make a fresh sum negative,
       * and the estimator is not defined on such a bar.
       */
      if( windowTotal > 0.0 )
      {
         outReal[outIdx] = sqrtA * sqrt(windowTotal / (double)optInTimePeriod);
      } else 
      {
         outReal[outIdx] = 0.0;
      }
      outIdx = outIdx + 1;
      i += 1;
   } while( i <= endIdx );
   if( termRing != &local_termRing[0] ) { TA_Free( termRing ); termRing = &local_termRing[0]; }
   *outNBElement= outIdx;
   *outBegIdx= startIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_ROGERSSATCHELL_Lookback( int optInTimePeriod, double optInAnnualization )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 10;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   if( optInAnnualization == TA_REAL_DEFAULT )
      optInAnnualization = 252;
   else if( !(optInAnnualization >= 0e0 && optInAnnualization <= TA_REAL_MAX) )
      return -1;
   return optInTimePeriod - 1;
}
