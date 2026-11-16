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
TA_LIB_API TA_RetCode TA_KST( int    startIdx,
                              int    endIdx,
                              const double inReal[],
                              int optInROC1Period,
                              int optInROC2Period,
                              int optInROC3Period,
                              int optInROC4Period,
                              int optInSMA1Period,
                              int optInSMA2Period,
                              int optInSMA3Period,
                              int optInSMA4Period,
                              int optInSignalPeriod,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outKST[],
                              double        outKSTSignal[] )
{
   int outIdx;
   int inIdx;
   int lookbackTotal;
   int sigStart;
   int den1;
   int den2;
   int den3;
   int den4;
   double prior;
   double roc;
   double kst;
   double sig;
   double total1;
   double total2;
   double total3;
   double total4;
   double sigTotal;
   double rcma1;
   double rcma2;
   double rcma3;
   double rcma4;
   double local_ring1[30];
   double *ring1 = &local_ring1[0];
   int ring1_Idx;
   int maxIdx_ring1;
   double local_ring2[30];
   double *ring2 = &local_ring2[0];
   int ring2_Idx;
   int maxIdx_ring2;
   double local_ring3[30];
   double *ring3 = &local_ring3[0];
   int ring3_Idx;
   int maxIdx_ring3;
   double local_ring4[30];
   double *ring4 = &local_ring4[0];
   int ring4_Idx;
   int maxIdx_ring4;
   double local_sigRing[30];
   double *sigRing = &local_sigRing[0];
   int sigRing_Idx;
   int maxIdx_sigRing;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInROC1Period == TA_INTEGER_DEFAULT )
      optInROC1Period = 10;
   else if( (int)optInROC1Period < 1 || (int)optInROC1Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInROC2Period == TA_INTEGER_DEFAULT )
      optInROC2Period = 15;
   else if( (int)optInROC2Period < 1 || (int)optInROC2Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInROC3Period == TA_INTEGER_DEFAULT )
      optInROC3Period = 20;
   else if( (int)optInROC3Period < 1 || (int)optInROC3Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInROC4Period == TA_INTEGER_DEFAULT )
      optInROC4Period = 30;
   else if( (int)optInROC4Period < 1 || (int)optInROC4Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSMA1Period == TA_INTEGER_DEFAULT )
      optInSMA1Period = 10;
   else if( (int)optInSMA1Period < 1 || (int)optInSMA1Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSMA2Period == TA_INTEGER_DEFAULT )
      optInSMA2Period = 10;
   else if( (int)optInSMA2Period < 1 || (int)optInSMA2Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSMA3Period == TA_INTEGER_DEFAULT )
      optInSMA3Period = 10;
   else if( (int)optInSMA3Period < 1 || (int)optInSMA3Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSMA4Period == TA_INTEGER_DEFAULT )
      optInSMA4Period = 15;
   else if( (int)optInSMA4Period < 1 || (int)optInSMA4Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSignalPeriod == TA_INTEGER_DEFAULT )
      optInSignalPeriod = 9;
   else if( (int)optInSignalPeriod < 1 || (int)optInSignalPeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outKST )
      return TA_BAD_PARAM;
   if( !outKSTSignal )
      return TA_BAD_PARAM;
   if( outKST == outKSTSignal )
      return TA_BAD_PARAM;

   /* Bit-exact with TA_ROC per leg into TA_SMA, each SMA called at the start
    * its consumer needs: keep every running sum's add/subtract order and the
    * left-to-right weighted sum, or the composite differential stops being
    * exact. Each ring holds its stage's last SMA-period values, so the only
    * inputs read are the current bar and each leg's lagged denominator,
    * both at or after the slot the outputs write: outKST or outKSTSignal
    * may alias inReal.
    */
   lookbackTotal = TA_KST_Lookback(optInROC1Period,optInROC2Period,optInROC3Period,optInROC4Period,optInSMA1Period,optInSMA2Period,optInSMA3Period,optInSMA4Period,optInSignalPeriod);
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
   if( optInSMA1Period < 1 ) return TA_INTERNAL_ERROR(442);
   if( (int)optInSMA1Period > (int)(sizeof(local_ring1)/sizeof(double)) )
   {
      ring1 = TA_Malloc( sizeof(double)*optInSMA1Period );
      if( !ring1 )
      {
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      ring1 = &local_ring1[0];
   }
   maxIdx_ring1 = (optInSMA1Period-1);
   ring1_Idx = 0;
   if( optInSMA2Period < 1 ) return TA_INTERNAL_ERROR(443);
   if( (int)optInSMA2Period > (int)(sizeof(local_ring2)/sizeof(double)) )
   {
      ring2 = TA_Malloc( sizeof(double)*optInSMA2Period );
      if( !ring2 )
      {
         if( ring1 != &local_ring1[0] ) TA_Free( ring1 );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      ring2 = &local_ring2[0];
   }
   maxIdx_ring2 = (optInSMA2Period-1);
   ring2_Idx = 0;
   if( optInSMA3Period < 1 ) return TA_INTERNAL_ERROR(444);
   if( (int)optInSMA3Period > (int)(sizeof(local_ring3)/sizeof(double)) )
   {
      ring3 = TA_Malloc( sizeof(double)*optInSMA3Period );
      if( !ring3 )
      {
         if( ring1 != &local_ring1[0] ) TA_Free( ring1 );
         if( ring2 != &local_ring2[0] ) TA_Free( ring2 );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      ring3 = &local_ring3[0];
   }
   maxIdx_ring3 = (optInSMA3Period-1);
   ring3_Idx = 0;
   if( optInSMA4Period < 1 ) return TA_INTERNAL_ERROR(445);
   if( (int)optInSMA4Period > (int)(sizeof(local_ring4)/sizeof(double)) )
   {
      ring4 = TA_Malloc( sizeof(double)*optInSMA4Period );
      if( !ring4 )
      {
         if( ring1 != &local_ring1[0] ) TA_Free( ring1 );
         if( ring2 != &local_ring2[0] ) TA_Free( ring2 );
         if( ring3 != &local_ring3[0] ) TA_Free( ring3 );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      ring4 = &local_ring4[0];
   }
   maxIdx_ring4 = (optInSMA4Period-1);
   ring4_Idx = 0;
   if( optInSignalPeriod < 1 ) return TA_INTERNAL_ERROR(446);
   if( (int)optInSignalPeriod > (int)(sizeof(local_sigRing)/sizeof(double)) )
   {
      sigRing = TA_Malloc( sizeof(double)*optInSignalPeriod );
      if( !sigRing )
      {
         if( ring1 != &local_ring1[0] ) TA_Free( ring1 );
         if( ring2 != &local_ring2[0] ) TA_Free( ring2 );
         if( ring3 != &local_ring3[0] ) TA_Free( ring3 );
         if( ring4 != &local_ring4[0] ) TA_Free( ring4 );
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      sigRing = &local_sigRing[0];
   }
   maxIdx_sigRing = (optInSignalPeriod-1);
   sigRing_Idx = 0;
   sigStart = startIdx - (optInSignalPeriod - 1);
   sigTotal = 0.0;
   total4 = sigTotal;
   total3 = total4;
   total2 = total3;
   total1 = total2;
   inIdx = sigStart - (optInSMA1Period - 1);
   den1 = inIdx - optInROC1Period;
   while( inIdx < sigStart )
   {
      prior = inReal[den1];
      den1 += 1;
      roc = (prior != 0.0) ? (inReal[inIdx] / prior - 1.0) * 100.0 : 0.0;
      total1 += roc;
      ring1[ring1_Idx] = roc;
      ring1_Idx++;
      if( ring1_Idx > maxIdx_ring1 ) ring1_Idx = 0;
      inIdx += 1;
   }
   inIdx = sigStart - (optInSMA2Period - 1);
   den2 = inIdx - optInROC2Period;
   while( inIdx < sigStart )
   {
      prior = inReal[den2];
      den2 += 1;
      roc = (prior != 0.0) ? (inReal[inIdx] / prior - 1.0) * 100.0 : 0.0;
      total2 += roc;
      ring2[ring2_Idx] = roc;
      ring2_Idx++;
      if( ring2_Idx > maxIdx_ring2 ) ring2_Idx = 0;
      inIdx += 1;
   }
   inIdx = sigStart - (optInSMA3Period - 1);
   den3 = inIdx - optInROC3Period;
   while( inIdx < sigStart )
   {
      prior = inReal[den3];
      den3 += 1;
      roc = (prior != 0.0) ? (inReal[inIdx] / prior - 1.0) * 100.0 : 0.0;
      total3 += roc;
      ring3[ring3_Idx] = roc;
      ring3_Idx++;
      if( ring3_Idx > maxIdx_ring3 ) ring3_Idx = 0;
      inIdx += 1;
   }
   inIdx = sigStart - (optInSMA4Period - 1);
   den4 = inIdx - optInROC4Period;
   while( inIdx < sigStart )
   {
      prior = inReal[den4];
      den4 += 1;
      roc = (prior != 0.0) ? (inReal[inIdx] / prior - 1.0) * 100.0 : 0.0;
      total4 += roc;
      ring4[ring4_Idx] = roc;
      ring4_Idx++;
      if( ring4_Idx > maxIdx_ring4 ) ring4_Idx = 0;
      inIdx += 1;
   }
   inIdx = sigStart;
   while( inIdx < startIdx )
   {
      prior = inReal[den1];
      den1 += 1;
      roc = (prior != 0.0) ? (inReal[inIdx] / prior - 1.0) * 100.0 : 0.0;
      total1 += roc;
      rcma1 = total1 / (double)optInSMA1Period;
      ring1[ring1_Idx] = roc;
      ring1_Idx++;
      if( ring1_Idx > maxIdx_ring1 ) ring1_Idx = 0;
      total1 -= ring1[ring1_Idx];
      prior = inReal[den2];
      den2 += 1;
      roc = (prior != 0.0) ? (inReal[inIdx] / prior - 1.0) * 100.0 : 0.0;
      total2 += roc;
      rcma2 = total2 / (double)optInSMA2Period;
      ring2[ring2_Idx] = roc;
      ring2_Idx++;
      if( ring2_Idx > maxIdx_ring2 ) ring2_Idx = 0;
      total2 -= ring2[ring2_Idx];
      prior = inReal[den3];
      den3 += 1;
      roc = (prior != 0.0) ? (inReal[inIdx] / prior - 1.0) * 100.0 : 0.0;
      total3 += roc;
      rcma3 = total3 / (double)optInSMA3Period;
      ring3[ring3_Idx] = roc;
      ring3_Idx++;
      if( ring3_Idx > maxIdx_ring3 ) ring3_Idx = 0;
      total3 -= ring3[ring3_Idx];
      prior = inReal[den4];
      den4 += 1;
      roc = (prior != 0.0) ? (inReal[inIdx] / prior - 1.0) * 100.0 : 0.0;
      total4 += roc;
      rcma4 = total4 / (double)optInSMA4Period;
      ring4[ring4_Idx] = roc;
      ring4_Idx++;
      if( ring4_Idx > maxIdx_ring4 ) ring4_Idx = 0;
      total4 -= ring4[ring4_Idx];
      kst = fma(4.0, rcma4, fma(3.0, rcma3, fma(2.0, rcma2, rcma1)));
      sigTotal += kst;
      sigRing[sigRing_Idx] = kst;
      sigRing_Idx++;
      if( sigRing_Idx > maxIdx_sigRing ) sigRing_Idx = 0;
      inIdx += 1;
   }
   outIdx = 0;
   while( inIdx <= endIdx )
   {
      prior = inReal[den1];
      den1 += 1;
      roc = (prior != 0.0) ? (inReal[inIdx] / prior - 1.0) * 100.0 : 0.0;
      total1 += roc;
      rcma1 = total1 / (double)optInSMA1Period;
      ring1[ring1_Idx] = roc;
      ring1_Idx++;
      if( ring1_Idx > maxIdx_ring1 ) ring1_Idx = 0;
      total1 -= ring1[ring1_Idx];
      prior = inReal[den2];
      den2 += 1;
      roc = (prior != 0.0) ? (inReal[inIdx] / prior - 1.0) * 100.0 : 0.0;
      total2 += roc;
      rcma2 = total2 / (double)optInSMA2Period;
      ring2[ring2_Idx] = roc;
      ring2_Idx++;
      if( ring2_Idx > maxIdx_ring2 ) ring2_Idx = 0;
      total2 -= ring2[ring2_Idx];
      prior = inReal[den3];
      den3 += 1;
      roc = (prior != 0.0) ? (inReal[inIdx] / prior - 1.0) * 100.0 : 0.0;
      total3 += roc;
      rcma3 = total3 / (double)optInSMA3Period;
      ring3[ring3_Idx] = roc;
      ring3_Idx++;
      if( ring3_Idx > maxIdx_ring3 ) ring3_Idx = 0;
      total3 -= ring3[ring3_Idx];
      prior = inReal[den4];
      den4 += 1;
      roc = (prior != 0.0) ? (inReal[inIdx] / prior - 1.0) * 100.0 : 0.0;
      total4 += roc;
      rcma4 = total4 / (double)optInSMA4Period;
      ring4[ring4_Idx] = roc;
      ring4_Idx++;
      if( ring4_Idx > maxIdx_ring4 ) ring4_Idx = 0;
      total4 -= ring4[ring4_Idx];
      kst = fma(4.0, rcma4, fma(3.0, rcma3, fma(2.0, rcma2, rcma1)));
      sigTotal += kst;
      sig = sigTotal / (double)optInSignalPeriod;
      sigRing[sigRing_Idx] = kst;
      sigRing_Idx++;
      if( sigRing_Idx > maxIdx_sigRing ) sigRing_Idx = 0;
      sigTotal -= sigRing[sigRing_Idx];
      outKST[outIdx] = kst;
      outKSTSignal[outIdx] = sig;
      outIdx += 1;
      inIdx += 1;
   }
   if( ring1 != &local_ring1[0] ) { TA_Free( ring1 ); ring1 = &local_ring1[0]; }
   if( ring2 != &local_ring2[0] ) { TA_Free( ring2 ); ring2 = &local_ring2[0]; }
   if( ring3 != &local_ring3[0] ) { TA_Free( ring3 ); ring3 = &local_ring3[0]; }
   if( ring4 != &local_ring4[0] ) { TA_Free( ring4 ); ring4 = &local_ring4[0]; }
   if( sigRing != &local_sigRing[0] ) { TA_Free( sigRing ); sigRing = &local_sigRing[0]; }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_KST_Lookback( int optInROC1Period, int optInROC2Period, int optInROC3Period, int optInROC4Period, int optInSMA1Period, int optInSMA2Period, int optInSMA3Period, int optInSMA4Period, int optInSignalPeriod )
{
   int legMax;
   int leg;
   if( (int)optInROC1Period == TA_INTEGER_DEFAULT )
      optInROC1Period = 10;
   else if( (int)optInROC1Period < 1 || (int)optInROC1Period > 100000 )
      return -1;
   if( (int)optInROC2Period == TA_INTEGER_DEFAULT )
      optInROC2Period = 15;
   else if( (int)optInROC2Period < 1 || (int)optInROC2Period > 100000 )
      return -1;
   if( (int)optInROC3Period == TA_INTEGER_DEFAULT )
      optInROC3Period = 20;
   else if( (int)optInROC3Period < 1 || (int)optInROC3Period > 100000 )
      return -1;
   if( (int)optInROC4Period == TA_INTEGER_DEFAULT )
      optInROC4Period = 30;
   else if( (int)optInROC4Period < 1 || (int)optInROC4Period > 100000 )
      return -1;
   if( (int)optInSMA1Period == TA_INTEGER_DEFAULT )
      optInSMA1Period = 10;
   else if( (int)optInSMA1Period < 1 || (int)optInSMA1Period > 100000 )
      return -1;
   if( (int)optInSMA2Period == TA_INTEGER_DEFAULT )
      optInSMA2Period = 10;
   else if( (int)optInSMA2Period < 1 || (int)optInSMA2Period > 100000 )
      return -1;
   if( (int)optInSMA3Period == TA_INTEGER_DEFAULT )
      optInSMA3Period = 10;
   else if( (int)optInSMA3Period < 1 || (int)optInSMA3Period > 100000 )
      return -1;
   if( (int)optInSMA4Period == TA_INTEGER_DEFAULT )
      optInSMA4Period = 15;
   else if( (int)optInSMA4Period < 1 || (int)optInSMA4Period > 100000 )
      return -1;
   if( (int)optInSignalPeriod == TA_INTEGER_DEFAULT )
      optInSignalPeriod = 9;
   else if( (int)optInSignalPeriod < 1 || (int)optInSignalPeriod > 100000 )
      return -1;
   legMax = optInROC1Period + optInSMA1Period - 1;
   leg = optInROC2Period + optInSMA2Period - 1;
   if( leg > legMax )
   {
      legMax = leg;
   }
   leg = optInROC3Period + optInSMA3Period - 1;
   if( leg > legMax )
   {
      legMax = leg;
   }
   leg = optInROC4Period + optInSMA4Period - 1;
   if( leg > legMax )
   {
      legMax = leg;
   }
   return legMax + optInSignalPeriod - 1;
}
