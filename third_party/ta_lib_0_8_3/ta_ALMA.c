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

TA_LIB_API TA_RetCode TA_ALMA( int    startIdx,
                               int    endIdx,
                               const double inReal[],
                               int optInTimePeriod,
                               double optInSigma,
                               double optInOffset,
                               int          *outBegIdx,
                               int          *outNBElement,
                               double        outReal[] )
{
   double sum;
   double s;
   double twoSSq;
   double d;
   double norm;
   double wj;
   double s0;
   double s1;
   double s2;
   double s3;
   int lookbackTotal;
   int outIdx;
   int i;
   int j;
   int w;
   int m;
   int b;
   double local_weights[30];
   double *weights = &local_weights[0];

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 9;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( optInSigma == TA_REAL_DEFAULT )
      optInSigma = 6;
   else if( !(optInSigma >= 1e-2 && optInSigma <= TA_REAL_MAX) )
      return TA_BAD_PARAM;
   if( optInOffset == TA_REAL_DEFAULT )
      optInOffset = 0.85;
   else if( !(optInOffset >= 0e0 && optInOffset <= 1e0) )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* Filled once and read as a plain array. */
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
   /* A dot product seeded with 0.0 would turn a -0.0 input into +0.0. */
   if( optInTimePeriod == 1 )
   {
      *outBegIdx= startIdx;
      *outNBElement= endIdx - startIdx + 1;
      i = startIdx;
      for( outIdx = 0; outIdx < (int)*outNBElement; outIdx += 1 )
      {
         outReal[outIdx] = inReal[i++];
      }
      return TA_SUCCESS;
   }
   if( optInTimePeriod < 1 ) return TA_INTERNAL_ERROR(459);
   if( (int)optInTimePeriod > (int)(sizeof(local_weights)/sizeof(double)) )
   {
      weights = TA_Malloc( sizeof(double)*optInTimePeriod );
      if( !weights )
      {
         return TA_ALLOC_ERR;
      }
   }
   else
   {
      weights = &local_weights[0];
   }
   /* The expression order is the spec: it is what makes the output bitwise
    * equal to Tulip's beta/alma.c. Keep the floor in binary64 and (j-m) squared
    * in double (an int square overflows at large periods).
    */
   m = (int)floor(optInOffset * (double)lookbackTotal);
   s = (double)optInTimePeriod / optInSigma;
   twoSSq = 2.0 * s * s;
   norm = 0.0;
   for( j = 0; j < optInTimePeriod; j += 1 )
   {
      d = (double)(j - m);
      weights[j] = exp(-(d * d) / twoSSq);
      norm += weights[j];
   }
   for( j = 0; j < optInTimePeriod; j += 1 )
   {
      weights[j] = weights[j] / norm;
   }
   /* 4 bars per pass, each with its own accumulator summed in the same
    * order as the base: the same bits, with 4 independent add chains in
    * flight instead of one. Every window of a pass is read before any of its
    * outputs is written, and the next pass reads past them, so outReal may
    * alias inReal.
    */
   outIdx = 0;
   i = startIdx;
   while( i + 3 <= endIdx )
   {
      s0 = 0.0;
      s1 = 0.0;
      s2 = 0.0;
      s3 = 0.0;
      b = i - lookbackTotal;
      for( j = 0; j < optInTimePeriod; j += 1 )
      {
         wj = weights[j];
         s0 += wj * inReal[b];
         s1 += wj * inReal[b + 1];
         s2 += wj * inReal[b + 2];
         s3 += wj * inReal[b + 3];
         b += 1;
      }
      outReal[outIdx] = s0;
      outReal[outIdx + 1] = s1;
      outReal[outIdx + 2] = s2;
      outReal[outIdx + 3] = s3;
      outIdx += 4;
      i += 4;
   }
   while( i <= endIdx )
   {
      sum = 0.0;
      w = 0;
      for( j = i - lookbackTotal; j <= i; j += 1 )
      {
         sum += weights[w] * inReal[j];
         w += 1;
      }
      outReal[outIdx] = sum;
      outIdx += 1;
      i += 1;
   }
   *outNBElement= outIdx;
   *outBegIdx= startIdx;
   if( weights != &local_weights[0] ) { TA_Free( weights ); weights = &local_weights[0]; }
   return TA_SUCCESS;
}

TA_LIB_API int TA_ALMA_Lookback( int optInTimePeriod, double optInSigma, double optInOffset )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 9;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   if( optInSigma == TA_REAL_DEFAULT )
      optInSigma = 6;
   else if( !(optInSigma >= 1e-2 && optInSigma <= TA_REAL_MAX) )
      return -1;
   if( optInOffset == TA_REAL_DEFAULT )
      optInOffset = 0.85;
   else if( !(optInOffset >= 0e0 && optInOffset <= 1e0) )
      return -1;
   return optInTimePeriod - 1;
}
