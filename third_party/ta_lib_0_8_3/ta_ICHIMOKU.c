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

TA_LIB_API TA_RetCode TA_ICHIMOKU( int    startIdx,
                                   int    endIdx,
                                   const double inHigh[],
                                   const double inLow[],
                                   int optInTenkanPeriod,
                                   int optInKijunPeriod,
                                   int optInSenkouBPeriod,
                                   int          *outBegIdx,
                                   int          *outNBElement,
                                   double        outTenkanSen[],
                                   double        outKijunSen[],
                                   double        outSenkouSpanA[],
                                   double        outSenkouSpanB[] )
{
   TA_RetCode retCode;
   int lookbackTotal;
   int n;
   int i;
   int tempBegIdx;
   int tempNbElement;
   double *tempT;
   double *tempK;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTenkanPeriod == TA_INTEGER_DEFAULT )
      optInTenkanPeriod = 9;
   else if( (int)optInTenkanPeriod < 2 || (int)optInTenkanPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInKijunPeriod == TA_INTEGER_DEFAULT )
      optInKijunPeriod = 26;
   else if( (int)optInKijunPeriod < 2 || (int)optInKijunPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSenkouBPeriod == TA_INTEGER_DEFAULT )
      optInSenkouBPeriod = 52;
   else if( (int)optInSenkouBPeriod < 2 || (int)optInSenkouBPeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outTenkanSen )
      return TA_BAD_PARAM;
   if( !outKijunSen )
      return TA_BAD_PARAM;
   if( !outSenkouSpanA )
      return TA_BAD_PARAM;
   if( !outSenkouSpanB )
      return TA_BAD_PARAM;
   if( outTenkanSen == outKijunSen || outTenkanSen == outSenkouSpanA || outTenkanSen == outSenkouSpanB || outKijunSen == outSenkouSpanA || outKijunSen == outSenkouSpanB || outSenkouSpanA == outSenkouSpanB )
      return TA_BAD_PARAM;

   *outBegIdx= 0;
   *outNBElement= 0;
   lookbackTotal = TA_ICHIMOKU_Lookback(optInTenkanPeriod,optInKijunPeriod,optInSenkouBPeriod);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   n = endIdx - startIdx + 1;
   tempT = malloc(n * sizeof(double));
   if( !tempT )
   {
      return TA_ALLOC_ERR;
   }
   tempK = malloc(n * sizeof(double));
   if( !tempK )
   {
      free(tempT);
      return TA_ALLOC_ERR;
   }
   /* An output may alias inHigh or inLow: no output is written before the
    * last midprice call, which is the last read of either input.
    */
   retCode = TA_MIDPRICE(startIdx,endIdx,inHigh,inLow,optInTenkanPeriod,&tempBegIdx,&tempNbElement,tempT);
   if( retCode != TA_SUCCESS )
   {
      free(tempT);
      free(tempK);
      return retCode;
   }
   retCode = TA_MIDPRICE(startIdx,endIdx,inHigh,inLow,optInKijunPeriod,&tempBegIdx,&tempNbElement,tempK);
   if( retCode != TA_SUCCESS )
   {
      free(tempT);
      free(tempK);
      return retCode;
   }
   retCode = TA_MIDPRICE(startIdx,endIdx,inHigh,inLow,optInSenkouBPeriod,&tempBegIdx,&tempNbElement,outSenkouSpanB);
   if( retCode != TA_SUCCESS )
   {
      free(tempT);
      free(tempK);
      return retCode;
   }
   /* Span A halves the two rounded midpoints. (hh+ll+hh+ll)/4 is a different
    * double in the last bit.
    */
   retCode = TA_MEDPRICE(0,n - 1,tempT,tempK,&tempBegIdx,&tempNbElement,outSenkouSpanA);
   if( retCode != TA_SUCCESS )
   {
      free(tempT);
      free(tempK);
      return retCode;
   }
   for( i = 0; i < n; i += 1 )
   {
      outTenkanSen[i] = tempT[i];
      outKijunSen[i] = tempK[i];
   }
   free(tempT);
   free(tempK);
   *outBegIdx= startIdx;
   *outNBElement= n;
   return TA_SUCCESS;
}

TA_LIB_API int TA_ICHIMOKU_Lookback( int optInTenkanPeriod, int optInKijunPeriod, int optInSenkouBPeriod )
{
   int longest;
   if( (int)optInTenkanPeriod == TA_INTEGER_DEFAULT )
      optInTenkanPeriod = 9;
   else if( (int)optInTenkanPeriod < 2 || (int)optInTenkanPeriod > 100000 )
      return -1;
   if( (int)optInKijunPeriod == TA_INTEGER_DEFAULT )
      optInKijunPeriod = 26;
   else if( (int)optInKijunPeriod < 2 || (int)optInKijunPeriod > 100000 )
      return -1;
   if( (int)optInSenkouBPeriod == TA_INTEGER_DEFAULT )
      optInSenkouBPeriod = 52;
   else if( (int)optInSenkouBPeriod < 2 || (int)optInSenkouBPeriod > 100000 )
      return -1;
   /* Nothing orders the three periods: any of them can be the longest. */
   longest = optInTenkanPeriod;
   if( optInKijunPeriod > longest )
   {
      longest = optInKijunPeriod;
   }
   if( optInSenkouBPeriod > longest )
   {
      longest = optInSenkouBPeriod;
   }
   return longest - 1;
}
