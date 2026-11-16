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

TA_LIB_API TA_RetCode TA_KC( int    startIdx,
                             int    endIdx,
                             const double inHigh[],
                             const double inLow[],
                             const double inClose[],
                             int optInTimePeriod,
                             int optInATRPeriod,
                             double optInNbDev,
                             int          *outBegIdx,
                             int          *outNBElement,
                             double        outRealUpperBand[],
                             double        outRealMiddleBand[],
                             double        outRealLowerBand[] )
{
   TA_RetCode retCode;
   int i;
   int lookbackTotal;
   int emaLookback;
   int tpStartIdx;
   int tempBegIdx;
   int tempNbElement;
   double tempReal;
   double middle;
   double *tempTP;
   double *tempATR;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInATRPeriod == TA_INTEGER_DEFAULT )
      optInATRPeriod = 10;
   else if( (int)optInATRPeriod < 1 || (int)optInATRPeriod > 100000 )
      return TA_BAD_PARAM;
   if( optInNbDev == TA_REAL_DEFAULT )
      optInNbDev = 2;
   else if( !(optInNbDev >= TA_REAL_MIN && optInNbDev <= TA_REAL_MAX) )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !inClose )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outRealUpperBand )
      return TA_BAD_PARAM;
   if( !outRealMiddleBand )
      return TA_BAD_PARAM;
   if( !outRealLowerBand )
      return TA_BAD_PARAM;
   if( outRealUpperBand == outRealMiddleBand || outRealUpperBand == outRealLowerBand || outRealMiddleBand == outRealLowerBand )
      return TA_BAD_PARAM;

   emaLookback = TA_EMA_Lookback(optInTimePeriod);
   lookbackTotal = TA_KC_Lookback(optInTimePeriod,optInATRPeriod,optInNbDev);
   /* Nothing to produce: the range ends before the lookback. Return before
    * touching anything, so that a caller-supplied input which stops short of
    * endIdx is never read past its end.
    */
   if( lookbackTotal > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   /* Each leg is entered at its OWN lookback, so each is the shipped function
    * over this range and nothing here over-warms the shorter one: a caller who
    * wants the band converged sets TA_FUNC_UNST_ATR, exactly as they would when
    * calling TA_ATR directly. The typical price is a derived input, so it is
    * materialized only over the window the moving average reads.
    */
   tpStartIdx = startIdx - emaLookback;
   tempTP = malloc((endIdx - tpStartIdx + 1) * sizeof(double));
   if( !tempTP )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_ALLOC_ERR;
   }
   tempATR = malloc((endIdx - startIdx + 1) * sizeof(double));
   if( !tempATR )
   {
      free(tempTP);
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_ALLOC_ERR;
   }
   retCode = TA_TYPPRICE(tpStartIdx,endIdx,inHigh,inLow,inClose,&tempBegIdx,&tempNbElement,tempTP);
   if( retCode != TA_SUCCESS )
   {
      free(tempTP);
      free(tempATR);
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   /* The ATR consumes the price inputs before the moving average below writes
    * the middle band, which may be aliased onto one of them.
    */
   retCode = TA_ATR(startIdx,endIdx,inHigh,inLow,inClose,optInATRPeriod,&tempBegIdx,&tempNbElement,tempATR);
   if( retCode != TA_SUCCESS )
   {
      free(tempTP);
      free(tempATR);
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   /* tempTP is bar-tpStartIdx relative, so entering the moving average at its
    * own lookback puts its first output on startIdx, where the ATR's already is.
    */
   retCode = TA_EMA(emaLookback,endIdx - tpStartIdx,tempTP,optInTimePeriod,outBegIdx,outNBElement,outRealMiddleBand);
   if( retCode != TA_SUCCESS || (int)*outNBElement == 0 )
   {
      free(tempTP);
      free(tempATR);
      *outNBElement= 0;
      return retCode;
   }
   *outBegIdx= startIdx;
   for( i = 0; i < (int)*outNBElement; i += 1 )
   {
      middle = outRealMiddleBand[i];
      tempReal = tempATR[i] * optInNbDev;
      outRealUpperBand[i] = middle + tempReal;
      outRealLowerBand[i] = middle - tempReal;
   }
   free(tempTP);
   free(tempATR);
   return TA_SUCCESS;
}

TA_LIB_API int TA_KC_Lookback( int optInTimePeriod, int optInATRPeriod, double optInNbDev )
{
   int emaLookback;
   int atrLookback;
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   if( (int)optInATRPeriod == TA_INTEGER_DEFAULT )
      optInATRPeriod = 10;
   else if( (int)optInATRPeriod < 1 || (int)optInATRPeriod > 100000 )
      return -1;
   if( optInNbDev == TA_REAL_DEFAULT )
      optInNbDev = 2;
   else if( !(optInNbDev >= TA_REAL_MIN && optInNbDev <= TA_REAL_MAX) )
      return -1;
   /* A band value needs BOTH the centre line and the ATR at the same bar, so the
    * first valid output is the later of the two lookbacks. Each term is exactly
    * the lookback of the function it comes from and is never restated here, which
    * is what makes KC inherit TA_FUNC_UNST_EMA and TA_FUNC_UNST_ATR from its two
    * callees. Reporting the honest max keeps outBegIdx == lookback (issue #99),
    * which streaming's Open depends on.
    */
   emaLookback = TA_EMA_Lookback(optInTimePeriod);
   atrLookback = TA_ATR_Lookback(optInATRPeriod);
   return (emaLookback > atrLookback) ? emaLookback : atrLookback;
}
