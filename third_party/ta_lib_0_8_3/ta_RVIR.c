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

TA_LIB_API TA_RetCode TA_RVIR( int    startIdx,
                               int    endIdx,
                               const double inHigh[],
                               const double inLow[],
                               int optInTimePeriod,
                               int optInStdDevPeriod,
                               int          *outBegIdx,
                               int          *outNBElement,
                               double        outReal[] )
{
   double *tempHigh;
   TA_RetCode retCode;
   int i;
   int lookbackTotal;
   int tempBegIdx;
   int tempNbElement;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInStdDevPeriod == TA_INTEGER_DEFAULT )
      optInStdDevPeriod = 10;
   else if( (int)optInStdDevPeriod < 2 || (int)optInStdDevPeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   *outBegIdx= 0;
   *outNBElement= 0;
   lookbackTotal = TA_RVIR_Lookback(optInTimePeriod,optInStdDevPeriod);
   /* Nothing is allocated and no input is read when the range cannot produce a
    * value, so a caller-supplied input that stops short of endIdx is never
    * read past its end (kc.c).
    */
   if( lookbackTotal > endIdx )
   {
      return TA_SUCCESS;
   }
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   /* Each leg is the shipped TA_RVI over this range, entered at the same
    * startIdx: same function, same parameters, same lookback, so the two
    * outputs are aligned bar for bar with no anchor to reconcile.
    *
    * Two calls rather than one fused loop. The fused body is expressible --
    * two copies of TA_RVI's state driven by one bar cursor -- but it carries
    * two window-start cursors, and the streaming analyzer accepts exactly one
    * ("extrema automaton: expected exactly one window-start variable"), so
    * fusing would cost this function the streaming tier. Composing keeps it,
    * for the reason KC's legs do: each leg streams as itself.
    */
   tempHigh = malloc((endIdx - startIdx + 1) * sizeof(double));
   if( !tempHigh )
   {
      return TA_ALLOC_ERR;
   }
   /* Either input may be aliased onto outReal. Both legs are safe against that
    * for TA_RVI's own reason -- its write index trails its read index by the
    * lookback and never overtakes it -- so the order of the two calls is an
    * implementation detail, not what makes the aliasing safe. Swapping them
    * leaves every value identical, which is why the aliasing test earns its
    * keep against the scratch buffer's extent rather than against this order.
    */
   retCode = TA_RVI(startIdx,endIdx,inHigh,optInTimePeriod,optInStdDevPeriod,&tempBegIdx,&tempNbElement,tempHigh);
   if( retCode != TA_SUCCESS )
   {
      free(tempHigh);
      return retCode;
   }
   retCode = TA_RVI(startIdx,endIdx,inLow,optInTimePeriod,optInStdDevPeriod,outBegIdx,outNBElement,outReal);
   if( retCode != TA_SUCCESS )
   {
      free(tempHigh);
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   /* Each leg has already resolved its own zero-total case to TA_RVI's neutral
    * 50 before the average is taken, so a tie in one series never drags the
    * other. Scaling by one half is exact, so the spelling of the average is
    * not a variant: 0.5*(a+b) is one rounding of a+b, as are (a+b)/2 and
    * 0.5*a + 0.5*b.
    */
   for( i = 0; i < *outNBElement; i += 1 )
   {
      outReal[i] = 0.5 * (tempHigh[i] + outReal[i]);
   }
   free(tempHigh);
   return TA_SUCCESS;
}

TA_LIB_API int TA_RVIR_Lookback( int optInTimePeriod, int optInStdDevPeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 1 || (int)optInTimePeriod > 100000 )
      return -1;
   if( (int)optInStdDevPeriod == TA_INTEGER_DEFAULT )
      optInStdDevPeriod = 10;
   else if( (int)optInStdDevPeriod < 2 || (int)optInStdDevPeriod > 100000 )
      return -1;
   /* Both legs are the shipped TA_RVI at the same parameters, so they warm on
    * the same bar and there is no max() to take. Stated as the callee's
    * lookback rather than restating the arithmetic, which is what makes this
    * function inherit TA_FUNC_UNST_RVI the way KC inherits its two (kc.c).
    */
   return TA_RVI_Lookback(optInTimePeriod,optInStdDevPeriod);
}
