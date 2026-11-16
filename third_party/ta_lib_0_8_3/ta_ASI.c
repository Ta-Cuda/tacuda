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

TA_LIB_API TA_RetCode TA_ASI( int    startIdx,
                              int    endIdx,
                              const double inOpen[],
                              const double inHigh[],
                              const double inLow[],
                              const double inClose[],
                              double optInLimitMove,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outReal[] )
{
   int i;
   int outIdx;
   double prevClose;
   double prevBody;
   double tempOpen;
   double tempHigh;
   double tempLow;
   double tempClose;
   double body;
   double n;
   double up;
   double dn;
   double rg;
   double k;
   double r;
   double swing;
   double sum;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( optInLimitMove == TA_REAL_DEFAULT )
      optInLimitMove = 3;
   else if( !(optInLimitMove >= 1e-8 && optInLimitMove <= TA_REAL_MAX) )
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

   /* SI's per-bar statements, kept identical: the regression test holds this
    * function bit for bit to CUMSUM over SI. Bar startIdx is read before the
    * anchor store, which may overwrite it when outReal aliases an input.
    */
   prevClose = inClose[startIdx];
   prevBody = prevClose - inOpen[startIdx];
   sum = 0.0;
   outReal[0] = sum;
   outIdx = 1;
   for( i = startIdx + 1; i <= endIdx; i += 1 )
   {
      tempOpen = inOpen[i];
      tempHigh = inHigh[i];
      tempLow = inLow[i];
      tempClose = inClose[i];
      body = tempClose - tempOpen;
      n = tempClose - prevClose;
      n += 0.5 * body;
      n += 0.25 * prevBody;
      up = fabs(tempHigh - prevClose);
      dn = fabs(tempLow - prevClose);
      rg = fabs(tempHigh - tempLow);
      k = max(up,dn);
      r = max(up - 0.5 * dn,dn - 0.5 * up);
      r = max(r,rg);
      r += 0.25 * fabs(prevBody);
      if( r == 0.0 )
      {
         swing = 0.0;
      } else 
      {
         swing = 50.0 * (n / r) * (k / optInLimitMove);
      }
      sum += swing;
      outReal[outIdx] = sum;
      outIdx += 1;
      prevClose = tempClose;
      prevBody = body;
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_ASI_Lookback( double optInLimitMove )
{
   if( optInLimitMove == TA_REAL_DEFAULT )
      optInLimitMove = 3;
   else if( !(optInLimitMove >= 1e-8 && optInLimitMove <= TA_REAL_MAX) )
      return -1;
   return 0;
}
