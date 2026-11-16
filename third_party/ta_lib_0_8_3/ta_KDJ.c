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

TA_LIB_API TA_RetCode TA_KDJ( int    startIdx,
                              int    endIdx,
                              const double inHigh[],
                              const double inLow[],
                              const double inClose[],
                              int optInFastK_Period,
                              int optInSlowK_Period,
                              TA_MAType optInSlowK_MAType,
                              int optInSlowD_Period,
                              TA_MAType optInSlowD_MAType,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outK[],
                              double        outD[],
                              double        outJ[] )
{
   TA_RetCode retCode;
   int i;
   int lookbackTotal;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInFastK_Period == TA_INTEGER_DEFAULT )
      optInFastK_Period = 9;
   else if( (int)optInFastK_Period < 1 || (int)optInFastK_Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSlowK_Period == TA_INTEGER_DEFAULT )
      optInSlowK_Period = 3;
   else if( (int)optInSlowK_Period < 1 || (int)optInSlowK_Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSlowK_MAType == TA_INTEGER_DEFAULT || optInSlowK_MAType == TA_MAType_DEFAULT )
      optInSlowK_MAType = 13;
   else if( (int)optInSlowK_MAType < TA_MATYPE_MIN || (int)optInSlowK_MAType > TA_MATYPE_MAX )
      return TA_BAD_PARAM;
   if( (int)optInSlowD_Period == TA_INTEGER_DEFAULT )
      optInSlowD_Period = 3;
   else if( (int)optInSlowD_Period < 1 || (int)optInSlowD_Period > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSlowD_MAType == TA_INTEGER_DEFAULT || optInSlowD_MAType == TA_MAType_DEFAULT )
      optInSlowD_MAType = 13;
   else if( (int)optInSlowD_MAType < TA_MATYPE_MIN || (int)optInSlowD_MAType > TA_MATYPE_MAX )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !inClose )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outK )
      return TA_BAD_PARAM;
   if( !outD )
      return TA_BAD_PARAM;
   if( !outJ )
      return TA_BAD_PARAM;
   if( outK == outD || outK == outJ || outD == outJ )
      return TA_BAD_PARAM;

   lookbackTotal = TA_KDJ_Lookback(optInFastK_Period,optInSlowK_Period,optInSlowK_MAType,optInSlowD_Period,optInSlowD_MAType);
   /* Nothing to produce: the range ends before the lookback. Answering here
    * keeps the sub-call out of the phantom-I/O sweep's zero-length range, where
    * its own argument check would reject before any array is touched.
    */
   if( lookbackTotal > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   retCode = TA_STOCH(startIdx,endIdx,inHigh,inLow,inClose,optInFastK_Period,optInSlowK_Period,optInSlowK_MAType,optInSlowD_Period,optInSlowD_MAType,outBegIdx,outNBElement,outK,outD);
   if( retCode != TA_SUCCESS || (int)*outNBElement == 0 )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   /* Keep this one Sub expression: spelling it as an Add of a negated term
    * would arm the multiply-add fusion and move the last bits of J.
    */
   for( i = 0; i < (int)*outNBElement; i += 1 )
   {
      outJ[i] = 3.0 * outK[i] - 2.0 * outD[i];
   }
   return TA_SUCCESS;
}

TA_LIB_API int TA_KDJ_Lookback( int optInFastK_Period, int optInSlowK_Period, TA_MAType optInSlowK_MAType, int optInSlowD_Period, TA_MAType optInSlowD_MAType )
{
   if( (int)optInFastK_Period == TA_INTEGER_DEFAULT )
      optInFastK_Period = 9;
   else if( (int)optInFastK_Period < 1 || (int)optInFastK_Period > 100000 )
      return -1;
   if( (int)optInSlowK_Period == TA_INTEGER_DEFAULT )
      optInSlowK_Period = 3;
   else if( (int)optInSlowK_Period < 1 || (int)optInSlowK_Period > 100000 )
      return -1;
   if( (int)optInSlowK_MAType == TA_INTEGER_DEFAULT || optInSlowK_MAType == TA_MAType_DEFAULT )
      optInSlowK_MAType = 13;
   else if( (int)optInSlowK_MAType < TA_MATYPE_MIN || (int)optInSlowK_MAType > TA_MATYPE_MAX )
      return -1;
   if( (int)optInSlowD_Period == TA_INTEGER_DEFAULT )
      optInSlowD_Period = 3;
   else if( (int)optInSlowD_Period < 1 || (int)optInSlowD_Period > 100000 )
      return -1;
   if( (int)optInSlowD_MAType == TA_INTEGER_DEFAULT || optInSlowD_MAType == TA_MAType_DEFAULT )
      optInSlowD_MAType = 13;
   else if( (int)optInSlowD_MAType < TA_MATYPE_MIN || (int)optInSlowD_MAType > TA_MATYPE_MAX )
      return -1;
   return TA_STOCH_Lookback(optInFastK_Period,optInSlowK_Period,optInSlowK_MAType,optInSlowD_Period,optInSlowD_MAType);
}
