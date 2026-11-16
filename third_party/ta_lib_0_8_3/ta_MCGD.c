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

TA_LIB_API TA_RetCode TA_MCGD( int    startIdx,
                               int    endIdx,
                               const double inReal[],
                               int optInTimePeriod,
                               int          *outBegIdx,
                               int          *outNBElement,
                               double        outReal[] )
{
   int i;
   int outIdx;
   int today;
   int lookbackTotal;
   int nbMCGD;
   double prevMD;
   double tempMD;
   double tempReal;
   double ratio;
   double period;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   *outBegIdx= 0;
   *outNBElement= 0;
   lookbackTotal = TA_MCGD_Lookback(optInTimePeriod);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   period = (double)optInTimePeriod;
   /* Keep the denominator free of pow(): a transcendental would break the
    * four languages' bit-identity.
    *
    * A step whose value is not finite (a zero price, a zero line meeting a
    * zero price, or a price so small against the line that the step
    * overflows) holds the previous value, since every later bar reads it.
    */
   today = startIdx - lookbackTotal;
   prevMD = inReal[today];
   today += 1;
   i = lookbackTotal;
   while( i != 0 )
   {
      tempReal = inReal[today];
      ratio = tempReal / prevMD;
      tempMD = prevMD + (tempReal - prevMD) / (period * (ratio * ratio * (ratio * ratio)));
      if( TA_IS_FINITE(tempMD) )
      {
         prevMD = tempMD;
      }
      today += 1;
      i -= 1;
   }
   outIdx = 1;
   outReal[0] = prevMD;
   nbMCGD = endIdx - startIdx + 1;
   while( --nbMCGD != 0 )
   {
      tempReal = inReal[today];
      ratio = tempReal / prevMD;
      tempMD = prevMD + (tempReal - prevMD) / (period * (ratio * ratio * (ratio * ratio)));
      if( TA_IS_FINITE(tempMD) )
      {
         prevMD = tempMD;
      }
      outReal[outIdx++] = prevMD;
      today += 1;
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_NOINLINE TA_LIB_API int TA_MCGD_Lookback( int optInTimePeriod )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 14;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   return optInTimePeriod - 1 + TA_GLOBALS_UNSTABLE(TA_FUNC_UNST_MCGD,Mcgd,5 * 4 * optInTimePeriod,5 * 8 * optInTimePeriod);
}
