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

TA_LIB_API TA_RetCode TA_STDDEV( int    startIdx,
                                 int    endIdx,
                                 const double inReal[],
                                 int optInTimePeriod,
                                 double optInNbDev,
                                 int          *outBegIdx,
                                 int          *outNBElement,
                                 double        outReal[] )
{
   int i;
   TA_RetCode retCode;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 5;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( optInNbDev == TA_REAL_DEFAULT )
      optInNbDev = 1;
   else if( !(optInNbDev >= TA_REAL_MIN && optInNbDev <= TA_REAL_MAX) )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* Nothing to produce: the range ends before the lookback. Return before
    * touching anything.
    *
    * Same shape as the guard in apo and bbands: the variance below runs on the
    * same range and its lookback IS stddev's, so it declines and yields 0,0
    * without reading. Observably identical, but it makes "a range that ends before
    * the lookback reads nothing" true of stddev itself rather than only of var.
    * Pinned by the zero-length no-I/O probe over every guarded core.
    */
   if( TA_STDDEV_Lookback(optInTimePeriod,optInNbDev) > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   /* Calculate the variance. */
   retCode = TA_VAR(startIdx,endIdx,inReal,optInTimePeriod,1.0,outBegIdx,outNBElement,outReal);
   if( retCode != TA_SUCCESS )
   {
      return retCode;
   }
   /* Calculate the square root of each variance, this
    * is the standard deviation.
    *
    * Multiply also by the ratio specified.
    *
    * Unconditional. var owns the sign: it returns a non-negative variance,
    * exactly 0 on a window of identical values (var.c). What used to
    * stand here instead - zero the output wherever the variance fell under
    * TA_EPSILON - compared a SQUARED quantity to a fixed 1e-14, which is a cliff
    * at a price level rather than a noise floor: a $100.00 instrument quoted in
    * 1e-8 ticks has a variance around 1e-16 and came back as exactly 0 on every
    * bar, with TA_SUCCESS and nothing to say it had been suppressed (#243).
    * Dropping it also leaves a pure map, which the branch had kept sqrt out of.
    */
   if( optInNbDev != 1.0 )
   {
      for( i = 0; i < (int)*outNBElement; i += 1 )
      {
         outReal[i] = sqrt(outReal[i]) * optInNbDev;
      }
   } else 
   {
      for( i = 0; i < (int)*outNBElement; i += 1 )
      {
         outReal[i] = sqrt(outReal[i]);
      }
   }
   return TA_SUCCESS;
}

TA_LIB_API int TA_STDDEV_Lookback( int optInTimePeriod, double optInNbDev )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 5;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   if( optInNbDev == TA_REAL_DEFAULT )
      optInNbDev = 1;
   else if( !(optInNbDev >= TA_REAL_MIN && optInNbDev <= TA_REAL_MAX) )
      return -1;
   /* Lookback is driven by the variance. */
   return TA_VAR_Lookback(optInTimePeriod,optInNbDev);
}
