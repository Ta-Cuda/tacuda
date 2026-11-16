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
TA_LIB_API TA_RetCode TA_BBW( int    startIdx,
                              int    endIdx,
                              const double inReal[],
                              int optInTimePeriod,
                              double optInNbDevUp,
                              double optInNbDevDn,
                              TA_MAType optInMAType,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outReal[] )
{
   TA_RetCode retCode;
   int i;
   int maBegIdx;
   int maNbElement;
   int offset;
   double middle;
   double deviation;
   double tempReal;
   double upper;
   double lower;
   double *tempBuffer;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return TA_BAD_PARAM;
   if( optInNbDevUp == TA_REAL_DEFAULT )
      optInNbDevUp = 2;
   else if( !(optInNbDevUp >= TA_REAL_MIN && optInNbDevUp <= TA_REAL_MAX) )
      return TA_BAD_PARAM;
   if( optInNbDevDn == TA_REAL_DEFAULT )
      optInNbDevDn = 2;
   else if( !(optInNbDevDn >= TA_REAL_MIN && optInNbDevDn <= TA_REAL_MAX) )
      return TA_BAD_PARAM;
   if( (int)optInMAType == TA_INTEGER_DEFAULT || optInMAType == TA_MAType_DEFAULT )
      optInMAType = 0;
   else if( (int)optInMAType < TA_MATYPE_MIN || (int)optInMAType > TA_MATYPE_MAX )
      return TA_BAD_PARAM;
   if( !inReal )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   if( optInMAType == TA_MAType_SMA )
   {
      /* Keep the middle band's divide in the recurrence, where the running
       * sums' dependency chain hides it; everything downstream of the
       * variance, the square root included, belongs in the tile pass, where
       * it vectorizes.
       */
      double _mid[256];
      double maTotal;
      double shift;
      double varTotal1;
      double varTotal2;
      double meanValue1;
      double variance;
      double _invPeriod;
      double _tempReal;
      double _peakTotal2;
      int _i;
      int _j;
      int _k;
      int _outIdx;
      int _tileBase;
      int _tileEnd;
      int _trailingIdx;
      int _windowStart;
      int _lookbackTotal;
      int _barsSinceReseed;
      _lookbackTotal = optInTimePeriod - 1;
      if( startIdx < _lookbackTotal )
      {
         startIdx = _lookbackTotal;
      }
      if( startIdx > endIdx )
      {
         *outBegIdx= 0;
         *outNBElement= 0;
         return TA_SUCCESS;
      }
      _invPeriod = 1.0 / (double)optInTimePeriod;
      _trailingIdx = startIdx - _lookbackTotal;
      shift = inReal[_trailingIdx];
      maTotal = 0.0;
      varTotal1 = 0.0;
      varTotal2 = 0.0;
      for( _j = _trailingIdx; _j < startIdx; _j += 1 )
      {
         maTotal += inReal[_j];
         _tempReal = inReal[_j] - shift;
         varTotal1 += _tempReal;
         _tempReal *= _tempReal;
         varTotal2 += _tempReal;
      }
      _i = startIdx;
      _outIdx = 0;
      _barsSinceReseed = 32 * optInTimePeriod;
      _peakTotal2 = varTotal2;
      do
      {
         if( endIdx - _i > 255 )
         {
            _tileEnd = _i + 255;
         } else 
         {
            _tileEnd = endIdx;
         }
         _tileBase = _outIdx;
         do
         {
            maTotal += inReal[_i];
            _tempReal = inReal[_i] - shift;
            varTotal1 += _tempReal;
            _tempReal *= _tempReal;
            varTotal2 += _tempReal;
            _peakTotal2 = (varTotal2 > _peakTotal2) ? varTotal2 : _peakTotal2;
            meanValue1 = varTotal1 * _invPeriod;
            variance = varTotal2 * _invPeriod - meanValue1 * meanValue1;
            _mid[_outIdx - _tileBase] = maTotal / optInTimePeriod;
            maTotal -= inReal[_trailingIdx];
            _tempReal = inReal[_trailingIdx] - shift;
            varTotal1 -= _tempReal;
            _tempReal *= _tempReal;
            varTotal2 -= _tempReal;
            _trailingIdx += 1;
            _barsSinceReseed -= 1;
            if( variance < 0.000001 * (_peakTotal2 * _invPeriod) || _barsSinceReseed <= 0 )
            {
               _barsSinceReseed = 32 * optInTimePeriod;
               _windowStart = _i - _lookbackTotal;
               _tempReal = 0.0;
               for( _j = _windowStart; _j <= _i; _j += 1 )
               {
                  _tempReal += inReal[_j];
               }
               shift = _tempReal * _invPeriod;
               varTotal1 = 0.0;
               varTotal2 = 0.0;
               for( _j = _windowStart; _j <= _i; _j += 1 )
               {
                  _tempReal = inReal[_j] - shift;
                  varTotal1 += _tempReal;
                  _tempReal *= _tempReal;
                  varTotal2 += _tempReal;
               }
               meanValue1 = varTotal1 * _invPeriod;
               variance = varTotal2 * _invPeriod - meanValue1 * meanValue1;
               if( variance < 0.000001 * (varTotal2 * _invPeriod) )
               {
                  shift = inReal[_i];
                  varTotal1 = 0.0;
                  varTotal2 = 0.0;
                  for( _j = _windowStart; _j <= _i; _j += 1 )
                  {
                     _tempReal = inReal[_j] - shift;
                     varTotal1 += _tempReal;
                     _tempReal *= _tempReal;
                     varTotal2 += _tempReal;
                  }
                  meanValue1 = varTotal1 * _invPeriod;
                  variance = varTotal2 * _invPeriod - meanValue1 * meanValue1;
               }
               _peakTotal2 = varTotal2;
               if( variance < 0.000000000001 * (varTotal2 * _invPeriod) )
               {
                  variance = 0.0;
               }
               _tempReal = inReal[_windowStart] - shift;
               varTotal1 -= _tempReal;
               _tempReal *= _tempReal;
               varTotal2 -= _tempReal;
            }
            /* outReal may be inReal: at startIdx == optInTimePeriod-1 this
             * slot is the first one the rebuild above reads.
             */
            outReal[_outIdx] = variance;
            _outIdx += 1;
            _i += 1;
         } while( _i <= _tileEnd );
         /* Each band is rounded as TA_BBANDS rounds it and the width is taken
          * from the two rounded bands, then scaled after the divide, which
          * keeps BBW bit-identical to ((upper - lower) / middle) * 100 over
          * TA_BBANDS' outputs.
          *
          * Store, then overwrite: gcc keeps a guarded or selected quotient
          * scalar under -ftrapping-math, and says nothing.
          */
         if( optInNbDevUp == optInNbDevDn )
         {
            for( _k = 0; _k < _outIdx - _tileBase; _k += 1 )
            {
               middle = _mid[_k];
               tempReal = sqrt(outReal[_tileBase + _k]) * optInNbDevUp;
               upper = middle + tempReal;
               lower = middle - tempReal;
               outReal[_tileBase + _k] = (upper - lower) / middle * 100.0;
               if( middle == 0.0 )
               {
                  outReal[_tileBase + _k] = 0.0;
               }
            }
         } else 
         {
            for( _k = 0; _k < _outIdx - _tileBase; _k += 1 )
            {
               middle = _mid[_k];
               deviation = sqrt(outReal[_tileBase + _k]);
               upper = fma(deviation, optInNbDevUp, middle);
               lower = middle - deviation * optInNbDevDn;
               outReal[_tileBase + _k] = (upper - lower) / middle * 100.0;
               if( middle == 0.0 )
               {
                  outReal[_tileBase + _k] = 0.0;
               }
            }
         }
      } while( _i <= endIdx );
      *outNBElement= _outIdx;
      *outBegIdx= startIdx;
      return TA_SUCCESS;
   }
   if( TA_BBW_Lookback(optInTimePeriod,optInNbDevUp,optInNbDevDn,optInMAType) > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   tempBuffer = malloc((endIdx - startIdx + 1) * sizeof(double));
   if( !tempBuffer )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_ALLOC_ERR;
   }
   /* Before the variance: it may be written over inReal. */
   retCode = TA_MA(startIdx,endIdx,inReal,optInTimePeriod,optInMAType,&maBegIdx,&maNbElement,tempBuffer);
   if( retCode != TA_SUCCESS )
   {
      free(tempBuffer);
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   /* From the moving average's begIdx, as TA_BBANDS enters its deviation:
    * the variance's shift and reseed schedule are anchored on its start.
    */
   retCode = TA_VAR(maBegIdx,endIdx,inReal,optInTimePeriod,1.0,outBegIdx,outNBElement,outReal);
   if( retCode != TA_SUCCESS )
   {
      free(tempBuffer);
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   offset = maNbElement - *outNBElement;
   if( optInNbDevUp == optInNbDevDn )
   {
      for( i = 0; i < (int)*outNBElement; i += 1 )
      {
         middle = tempBuffer[i + offset];
         tempReal = sqrt(outReal[i]) * optInNbDevUp;
         upper = middle + tempReal;
         lower = middle - tempReal;
         outReal[i] = (upper - lower) / middle * 100.0;
         if( middle == 0.0 )
         {
            outReal[i] = 0.0;
         }
      }
   } else 
   {
      for( i = 0; i < (int)*outNBElement; i += 1 )
      {
         middle = tempBuffer[i + offset];
         deviation = sqrt(outReal[i]);
         upper = fma(deviation, optInNbDevUp, middle);
         lower = middle - deviation * optInNbDevDn;
         outReal[i] = (upper - lower) / middle * 100.0;
         if( middle == 0.0 )
         {
            outReal[i] = 0.0;
         }
      }
   }
   free(tempBuffer);
   return TA_SUCCESS;
}

TA_LIB_API int TA_BBW_Lookback( int optInTimePeriod, double optInNbDevUp, double optInNbDevDn, TA_MAType optInMAType )
{
   if( (int)optInTimePeriod == TA_INTEGER_DEFAULT )
      optInTimePeriod = 20;
   else if( (int)optInTimePeriod < 2 || (int)optInTimePeriod > 100000 )
      return -1;
   if( optInNbDevUp == TA_REAL_DEFAULT )
      optInNbDevUp = 2;
   else if( !(optInNbDevUp >= TA_REAL_MIN && optInNbDevUp <= TA_REAL_MAX) )
      return -1;
   if( optInNbDevDn == TA_REAL_DEFAULT )
      optInNbDevDn = 2;
   else if( !(optInNbDevDn >= TA_REAL_MIN && optInNbDevDn <= TA_REAL_MAX) )
      return -1;
   if( (int)optInMAType == TA_INTEGER_DEFAULT || optInMAType == TA_MAType_DEFAULT )
      optInMAType = 0;
   else if( (int)optInMAType < TA_MATYPE_MIN || (int)optInMAType > TA_MATYPE_MAX )
      return -1;
   return TA_BBANDS_Lookback(optInTimePeriod,optInNbDevUp,optInNbDevDn,optInMAType);
}
