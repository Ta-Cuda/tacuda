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
TA_LIB_API TA_RetCode TA_BBANDS( int    startIdx,
                                 int    endIdx,
                                 const double inReal[],
                                 int optInTimePeriod,
                                 double optInNbDevUp,
                                 double optInNbDevDn,
                                 TA_MAType optInMAType,
                                 int          *outBegIdx,
                                 int          *outNBElement,
                                 double        outRealUpperBand[],
                                 double        outRealMiddleBand[],
                                 double        outRealLowerBand[] )
{
   TA_RetCode retCode;
   int i;
   int maBegIdx;
   int shiftIdx;
   double tempReal;
   double tempReal2;
   double *tempBuffer1;
   double *tempBuffer2;

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
   if( !outRealUpperBand )
      return TA_BAD_PARAM;
   if( !outRealMiddleBand )
      return TA_BAD_PARAM;
   if( !outRealLowerBand )
      return TA_BAD_PARAM;
   if( outRealUpperBand == outRealMiddleBand || outRealUpperBand == outRealLowerBand || outRealMiddleBand == outRealLowerBand )
      return TA_BAD_PARAM;

   if( optInMAType == TA_MAType_SMA )
   {
      /* SMA fast path: the middle band (SMA) and the standard deviation share one
       * pass over the window below. Bit-identical to the general MA + STDDEV path
       * (which the stream composes for every MA type).
       *
       * Identify TWO temporary buffers among the outputs so the calculation
       * needs no memory allocation; whenever possible make tempBuffer1 be the
       * middle band output, saving one copy operation.
       */
      if( inReal == outRealUpperBand )
      {
         tempBuffer1 = outRealMiddleBand;
         tempBuffer2 = outRealLowerBand;
      } else if( inReal == outRealLowerBand )
      {
         tempBuffer1 = outRealMiddleBand;
         tempBuffer2 = outRealUpperBand;
      } else if( inReal == outRealMiddleBand )
      {
         tempBuffer1 = outRealLowerBand;
         tempBuffer2 = outRealUpperBand;
      } else 
      {
         tempBuffer1 = outRealMiddleBand;
         tempBuffer2 = outRealUpperBand;
      }
      /* Check that the caller is not doing tricky things.
       * (like using the input buffer in two output!)
       */
      if( tempBuffer1 == inReal || tempBuffer2 == inReal )
      {
         return TA_BAD_PARAM;
      }
      /* One pass with two independent recurrences: the SMA running sum (maTotal,
       * mean -> tempBuffer1, bit-identical to TA_MA(SMA)) and the shifted-data
       * variance (-> tempBuffer2, bit-identical to TA_STDDEV/TA_VAR - see var.c).
       * The variance carries its own shift and reseed; the SMA sum is untouched by
       * it. tempBuffer1/2 never alias inReal (checked above).
       */
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
      int _outIdx;
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
         maTotal += inReal[_i];
         _tempReal = inReal[_i] - shift;
         varTotal1 += _tempReal;
         _tempReal *= _tempReal;
         varTotal2 += _tempReal;
         _peakTotal2 = (varTotal2 > _peakTotal2) ? varTotal2 : _peakTotal2;
         meanValue1 = varTotal1 * _invPeriod;
         variance = varTotal2 * _invPeriod - meanValue1 * meanValue1;
         tempBuffer1[_outIdx] = maTotal / optInTimePeriod;
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
            /* The floor from var.c, verbatim: it owns the sign, so the
             * square root below can be unconditional.
             */
            if( variance < 0.000000000001 * (varTotal2 * _invPeriod) )
            {
               variance = 0.0;
            }
            _tempReal = inReal[_windowStart] - shift;
            varTotal1 -= _tempReal;
            _tempReal *= _tempReal;
            varTotal2 -= _tempReal;
         }
         /* The TA_EPSILON test that used to stand here compared a SQUARED
          * quantity to a fixed 1e-14 and flattened all three bands onto each
          * other for any finely quoted series (#243). What replaces it skips
          * the root ONLY where the answer is already known, because the
          * rebuild above has made it exactly 0 -- worth doing because
          * this root, unlike stddev.c's, sits in the fused loop with a
          * carried dependency and cannot vectorize, so running it on flat
          * input cost 1.59x.
          *
          * `!= 0.0` and not `> 0.0`: the two differ only on NaN, and `> 0.0`
          * sends it to the false arm, which would emit a zero-width band
          * where stddev.c, BBANDS' own non-SMA paths and BBANDS_Open all
          * return NaN. Reaching it needs a squared deviation to overflow
          * (|x| ~ 1.3e154, far outside TA_REAL_MAX), so nothing in the
          * declared domain can tell the two apart -- but there is no reason
          * to buy the disagreement, and `!= 0.0` is what this is for.
          */
         tempBuffer2[_outIdx] = (variance != 0.0) ? sqrt(variance) : 0.0;
         _outIdx += 1;
         _i += 1;
      } while( _i <= endIdx );
      *outNBElement= _outIdx;
      *outBegIdx= startIdx;
      /* Copy the MA calculation into the middle band ouput, unless
       * the calculation was done into it already!
       */
      if( tempBuffer1 != outRealMiddleBand )
      {
         memcpy(outRealMiddleBand,tempBuffer1,*outNBElement * sizeof(double));
      }
      /* Now do a tight loop to calculate the upper/lower band at the same time. */
      if( optInNbDevUp == optInNbDevDn )
      {
         for( i = 0; i < (int)*outNBElement; i += 1 )
         {
            tempReal = tempBuffer2[i] * optInNbDevUp;
            tempReal2 = outRealMiddleBand[i];
            outRealUpperBand[i] = tempReal2 + tempReal;
            outRealLowerBand[i] = tempReal2 - tempReal;
         }
      } else 
      {
         for( i = 0; i < (int)*outNBElement; i += 1 )
         {
            tempReal = tempBuffer2[i];
            tempReal2 = outRealMiddleBand[i];
            outRealUpperBand[i] = fma(tempReal, optInNbDevUp, tempReal2);
            outRealLowerBand[i] = tempReal2 - tempReal * optInNbDevDn;
         }
      }
      return TA_SUCCESS;
   }
   /* General path (every MA type other than SMA): the middle band is the moving
    * average and the deviation is the standard deviation of the input, combined
    * at the same bar. Two intermediate buffers are allocated so the input may
    * safely alias an output (it is only read here).
    */
   /* Nothing to produce: the range ends before the lookback. Return before
    * touching anything.
    *
    * Without this the moving average below runs first, and for the MA types whose
    * lookback is BELOW the deviation's - TA_MAType_DISABLED (MA lookback 0) and
    * TA_MAType_MAMA at optInTimePeriod >= 34 - it reads the whole range and
    * computes a middle band the empty standard deviation then discards.
    * Observably identical (the empty deviation already yields 0,0 here), but it
    * is the difference between "a range that ends before the lookback reads nothing"
    * being true of this function and being false: with a caller-supplied inReal
    * that stops short of endIdx, that discarded work is an out-of-bounds read.
    * The SMA fast path above needs no such guard - its own lookback IS the
    * deviation's, so its clamp already covers it. Pinned by the zero-length
    * no-I/O probe over every guarded core, at every MA type.
    */
   if( TA_BBANDS_Lookback(optInTimePeriod,optInNbDevUp,optInNbDevDn,optInMAType) > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   tempBuffer1 = malloc((endIdx - startIdx + 1) * sizeof(double));
   if( !tempBuffer1 )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_ALLOC_ERR;
   }
   tempBuffer2 = malloc((endIdx - startIdx + 1) * sizeof(double));
   if( !tempBuffer2 )
   {
      free(tempBuffer1);
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_ALLOC_ERR;
   }
   /* Calculate the middle band moving average. */
   retCode = TA_MA(startIdx,endIdx,inReal,optInTimePeriod,optInMAType,outBegIdx,outNBElement,tempBuffer1);
   if( retCode != TA_SUCCESS || (int)*outNBElement == 0 )
   {
      *outNBElement= 0;
      free(tempBuffer1);
      free(tempBuffer2);
      return retCode;
   }
   /* Remember where the moving average begins, to realign it below. */
   maBegIdx = (int)*outBegIdx;
   /* Calculate the Standard Deviation into tempBuffer2. */
   retCode = TA_STDDEV((int)*outBegIdx,endIdx,inReal,optInTimePeriod,1.0,outBegIdx,outNBElement,tempBuffer2);
   if( retCode != TA_SUCCESS )
   {
      *outNBElement= 0;
      free(tempBuffer1);
      free(tempBuffer2);
      return retCode;
   }
   /* When the standard deviation (lookback optInTimePeriod-1) clamps to a later
    * begIdx than the moving average did - as with TA_MAType_MAMA (constant
    * lookback 32) and optInTimePeriod >= 34 - the MA in tempBuffer1 still starts
    * at the earlier maBegIdx. Copy it forward from that shift into the middle
    * band so each band value pairs the moving average and standard deviation of
    * the same bar. The guarded subtraction keeps shiftIdx non-negative even when
    * the standard deviation produced no output (an empty range leaves *outBegIdx
    * at 0), which the unconditional copy below then handles as a zero-length move.
    */
   if( (int)*outBegIdx > maBegIdx )
   {
      shiftIdx = (int)*outBegIdx - maBegIdx;
   } else 
   {
      shiftIdx = 0;
   }
   memmove(outRealMiddleBand,&tempBuffer1[shiftIdx],*outNBElement * sizeof(double));
   /* Now do a tight loop to calculate the upper/lower band at the same time. */
   if( optInNbDevUp == optInNbDevDn )
   {
      for( i = 0; i < (int)*outNBElement; i += 1 )
      {
         tempReal = tempBuffer2[i] * optInNbDevUp;
         tempReal2 = outRealMiddleBand[i];
         outRealUpperBand[i] = tempReal2 + tempReal;
         outRealLowerBand[i] = tempReal2 - tempReal;
      }
   } else 
   {
      for( i = 0; i < (int)*outNBElement; i += 1 )
      {
         tempReal2 = outRealMiddleBand[i];
         outRealUpperBand[i] = fma(tempBuffer2[i], optInNbDevUp, tempReal2);
         outRealLowerBand[i] = tempReal2 - tempBuffer2[i] * optInNbDevDn;
      }
   }
   free(tempBuffer1);
   free(tempBuffer2);
   return TA_SUCCESS;
}

TA_LIB_API int TA_BBANDS_Lookback( int optInTimePeriod, double optInNbDevUp, double optInNbDevDn, TA_MAType optInMAType )
{
   int maLookback;
   int stddevLookback;
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
   /* A band value needs BOTH the middle-band moving average and the standard
    * deviation of the outer bands, so the first valid output is the later of the
    * two lookbacks. For every MA type whose lookback is at least
    * optInTimePeriod-1 this equals the MA lookback (unchanged); it rises above it
    * only when the MA lookback is the smaller one - TA_MAType_MAMA at
    * optInTimePeriod >= 34 (constant MAMA lookback 32) and TA_MAType_DISABLED
    * (MA lookback 0, issue #93). Reporting the honest value keeps
    * outBegIdx == lookback: issue #99 kept it at the MA lookback, which
    * under-reported those cases (outBegIdx > lookback) and broke streaming, whose
    * Open is tied to lookback+1. The middle band still begins at the MA's earlier
    * begIdx internally and is realigned to this later bar in bbands() below.
    */
   maLookback = TA_MA_Lookback(optInTimePeriod,optInMAType);
   stddevLookback = TA_STDDEV_Lookback(optInTimePeriod,1.0);
   return (maLookback > stddevLookback) ? maLookback : stddevLookback;
}
