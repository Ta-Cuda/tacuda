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
TA_LIB_API TA_RetCode TA_PVO( int    startIdx,
                              int    endIdx,
                              const double inVolume[],
                              int optInFastPeriod,
                              int optInSlowPeriod,
                              TA_MAType optInMAType,
                              int          *outBegIdx,
                              int          *outNBElement,
                              double        outReal[] )
{
   double *tempBuffer;
   TA_RetCode retCode;
   double tempReal;
   int tempInteger;
   int fastBeg;
   int fastNb;
   int offset;
   int slowLookback;
   int windowed;
   int zeroRun;
   int i;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInFastPeriod == TA_INTEGER_DEFAULT )
      optInFastPeriod = 12;
   else if( (int)optInFastPeriod < 2 || (int)optInFastPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInSlowPeriod == TA_INTEGER_DEFAULT )
      optInSlowPeriod = 26;
   else if( (int)optInSlowPeriod < 2 || (int)optInSlowPeriod > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInMAType == TA_INTEGER_DEFAULT || optInMAType == TA_MAType_DEFAULT )
      optInMAType = 1;
   else if( (int)optInMAType < TA_MATYPE_MIN || (int)optInMAType > TA_MATYPE_MAX )
      return TA_BAD_PARAM;
   if( !inVolume )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outReal )
      return TA_BAD_PARAM;

   /* Nothing to produce: the range ends before the lookback. Return before
    * touching anything.
    *
    * Without this the fast MA below runs first, and its lookback is SMALLER
    * than pvo's own — so it reads the whole range and computes a result the
    * empty slow MA then discards. Observably identical (the slow MA's own early
    * return already yields 0,0 here), but it is the difference between "a range
    * that ends before the lookback reads nothing" being true of this function and
    * being false: with a caller-supplied inVolume that stops short of endIdx, that
    * discarded work is an out-of-bounds read. Pinned by the zero-length no-I/O
    * probe over every guarded core.
    */
   if( TA_MA_Lookback(max(optInSlowPeriod,optInFastPeriod),optInMAType) > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   if( optInMAType == TA_MAType_SMA )
   {
      /* SMA fast path: the fast window is the newest optInFastPeriod bars of the
       * slow one, so ONE pass over the input serves both moving averages - two
       * running sums, no intermediate buffer and no allocation, where the general
       * path below makes two passes and allocates the fast MA in full.
       *
       * Bit-identical to that path. Each sum sees exactly the add/subtract
       * sequence TA_SMA gives it at its own period, starting from its own first
       * output bar - which is why the fast sum is walked alone over the bars the
       * slow MA does not reach (its running total is path-dependent, so arriving
       * at the first output bar by a shorter route would change the low bits) -
       * and each quotient is formed as sma.c forms it: the total AFTER adding the
       * new bar and BEFORE dropping the trailing one, divided by the period.
       *
       * SMA is one of the windowed types, so the dead-window rule of #454 applies
       * here too: _zeroRun is the same counter the general path keeps, warmed over
       * the same bars (which are the bars the slow sum is seeded from) and held at
       * the slow lookback once the window is dead.
       *
       * inVolume may alias outReal, as it may in the general path. outReal[_outIdx]
       * is written at bar _i with _outIdx <= _i-optInSlowPeriod+1 <= both trailing
       * indices, and every read of bar _i happens before that write, so no bar is
       * overwritten before its last read.
       *
       * Every read is inside [0, endIdx]: the guard above leaves the slow
       * lookback no greater than endIdx, and the public tier rejects
       * endIdx < startIdx, so _slowStart <= endIdx and the seeding loops stop
       * one bar below it. There is nothing left for an empty-output arm to
       * catch, which is why this path has none.
       */
      double _fastTotal;
      double _slowTotal;
      double _fastValue;
      double _slowValue;
      double _slowMA;
      int _i;
      int _j;
      int _outIdx;
      int _fastStart;
      int _slowStart;
      int _fastTrailing;
      int _slowTrailing;
      int _slowLookback;
      int _zeroRun;
      /* Make sure slow is really slower than the fast period! if not, swap... */
      if( optInSlowPeriod < optInFastPeriod )
      {
         tempInteger = optInSlowPeriod;
         optInSlowPeriod = optInFastPeriod;
         optInFastPeriod = tempInteger;
      }
      _fastStart = optInFastPeriod - 1;
      if( _fastStart < startIdx )
      {
         _fastStart = startIdx;
      }
      _slowStart = optInSlowPeriod - 1;
      if( _slowStart < startIdx )
      {
         _slowStart = startIdx;
      }
      _fastTrailing = _fastStart - (optInFastPeriod - 1);
      _fastTotal = 0.0;
      for( _j = _fastTrailing; _j < _fastStart; _j += 1 )
      {
         _fastTotal += inVolume[_j];
      }
      /* One loop seeds the slow sum and warms the dead-window counter: the bars
       * it walks, [_slowStart-_slowLookback, _slowStart), are exactly the ones
       * the general path warms _zeroRun over.
       */
      _slowLookback = optInSlowPeriod - 1;
      _zeroRun = 0;
      _slowTrailing = _slowStart - _slowLookback;
      _slowTotal = 0.0;
      for( _j = _slowTrailing; _j < _slowStart; _j += 1 )
      {
         _slowTotal += inVolume[_j];
         _zeroRun = (fabs(inVolume[_j]) <= 0.0) ? _zeroRun + 1 : 0;
      }
      /* The bars the fast MA has and the slow one does not: advance the fast sum
       * alone. No output, but the sum must arrive at _slowStart along the same
       * path TA_SMA would have taken.
       */
      for( _i = _fastStart; _i < _slowStart; _i += 1 )
      {
         _fastTotal += inVolume[_i];
         _fastTotal -= inVolume[_fastTrailing];
         _fastTrailing += 1;
      }
      _outIdx = 0;
      for( _i = _slowStart; _i <= endIdx; _i += 1 )
      {
         _zeroRun = (fabs(inVolume[_i]) <= 0.0) ? _zeroRun + 1 : 0;
         _fastTotal += inVolume[_i];
         _fastValue = _fastTotal;
         _fastTotal -= inVolume[_fastTrailing];
         _fastTrailing += 1;
         _slowTotal += inVolume[_i];
         _slowValue = _slowTotal;
         _slowTotal -= inVolume[_slowTrailing];
         _slowTrailing += 1;
         _slowMA = _slowValue / (double)optInSlowPeriod;
         if( _zeroRun > _slowLookback )
         {
            _zeroRun = _slowLookback;
            outReal[_outIdx] = 0.0;
         } else if( !TA_IS_ZERO(_slowMA) )
         {
            outReal[_outIdx] = (_fastValue / (double)optInFastPeriod - _slowMA) / _slowMA * 100.0;
         } else 
         {
            outReal[_outIdx] = 0.0;
         }
         _outIdx += 1;
      }
      *outBegIdx= _slowStart;
      *outNBElement= _outIdx;
      return TA_SUCCESS;
   }
   if( optInMAType == TA_MAType_EMA )
   {
      /* EMA fast path: both recursions in one loop, no buffer. Bit-identical to
       * the general path only while each EMA is seeded at its OWN lookback and
       * keeps ema.c's recursion spelling: the fast EMA starts earlier than the
       * slow one, and a shared seed bar would change every output.
       */
      double _eFastK;
      double _eSlowK;
      double _eFast;
      double _eSlow;
      double _eX;
      double _eFastBeta;
      double _eSlowBeta;
      int _eN;
      int _eToday;
      int _eFastToday;
      int _eSlowToday;
      int _eSlowStart;
      int _eOutIdx;
      if( optInSlowPeriod < optInFastPeriod )
      {
         tempInteger = optInSlowPeriod;
         optInSlowPeriod = optInFastPeriod;
         optInFastPeriod = tempInteger;
      }
      _eFastBeta = (double)(optInFastPeriod - 1) / (double)(optInFastPeriod + 1);
      _eFastK = 1.0 - _eFastBeta;
      if( _eFastBeta < 0.5 )
      {
         _eFastBeta = 1.0 - _eFastK;
      }
      _eSlowBeta = (double)(optInSlowPeriod - 1) / (double)(optInSlowPeriod + 1);
      _eSlowK = 1.0 - _eSlowBeta;
      if( _eSlowBeta < 0.5 )
      {
         _eSlowBeta = 1.0 - _eSlowK;
      }
      _eFastToday = TA_EMA_Lookback(optInFastPeriod);
      if( _eFastToday < startIdx )
      {
         _eFastToday = startIdx;
      }
      _eFastToday -= TA_EMA_Lookback(optInFastPeriod);
      _eSlowStart = TA_EMA_Lookback(optInSlowPeriod);
      if( _eSlowStart < startIdx )
      {
         _eSlowStart = startIdx;
      }
      _eSlowToday = _eSlowStart - TA_EMA_Lookback(optInSlowPeriod);
      _eFast = 0.0;
      for( _eN = 0; _eN < optInFastPeriod; _eN += 1 )
      {
         _eFast += inVolume[_eFastToday++];
      }
      _eFast = _eFast / optInFastPeriod;
      while( _eFastToday <= _eSlowStart )
      {
         _eFast = fma(_eFastBeta, _eFast, _eFastK * inVolume[_eFastToday++]);
      }
      _eSlow = 0.0;
      for( _eN = 0; _eN < optInSlowPeriod; _eN += 1 )
      {
         _eSlow += inVolume[_eSlowToday++];
      }
      _eSlow = _eSlow / optInSlowPeriod;
      while( _eSlowToday <= _eSlowStart )
      {
         _eSlow = fma(_eSlowBeta, _eSlow, _eSlowK * inVolume[_eSlowToday++]);
      }
      _eOutIdx = 0;
      if( !TA_IS_ZERO(_eSlow) )
      {
         outReal[_eOutIdx] = (_eFast - _eSlow) / _eSlow * 100.0;
      } else 
      {
         outReal[_eOutIdx] = 0.0;
      }
      _eOutIdx += 1;
      _eToday = _eSlowStart + 1;
      while( _eToday <= endIdx )
      {
         _eX = inVolume[_eToday++];
         _eFast = fma(_eFastBeta, _eFast, _eFastK * _eX);
         _eSlow = fma(_eSlowBeta, _eSlow, _eSlowK * _eX);
         if( !TA_IS_ZERO(_eSlow) )
         {
            outReal[_eOutIdx] = (_eFast - _eSlow) / _eSlow * 100.0;
         } else 
         {
            outReal[_eOutIdx] = 0.0;
         }
         _eOutIdx += 1;
      }
      *outBegIdx= _eSlowStart;
      *outNBElement= _eOutIdx;
      return TA_SUCCESS;
   }
   /* Allocate an intermediate buffer. */
   tempBuffer = malloc((endIdx - startIdx + 1) * sizeof(double));
   if( !tempBuffer )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_ALLOC_ERR;
   }
   /* Make sure slow is really slower than
    * the fast period! if not, swap...
    */
   if( optInSlowPeriod < optInFastPeriod )
   {
      /* swap */
      tempInteger = optInSlowPeriod;
      optInSlowPeriod = optInFastPeriod;
      optInFastPeriod = tempInteger;
   }
   /* Calculate the fast MA into the tempBuffer. */
   retCode = TA_MA(startIdx,endIdx,inVolume,optInFastPeriod,optInMAType,&fastBeg,&fastNb,tempBuffer);
   if( retCode != TA_SUCCESS )
   {
      free(tempBuffer);
      return retCode;
   }
   /* Calculate the slow MA into the output. */
   retCode = TA_MA(startIdx,endIdx,inVolume,optInSlowPeriod,optInMAType,outBegIdx,outNBElement,outReal);
   if( retCode != TA_SUCCESS )
   {
      free(tempBuffer);
      return retCode;
   }
   /* fastNb - *outNBElement == slowBeg - fastBeg (the fast MA has at least as
    * many outputs), so tempBuffer[i+offset] is the fast MA at the same bar as
    * outReal[i], with a non-negative index. An empty slow MA skips the loop.
    */
   offset = fastNb - *outNBElement;
   /* A windowed slow MA (SMA, WMA, TRIMA, HMA) over bars that are all exactly
    * zero is exactly zero, but its running sums leave residue there that
    * TA_IS_ZERO does not catch, and residue over residue is noise where 0 is
    * documented. zeroRun counts the trailing zero bars, held at slowLookback once
    * the window is dead. The recursive MA types really are nonzero on such a
    * window, so they keep the plain loop.
    */
   slowLookback = TA_MA_Lookback(optInSlowPeriod,optInMAType);
   windowed = (optInMAType == TA_MAType_SMA || optInMAType == TA_MAType_WMA || optInMAType == TA_MAType_TRIMA || optInMAType == TA_MAType_HMA) ? 1 : 0;
   zeroRun = 0;
   for( i = *outBegIdx - slowLookback; i < *outBegIdx; i += 1 )
   {
      zeroRun = (fabs(inVolume[i]) <= 0.0) ? zeroRun + 1 : 0;
   }
   if( windowed != 0 )
   {
      for( i = 0; i < (int)*outNBElement; i += 1 )
      {
         zeroRun = (fabs(inVolume[*outBegIdx + i]) <= 0.0) ? zeroRun + 1 : 0;
         tempReal = outReal[i];
         if( zeroRun > slowLookback )
         {
            zeroRun = slowLookback;
            outReal[i] = 0.0;
         } else if( !TA_IS_ZERO(tempReal) )
         {
            outReal[i] = (tempBuffer[i + offset] - tempReal) / tempReal * 100.0;
         } else 
         {
            outReal[i] = 0.0;
         }
      }
   } else 
   {
      /* Calculate ((fast MA)-(slow MA))/(slow MA) in the output. */
      for( i = 0; i < (int)*outNBElement; i += 1 )
      {
         tempReal = outReal[i];
         if( !TA_IS_ZERO(tempReal) )
         {
            outReal[i] = (tempBuffer[i + offset] - tempReal) / tempReal * 100.0;
         } else 
         {
            outReal[i] = 0.0;
         }
      }
   }
   free(tempBuffer);
   return TA_SUCCESS;
}

TA_LIB_API int TA_PVO_Lookback( int optInFastPeriod, int optInSlowPeriod, TA_MAType optInMAType )
{
   if( (int)optInFastPeriod == TA_INTEGER_DEFAULT )
      optInFastPeriod = 12;
   else if( (int)optInFastPeriod < 2 || (int)optInFastPeriod > 100000 )
      return -1;
   if( (int)optInSlowPeriod == TA_INTEGER_DEFAULT )
      optInSlowPeriod = 26;
   else if( (int)optInSlowPeriod < 2 || (int)optInSlowPeriod > 100000 )
      return -1;
   if( (int)optInMAType == TA_INTEGER_DEFAULT || optInMAType == TA_MAType_DEFAULT )
      optInMAType = 1;
   else if( (int)optInMAType < TA_MATYPE_MIN || (int)optInMAType > TA_MATYPE_MAX )
      return -1;
   /* Lookback is driven by the slowest MA. */
   return TA_MA_Lookback(max(optInSlowPeriod,optInFastPeriod),optInMAType);
}
