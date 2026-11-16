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

TA_LIB_API TA_RetCode TA_SQZMOM( int    startIdx,
                                 int    endIdx,
                                 const double inHigh[],
                                 const double inLow[],
                                 const double inClose[],
                                 int optInBBPeriod,
                                 double optInNbDev,
                                 int optInKCPeriod,
                                 double optInFactorWide,
                                 double optInFactorNormal,
                                 double optInFactorNarrow,
                                 int          *outBegIdx,
                                 int          *outNBElement,
                                 double        outMomentum[],
                                 int        outSqueeze[] )
{
   TA_RetCode retCode;
   int lookbackTotal;
   int stageLookback;
   int stageIdx;
   int stageLen;
   int i;
   int n;
   int tempBegIdx;
   int tempNbElement;
   int kcmNb;
   int sdNb;
   int kcmShift;
   double *tempDev;
   double *tempWork;
   double *tempKCM;
   double *tempMid;
   double *tempSD;
   double dev;
   double up;
   double lo;
   double centre;
   double band;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInBBPeriod == TA_INTEGER_DEFAULT )
      optInBBPeriod = 20;
   else if( (int)optInBBPeriod < 2 || (int)optInBBPeriod > 100000 )
      return TA_BAD_PARAM;
   if( optInNbDev == TA_REAL_DEFAULT )
      optInNbDev = 2;
   else if( !(optInNbDev >= 0e0 && optInNbDev <= TA_REAL_MAX) )
      return TA_BAD_PARAM;
   if( (int)optInKCPeriod == TA_INTEGER_DEFAULT )
      optInKCPeriod = 20;
   else if( (int)optInKCPeriod < 2 || (int)optInKCPeriod > 100000 )
      return TA_BAD_PARAM;
   if( optInFactorWide == TA_REAL_DEFAULT )
      optInFactorWide = 2;
   else if( !(optInFactorWide >= 0e0 && optInFactorWide <= TA_REAL_MAX) )
      return TA_BAD_PARAM;
   if( optInFactorNormal == TA_REAL_DEFAULT )
      optInFactorNormal = 1.5;
   else if( !(optInFactorNormal >= 0e0 && optInFactorNormal <= TA_REAL_MAX) )
      return TA_BAD_PARAM;
   if( optInFactorNarrow == TA_REAL_DEFAULT )
      optInFactorNarrow = 1;
   else if( !(optInFactorNarrow >= 0e0 && optInFactorNarrow <= TA_REAL_MAX) )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !inClose )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outMomentum )
      return TA_BAD_PARAM;
   if( !outSqueeze )
      return TA_BAD_PARAM;
   if( (const void *)outMomentum == (const void *)outSqueeze )
      return TA_BAD_PARAM;

   /* Each rolling statistic is a call to the shipped function, so it carries
    * that function's own arithmetic bit for bit:
    *
    *   BBU, BBL = sma(close, b) +/- nbdev * stddev(close, b, 1.0)
    *   KCM      = sma(close, k)
    *   BAND     = sma(trange(high,low,close), k)      the simple mean of TR
    *   DMID     = midprice(high, low, k)
    *   DEV      = close - (DMID + KCM)/2
    *   momentum = linearreg(DEV, k)
    *
    * A composed sub-call cannot be handed an MA-type constant, so the bands
    * are the sma + stddev pair and not a bbands call.
    *
    * Two chains need their input one window earlier than the first output bar:
    * sma(k) over TR, and linearreg(k) over DEV. Both windows are k-1 long, so
    * one staged range serves both.
    */
   /* A channel wider than the one it should nest in breaks the ladder: a bar
    * could then be inside "narrow" and outside "normal". Ties are allowed, and
    * wide == normal is the classic on/off/none squeeze.
    */
   if( optInFactorNormal > optInFactorWide || optInFactorNarrow > optInFactorNormal )
   {
      return TA_BAD_PARAM;
   }
   lookbackTotal = TA_SQZMOM_Lookback(optInBBPeriod,optInNbDev,optInKCPeriod,optInFactorWide,optInFactorNormal,optInFactorNarrow);
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
   if( startIdx > endIdx )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_SUCCESS;
   }
   n = endIdx - startIdx + 1;
   stageLookback = TA_LINEARREG_Lookback(optInKCPeriod);
   stageIdx = startIdx - stageLookback;
   stageLen = n + stageLookback;
   tempDev = malloc(stageLen * sizeof(double));
   if( !tempDev )
   {
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_ALLOC_ERR;
   }
   tempWork = malloc(stageLen * sizeof(double));
   if( !tempWork )
   {
      free(tempDev);
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_ALLOC_ERR;
   }
   tempKCM = malloc(stageLen * sizeof(double));
   if( !tempKCM )
   {
      free(tempDev);
      free(tempWork);
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_ALLOC_ERR;
   }
   tempMid = malloc(n * sizeof(double));
   if( !tempMid )
   {
      free(tempDev);
      free(tempWork);
      free(tempKCM);
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_ALLOC_ERR;
   }
   tempSD = malloc(n * sizeof(double));
   if( !tempSD )
   {
      free(tempDev);
      free(tempWork);
      free(tempKCM);
      free(tempMid);
      *outBegIdx= 0;
      *outNBElement= 0;
      return TA_ALLOC_ERR;
   }
   /* The momentum anchor: the Donchian midpoint and the Keltner centre, averaged.
    * midprice is bit-identical to donchian's middle band.
    */
   retCode = TA_MIDPRICE(stageIdx,endIdx,inHigh,inLow,optInKCPeriod,outBegIdx,outNBElement,tempDev);
   if( retCode != TA_SUCCESS )
   {
      free(tempDev);
      free(tempWork);
      free(tempKCM);
      free(tempMid);
      free(tempSD);
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   retCode = TA_SMA(stageIdx,endIdx,inClose,optInKCPeriod,&tempBegIdx,&kcmNb,tempKCM);
   if( retCode != TA_SUCCESS )
   {
      free(tempDev);
      free(tempWork);
      free(tempKCM);
      free(tempMid);
      free(tempSD);
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   /* DEV, over the staged range. The close is read at the index midprice
    * reported for the range, which is what lets this loop stream: keep
    * outBegIdx as midprice's receiver, and reuse it for no call above here.
    */
   for( i = 0; i < stageLen; i += 1 )
   {
      tempDev[i] = inClose[*outBegIdx + i] - (tempDev[i] + tempKCM[i]) / 2.0;
   }
   /* The compression band: the simple mean of the true range, not an ATR.
    * trange's own lookback is 1, and stageIdx is at least 1 because the
    * compression chain alone puts lookbackTotal at k or more.
    */
   retCode = TA_TRANGE(stageIdx,endIdx,inHigh,inLow,inClose,&tempBegIdx,&tempNbElement,tempWork);
   if( retCode != TA_SUCCESS )
   {
      free(tempDev);
      free(tempWork);
      free(tempKCM);
      free(tempMid);
      free(tempSD);
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   retCode = TA_SMA(0,stageLen - 1,tempWork,optInKCPeriod,&tempBegIdx,&tempNbElement,tempWork);
   if( retCode != TA_SUCCESS )
   {
      free(tempDev);
      free(tempWork);
      free(tempKCM);
      free(tempMid);
      free(tempSD);
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   retCode = TA_SMA(startIdx,endIdx,inClose,optInBBPeriod,&tempBegIdx,&tempNbElement,tempMid);
   if( retCode != TA_SUCCESS )
   {
      free(tempDev);
      free(tempWork);
      free(tempKCM);
      free(tempMid);
      free(tempSD);
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   retCode = TA_STDDEV(startIdx,endIdx,inClose,optInBBPeriod,1.0,&tempBegIdx,&sdNb,tempSD);
   if( retCode != TA_SUCCESS )
   {
      free(tempDev);
      free(tempWork);
      free(tempKCM);
      free(tempMid);
      free(tempSD);
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   /* The centre is staged one regression window early; the shift puts it on
    * the bar the bands are on.
    */
   kcmShift = kcmNb - sdNb;
   /* LAST of the calls, because it is the first to write a caller buffer. Every
    * read of high, low and close is above it, so an output aliased onto one of
    * them -- which the abstract layer tests on purpose -- is still intact when it
    * is read.
    *
    * The regression window closes on the first output bar, so entering it at 0 of
    * the staged range puts its first fitted value on startIdx.
    */
   retCode = TA_LINEARREG(0,stageLen - 1,tempDev,optInKCPeriod,&tempBegIdx,&tempNbElement,outMomentum);
   if( retCode != TA_SUCCESS )
   {
      free(tempDev);
      free(tempWork);
      free(tempKCM);
      free(tempMid);
      free(tempSD);
      *outBegIdx= 0;
      *outNBElement= 0;
      return retCode;
   }
   /* The ordinal. Every comparison is strict: a band exactly on a channel
    * edge is not inside that channel and not outside it either.
    *
    * Keep the deviation a rounded product added to and subtracted from the
    * middle band, as bbands does. Written as one expression per band, the
    * upper one contracts into a fused multiply-add and the lower one cannot,
    * and a band within one ulp of an edge then changes level.
    */
   for( i = 0; i < n; i += 1 )
   {
      dev = tempSD[i] * optInNbDev;
      up = tempMid[i] + dev;
      lo = tempMid[i] - dev;
      centre = tempKCM[i + kcmShift];
      band = tempWork[i];
      if( lo > centre - optInFactorNarrow * band && up < centre + optInFactorNarrow * band )
      {
         outSqueeze[i] = 3;
      } else if( lo > centre - optInFactorNormal * band && up < centre + optInFactorNormal * band )
      {
         outSqueeze[i] = 2;
      } else if( lo > centre - optInFactorWide * band && up < centre + optInFactorWide * band )
      {
         outSqueeze[i] = 1;
      } else if( lo < centre - optInFactorWide * band && up > centre + optInFactorWide * band )
      {
         outSqueeze[i] = -1;
      } else 
      {
         outSqueeze[i] = 0;
      }
   }
   free(tempDev);
   free(tempWork);
   free(tempKCM);
   free(tempMid);
   free(tempSD);
   *outBegIdx= startIdx;
   *outNBElement= n;
   return TA_SUCCESS;
}

TA_LIB_API int TA_SQZMOM_Lookback( int optInBBPeriod, double optInNbDev, int optInKCPeriod, double optInFactorWide, double optInFactorNormal, double optInFactorNarrow )
{
   int lookbackTotal;
   int leg;
   if( (int)optInBBPeriod == TA_INTEGER_DEFAULT )
      optInBBPeriod = 20;
   else if( (int)optInBBPeriod < 2 || (int)optInBBPeriod > 100000 )
      return -1;
   if( optInNbDev == TA_REAL_DEFAULT )
      optInNbDev = 2;
   else if( !(optInNbDev >= 0e0 && optInNbDev <= TA_REAL_MAX) )
      return -1;
   if( (int)optInKCPeriod == TA_INTEGER_DEFAULT )
      optInKCPeriod = 20;
   else if( (int)optInKCPeriod < 2 || (int)optInKCPeriod > 100000 )
      return -1;
   if( optInFactorWide == TA_REAL_DEFAULT )
      optInFactorWide = 2;
   else if( !(optInFactorWide >= 0e0 && optInFactorWide <= TA_REAL_MAX) )
      return -1;
   if( optInFactorNormal == TA_REAL_DEFAULT )
      optInFactorNormal = 1.5;
   else if( !(optInFactorNormal >= 0e0 && optInFactorNormal <= TA_REAL_MAX) )
      return -1;
   if( optInFactorNarrow == TA_REAL_DEFAULT )
      optInFactorNarrow = 1;
   else if( !(optInFactorNarrow >= 0e0 && optInFactorNarrow <= TA_REAL_MAX) )
      return -1;
   /* The range check cannot order the three factors; without this the lookback
    * answers a usable number for a call that cannot run.
    */
   if( optInFactorNormal > optInFactorWide || optInFactorNarrow > optInFactorNormal )
   {
      return -1;
   }
   /* Three chains reach the first bar at which both outputs exist, and the
    * first output is the longest of them:
    *
    *   bands        sma(b) / stddev(b)           = b-1
    *   compression  trange -> sma(k)             = 1 + (k-1) = k
    *   momentum     midprice(k) -> linearreg(k)  = (k-1) + (k-1) = 2(k-1)
    *
    * The Keltner centre, sma(k) on the close, is k-1 and never dominates the
    * momentum chain it also feeds. At b = k = 20 the three read 19, 20 and 38,
    * which is where pandas, ta4j and trading-signals put their first value.
    * The factors scale a band that already exists, so they do not enter.
    */
   lookbackTotal = TA_SMA_Lookback(optInBBPeriod);
   leg = TA_STDDEV_Lookback(optInBBPeriod,1.0);
   if( leg > lookbackTotal )
   {
      lookbackTotal = leg;
   }
   leg = TA_TRANGE_Lookback() + TA_SMA_Lookback(optInKCPeriod);
   if( leg > lookbackTotal )
   {
      lookbackTotal = leg;
   }
   leg = TA_MIDPRICE_Lookback(optInKCPeriod) + TA_LINEARREG_Lookback(optInKCPeriod);
   if( leg > lookbackTotal )
   {
      lookbackTotal = leg;
   }
   return lookbackTotal;
}
