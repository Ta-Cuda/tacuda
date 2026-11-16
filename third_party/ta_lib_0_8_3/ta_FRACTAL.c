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

TA_LIB_API TA_RetCode TA_FRACTAL( int    startIdx,
                                  int    endIdx,
                                  const double inHigh[],
                                  const double inLow[],
                                  int optInLeftBars,
                                  int optInRightBars,
                                  int          *outBegIdx,
                                  int          *outNBElement,
                                  int        outSwingHigh[],
                                  int        outSwingLow[] )
{
   int today;
   int outIdx;
   int lookbackTotal;
   int i;
   double pivotHigh;
   double pivotLow;
   double otherHigh;
   double otherLow;
   double tempHigh;
   double tempLow;

   if( (startIdx < 0) || (startIdx > TA_INDEX_MAX) )
      return TA_OUT_OF_RANGE_START_INDEX;
   if( (endIdx < 0) || (endIdx > TA_INDEX_MAX) || (endIdx < startIdx) )
      return TA_OUT_OF_RANGE_END_INDEX;

   if( (int)optInLeftBars == TA_INTEGER_DEFAULT )
      optInLeftBars = 2;
   else if( (int)optInLeftBars < 1 || (int)optInLeftBars > 100000 )
      return TA_BAD_PARAM;
   if( (int)optInRightBars == TA_INTEGER_DEFAULT )
      optInRightBars = 2;
   else if( (int)optInRightBars < 1 || (int)optInRightBars > 100000 )
      return TA_BAD_PARAM;
   if( !inHigh )
      return TA_BAD_PARAM;
   if( !inLow )
      return TA_BAD_PARAM;
   if( !outBegIdx || !outNBElement )
      return TA_BAD_PARAM;
   if( !outSwingHigh )
      return TA_BAD_PARAM;
   if( !outSwingLow )
      return TA_BAD_PARAM;
   if( outSwingHigh == outSwingLow )
      return TA_BAD_PARAM;

   *outBegIdx= 0;
   *outNBElement= 0;
   lookbackTotal = TA_FRACTAL_Lookback(optInLeftBars,optInRightBars);
   if( startIdx < lookbackTotal )
   {
      startIdx = lookbackTotal;
   }
   /* Make sure there is still something to evaluate. */
   if( startIdx > endIdx )
   {
      return TA_SUCCESS;
   }
   outIdx = 0;
   today = startIdx;
   while( today <= endIdx )
   {
      pivotHigh = 0.0;
      pivotLow = 0.0;
      otherHigh = 0.0;
      otherLow = 0.0;
      for( i = optInLeftBars + optInRightBars; i >= 0; i -= 1 )
      {
         tempHigh = inHigh[today - i];
         tempLow = inLow[today - i];
         if( i == optInRightBars )
         {
            pivotHigh = tempHigh;
            pivotLow = tempLow;
         } else if( i == optInLeftBars + optInRightBars )
         {
            /* optInLeftBars is at least 1, so the oldest bar of the window is
             * never the pivot and always seeds the rest-of-window extrema.
             */
            otherHigh = tempHigh;
            otherLow = tempLow;
         } else 
         {
            if( tempHigh > otherHigh )
            {
               otherHigh = tempHigh;
            }
            if( tempLow < otherLow )
            {
               otherLow = tempLow;
            }
         }
      }
      if( pivotHigh > otherHigh )
      {
         outSwingHigh[outIdx] = 100;
      } else 
      {
         outSwingHigh[outIdx] = 0;
      }
      if( pivotLow < otherLow )
      {
         outSwingLow[outIdx] = 100;
      } else 
      {
         outSwingLow[outIdx] = 0;
      }
      outIdx += 1;
      today += 1;
   }
   *outBegIdx= startIdx;
   *outNBElement= outIdx;
   return TA_SUCCESS;
}

TA_LIB_API int TA_FRACTAL_Lookback( int optInLeftBars, int optInRightBars )
{
   if( (int)optInLeftBars == TA_INTEGER_DEFAULT )
      optInLeftBars = 2;
   else if( (int)optInLeftBars < 1 || (int)optInLeftBars > 100000 )
      return -1;
   if( (int)optInRightBars == TA_INTEGER_DEFAULT )
      optInRightBars = 2;
   else if( (int)optInRightBars < 1 || (int)optInRightBars > 100000 )
      return -1;
   return optInLeftBars + optInRightBars;
}
