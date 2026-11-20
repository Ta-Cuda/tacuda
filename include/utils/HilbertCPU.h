/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef HILBERT_CPU_H
#define HILBERT_CPU_H

#include <cmath>
#include <limits>

// The host API runs the same state machine as the HT_* kernels. Sharing the
// core keeps the TA-Lib operation order on both sides and makes the host read
// the build-time sin/cos weights of the kernels instead of calling sin/cos at
// runtime, whose result depends on the math library's FMA dispatch.
#include <utils/HilbertCore.h>

namespace tacuda {
namespace hilbert_cpu {

constexpr double DEG2RAD = 0.017453292519943295;  // PI / 180

using tacuda::hilbert::HilbertState;
using tacuda::hilbert::WMAState;
using tacuda::hilbert::initWMA;
using tacuda::hilbert::doWMAStep;
using tacuda::hilbert::processHilbertStep;
using tacuda::hilbert::updatePeriod;
using tacuda::hilbert::updateDCPhase;

// ============================================================================
// CPU implementations of HT indicators
// ============================================================================

inline void ht_dcperiod_cpu(const float* input, float* output, int size) {
    const int lookbackTotal = 32;
    for (int i = 0; i < size; ++i) 
        output[i] = std::numeric_limits<float>::quiet_NaN();
    if (size <= lookbackTotal) return;
    
    HilbertState hs; hs.init();
    WMAState wma; wma.init();
    
    int startIdx = lookbackTotal;
    wma.trailingWMAIdx = 0;
    int today = 0;
    
    initWMA(wma, input, today);
    
    double smoothedValue;
    for (int i = 0; i < 9; ++i)
        smoothedValue = doWMAStep(wma, input, static_cast<double>(input[today++]));
    
    while (today < size) {
        smoothedValue = doWMAStep(wma, input, static_cast<double>(input[today]));
        double outI1, outQ1;
        processHilbertStep(hs, smoothedValue, today, outI1, outQ1);
        updatePeriod(hs);
        
        if (today >= startIdx)
            output[today] = static_cast<float>(hs.smoothPeriod);
        today++;
    }
}

inline void ht_dcphase_cpu(const float* input, float* output, int size) {
    const int lookbackTotal = 63;
    for (int i = 0; i < size; ++i)
        output[i] = std::numeric_limits<float>::quiet_NaN();
    if (size <= lookbackTotal) return;
    
    HilbertState hs; hs.init();
    WMAState wma; wma.init();
    double smoothPrice[50] = {0.0};
    int smoothPrice_Idx = 0;
    
    int startIdx = lookbackTotal;
    wma.trailingWMAIdx = 0;
    int today = 0;
    
    initWMA(wma, input, today);
    
    double smoothedValue;
    for (int i = 0; i < 34; ++i)
        smoothedValue = doWMAStep(wma, input, static_cast<double>(input[today++]));
    
    // The DC phase is stateful and updated on EVERY bar (including warm-in
    // bars before the first output), like TA-Lib.
    double DCPhase = 0.0;
    while (today < size) {
        smoothedValue = doWMAStep(wma, input, static_cast<double>(input[today]));
        smoothPrice[smoothPrice_Idx] = smoothedValue;
        
        double outI1, outQ1;
        processHilbertStep(hs, smoothedValue, today, outI1, outQ1);
        updatePeriod(hs);
        updateDCPhase(smoothPrice, smoothPrice_Idx, hs.smoothPeriod, DCPhase);
        
        if (today >= startIdx) {
            output[today] = static_cast<float>(DCPhase);
        }
        
        if (++smoothPrice_Idx == 50) smoothPrice_Idx = 0;
        today++;
    }
}

inline void ht_phasor_cpu(const float* input, float* outInPhase, float* outQuadrature, int size) {
    const int lookbackTotal = 32;
    for (int i = 0; i < size; ++i) {
        outInPhase[i] = std::numeric_limits<float>::quiet_NaN();
        outQuadrature[i] = std::numeric_limits<float>::quiet_NaN();
    }
    if (size <= lookbackTotal) return;
    
    HilbertState hs; hs.init();
    WMAState wma; wma.init();
    
    int startIdx = lookbackTotal;
    wma.trailingWMAIdx = 0;
    int today = 0;
    
    initWMA(wma, input, today);
    
    double smoothedValue;
    for (int i = 0; i < 9; ++i)
        smoothedValue = doWMAStep(wma, input, static_cast<double>(input[today++]));
    
    while (today < size) {
        smoothedValue = doWMAStep(wma, input, static_cast<double>(input[today]));
        double outI1, outQ1;
        processHilbertStep(hs, smoothedValue, today, outI1, outQ1);
        updatePeriod(hs);
        
        if (today >= startIdx) {
            outInPhase[today] = static_cast<float>(outI1);
            outQuadrature[today] = static_cast<float>(outQ1);
        }
        today++;
    }
}

inline void ht_sine_cpu(const float* input, float* outSine, float* outLeadSine, int size) {
    const int lookbackTotal = 63;
    for (int i = 0; i < size; ++i) {
        outSine[i] = std::numeric_limits<float>::quiet_NaN();
        outLeadSine[i] = std::numeric_limits<float>::quiet_NaN();
    }
    if (size <= lookbackTotal) return;
    
    HilbertState hs; hs.init();
    WMAState wma; wma.init();
    double smoothPrice[50] = {0.0};
    int smoothPrice_Idx = 0;
    
    int startIdx = lookbackTotal;
    wma.trailingWMAIdx = 0;
    int today = 0;
    
    initWMA(wma, input, today);
    
    double smoothedValue;
    for (int i = 0; i < 34; ++i)
        smoothedValue = doWMAStep(wma, input, static_cast<double>(input[today++]));
    
    // The DC phase is stateful and updated on EVERY bar (including warm-in
    // bars before the first output), like TA-Lib.
    double DCPhase = 0.0;
    while (today < size) {
        smoothedValue = doWMAStep(wma, input, static_cast<double>(input[today]));
        smoothPrice[smoothPrice_Idx] = smoothedValue;
        
        double outI1, outQ1;
        processHilbertStep(hs, smoothedValue, today, outI1, outQ1);
        updatePeriod(hs);
        updateDCPhase(smoothPrice, smoothPrice_Idx, hs.smoothPeriod, DCPhase);
        
        if (today >= startIdx) {
            outSine[today] = static_cast<float>(std::sin(DCPhase * DEG2RAD));
            outLeadSine[today] = static_cast<float>(std::sin((DCPhase + 45.0) * DEG2RAD));
        }
        
        if (++smoothPrice_Idx == 50) smoothPrice_Idx = 0;
        today++;
    }
}

inline void ht_trendline_cpu(const float* input, float* output, int size) {
    const int lookbackTotal = 63;
    for (int i = 0; i < size; ++i)
        output[i] = std::numeric_limits<float>::quiet_NaN();
    int first = 0;
    while (first < size && std::isnan(input[first])) ++first;
    if (size - first <= lookbackTotal) return;
    input += first;
    output += first;
    size -= first;
    
    HilbertState hs; hs.init();
    WMAState wma; wma.init();
    double iTrend1 = 0.0, iTrend2 = 0.0, iTrend3 = 0.0;
    
    int startIdx = lookbackTotal;
    wma.trailingWMAIdx = 0;
    int today = 0;
    
    initWMA(wma, input, today);
    
    double smoothedValue;
    for (int i = 0; i < 34; ++i)
        smoothedValue = doWMAStep(wma, input, static_cast<double>(input[today++]));
    
    while (today < size) {
        smoothedValue = doWMAStep(wma, input, static_cast<double>(input[today]));
        
        double outI1, outQ1;
        processHilbertStep(hs, smoothedValue, today, outI1, outQ1);
        updatePeriod(hs);
        
        // No clamps: TA-Lib sums DCPeriodInt bars as-is (smoothPeriod ramps
        // slowly enough that idx never underruns).
        double DCPeriod = hs.smoothPeriod + 0.5;
        int DCPeriodInt = static_cast<int>(DCPeriod);
        
        double avgPrice = 0.0;
        int idx = today;
        for (int i = 0; i < DCPeriodInt; ++i)
            avgPrice += static_cast<double>(input[idx--]);
        if (DCPeriodInt > 0)
            avgPrice /= static_cast<double>(DCPeriodInt);
        
        double trendline = (4.0 * avgPrice + 3.0 * iTrend1 + 2.0 * iTrend2 + iTrend3) / 10.0;
        iTrend3 = iTrend2;
        iTrend2 = iTrend1;
        iTrend1 = avgPrice;
        
        if (today >= startIdx)
            output[today] = static_cast<float>(trendline);
        today++;
    }
}

inline void ht_trendmode_cpu(const float* input, float* output, int size) {
    const int lookbackTotal = 63;
    for (int i = 0; i < size; ++i)
        output[i] = 0.0f;
    if (size <= lookbackTotal) return;
    
    HilbertState hs; hs.init();
    WMAState wma; wma.init();
    double smoothPrice[50] = {0.0};
    int smoothPrice_Idx = 0;
    double iTrend1 = 0.0, iTrend2 = 0.0, iTrend3 = 0.0;
    int daysInTrend = 0;
    double DCPhase = 0.0, prevDCPhase = 0.0;
    double sine = 0.0, leadSine = 0.0;
    double prevSine = 0.0, prevLeadSine = 0.0;
    
    int startIdx = lookbackTotal;
    wma.trailingWMAIdx = 0;
    int today = 0;
    
    initWMA(wma, input, today);
    
    double smoothedValue;
    for (int i = 0; i < 34; ++i)
        smoothedValue = doWMAStep(wma, input, static_cast<double>(input[today++]));
    
    // Literal ta_HT_TRENDMODE.c ordering: prev* snapshots taken BEFORE
    // recomputing; the stateful DC phase updates on every bar.
    while (today < size) {
        smoothedValue = doWMAStep(wma, input, static_cast<double>(input[today]));
        smoothPrice[smoothPrice_Idx] = smoothedValue;
        
        double outI1, outQ1;
        processHilbertStep(hs, smoothedValue, today, outI1, outQ1);
        updatePeriod(hs);
        
        prevDCPhase = DCPhase;
        updateDCPhase(smoothPrice, smoothPrice_Idx, hs.smoothPeriod, DCPhase);
        
        prevSine = sine;
        prevLeadSine = leadSine;
        sine = std::sin(DCPhase * DEG2RAD);
        leadSine = std::sin((DCPhase + 45.0) * DEG2RAD);
        
        // No clamps: TA-Lib sums DCPeriodInt bars as-is.
        double DCPeriod = hs.smoothPeriod + 0.5;
        int DCPeriodInt = static_cast<int>(DCPeriod);
        
        double avgPrice = 0.0;
        int idx = today;
        for (int i = 0; i < DCPeriodInt; ++i)
            avgPrice += static_cast<double>(input[idx--]);
        if (DCPeriodInt > 0)
            avgPrice /= static_cast<double>(DCPeriodInt);
        
        double trendline = (4.0 * avgPrice + 3.0 * iTrend1 + 2.0 * iTrend2 + iTrend3) / 10.0;
        iTrend3 = iTrend2;
        iTrend2 = iTrend1;
        iTrend1 = avgPrice;
        
        int trend = 1;
        if (((sine > leadSine) && (prevSine <= prevLeadSine)) ||
            ((sine < leadSine) && (prevSine >= prevLeadSine))) {
            daysInTrend = 0;
            trend = 0;
        }
        daysInTrend++;
        if (daysInTrend < (0.5 * hs.smoothPeriod))
            trend = 0;
        
        double tempReal = DCPhase - prevDCPhase;
        if ((hs.smoothPeriod != 0.0) &&
            (tempReal > (0.67 * 360.0 / hs.smoothPeriod)) &&
            (tempReal < (1.5 * 360.0 / hs.smoothPeriod)))
            trend = 0;
        
        tempReal = smoothPrice[smoothPrice_Idx];
        if ((trendline != 0.0) && (std::fabs((tempReal - trendline) / trendline) >= 0.015))
            trend = 1;
        
        if (today >= startIdx)
            output[today] = static_cast<float>(trend);
        
        if (++smoothPrice_Idx == 50) smoothPrice_Idx = 0;
        today++;
    }
}

} // namespace hilbert_cpu
} // namespace tacuda

#endif // HILBERT_CPU_H

