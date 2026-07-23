/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef HILBERT_CPU_H
#define HILBERT_CPU_H

#include <cmath>
#include <limits>
#include <cstring>

namespace tacuda {
namespace hilbert_cpu {

// Constants
constexpr double HT_A = 0.0962;
constexpr double HT_B = 0.5769;
constexpr double RAD2DEG = 57.29577951308232;  // 180 / PI
constexpr double DEG2RAD = 0.017453292519943295;  // PI / 180
constexpr double PI2 = 6.283185307179586;  // 2*PI

// Hilbert Transform state structure
struct HilbertState {
    double detrender_Odd[3], detrender_Even[3];
    double prev_detrender_Odd, prev_detrender_Even;
    double prev_detrender_input_Odd, prev_detrender_input_Even;
    
    double Q1_Odd[3], Q1_Even[3];
    double prev_Q1_Odd, prev_Q1_Even;
    double prev_Q1_input_Odd, prev_Q1_input_Even;
    
    double jI_Odd[3], jI_Even[3];
    double prev_jI_Odd, prev_jI_Even;
    double prev_jI_input_Odd, prev_jI_input_Even;
    
    double jQ_Odd[3], jQ_Even[3];
    double prev_jQ_Odd, prev_jQ_Even;
    double prev_jQ_input_Odd, prev_jQ_input_Even;
    
    double I1ForOddPrev2, I1ForOddPrev3;
    double I1ForEvenPrev2, I1ForEvenPrev3;
    
    double Q2, I2, prevQ2, prevI2;
    double Re, Im;
    double period, smoothPeriod;
    int hilbertIdx;
    
    double detrender, Q1, jI, jQ;
    
    void init() {
        std::memset(this, 0, sizeof(*this));
    }
};

// WMA state
struct WMAState {
    double periodWMASub, periodWMASum, trailingWMAValue;
    int trailingWMAIdx;
    
    void init() {
        periodWMASub = periodWMASum = trailingWMAValue = 0.0;
        trailingWMAIdx = 0;
    }
};

inline void initWMA(WMAState& wma, const float* input, int& today) {
    double t = static_cast<double>(input[today++]);
    wma.periodWMASub = t;
    wma.periodWMASum = t;
    t = static_cast<double>(input[today++]);
    wma.periodWMASub += t;
    wma.periodWMASum += t * 2.0;
    t = static_cast<double>(input[today++]);
    wma.periodWMASub += t;
    wma.periodWMASum += t * 3.0;
    wma.trailingWMAValue = 0.0;
}

inline double doWMAStep(WMAState& wma, const float* input, double newPrice) {
    wma.periodWMASub += newPrice;
    wma.periodWMASub -= wma.trailingWMAValue;
    wma.periodWMASum += newPrice * 4.0;
    wma.trailingWMAValue = static_cast<double>(input[wma.trailingWMAIdx++]);
    double smoothed = wma.periodWMASum * 0.1;
    wma.periodWMASum -= wma.periodWMASub;
    return smoothed;
}

inline void processHilbertStep(HilbertState& hs, double smoothedValue, int today,
                                double& outI1, double& outQ1) {
    double adjustedPrevPeriod = (0.075 * hs.period) + 0.54;
    double hilbertTempReal;
    
    if ((today & 1) == 0) {
        hilbertTempReal = HT_A * smoothedValue;
        hs.detrender = -hs.detrender_Even[hs.hilbertIdx];
        hs.detrender_Even[hs.hilbertIdx] = hilbertTempReal;
        hs.detrender += hilbertTempReal - hs.prev_detrender_Even;
        hs.prev_detrender_Even = HT_B * hs.prev_detrender_input_Even;
        hs.detrender += hs.prev_detrender_Even;
        hs.prev_detrender_input_Even = smoothedValue;
        hs.detrender *= adjustedPrevPeriod;

        hilbertTempReal = HT_A * hs.detrender;
        hs.Q1 = -hs.Q1_Even[hs.hilbertIdx];
        hs.Q1_Even[hs.hilbertIdx] = hilbertTempReal;
        hs.Q1 += hilbertTempReal - hs.prev_Q1_Even;
        hs.prev_Q1_Even = HT_B * hs.prev_Q1_input_Even;
        hs.Q1 += hs.prev_Q1_Even;
        hs.prev_Q1_input_Even = hs.detrender;
        hs.Q1 *= adjustedPrevPeriod;

        hilbertTempReal = HT_A * hs.I1ForEvenPrev3;
        hs.jI = -hs.jI_Even[hs.hilbertIdx];
        hs.jI_Even[hs.hilbertIdx] = hilbertTempReal;
        hs.jI += hilbertTempReal - hs.prev_jI_Even;
        hs.prev_jI_Even = HT_B * hs.prev_jI_input_Even;
        hs.jI += hs.prev_jI_Even;
        hs.prev_jI_input_Even = hs.I1ForEvenPrev3;
        hs.jI *= adjustedPrevPeriod;

        hilbertTempReal = HT_A * hs.Q1;
        hs.jQ = -hs.jQ_Even[hs.hilbertIdx];
        hs.jQ_Even[hs.hilbertIdx] = hilbertTempReal;
        hs.jQ += hilbertTempReal - hs.prev_jQ_Even;
        hs.prev_jQ_Even = HT_B * hs.prev_jQ_input_Even;
        hs.jQ += hs.prev_jQ_Even;
        hs.prev_jQ_input_Even = hs.Q1;
        hs.jQ *= adjustedPrevPeriod;

        if (++hs.hilbertIdx == 3) hs.hilbertIdx = 0;

        hs.Q2 = (0.2 * (hs.Q1 + hs.jI)) + (0.8 * hs.prevQ2);
        hs.I2 = (0.2 * (hs.I1ForEvenPrev3 - hs.jQ)) + (0.8 * hs.prevI2);

        outI1 = hs.I1ForEvenPrev3;
        outQ1 = hs.Q1;

        hs.I1ForOddPrev3 = hs.I1ForOddPrev2;
        hs.I1ForOddPrev2 = hs.detrender;
    } else {
        hilbertTempReal = HT_A * smoothedValue;
        hs.detrender = -hs.detrender_Odd[hs.hilbertIdx];
        hs.detrender_Odd[hs.hilbertIdx] = hilbertTempReal;
        hs.detrender += hilbertTempReal - hs.prev_detrender_Odd;
        hs.prev_detrender_Odd = HT_B * hs.prev_detrender_input_Odd;
        hs.detrender += hs.prev_detrender_Odd;
        hs.prev_detrender_input_Odd = smoothedValue;
        hs.detrender *= adjustedPrevPeriod;

        hilbertTempReal = HT_A * hs.detrender;
        hs.Q1 = -hs.Q1_Odd[hs.hilbertIdx];
        hs.Q1_Odd[hs.hilbertIdx] = hilbertTempReal;
        hs.Q1 += hilbertTempReal - hs.prev_Q1_Odd;
        hs.prev_Q1_Odd = HT_B * hs.prev_Q1_input_Odd;
        hs.Q1 += hs.prev_Q1_Odd;
        hs.prev_Q1_input_Odd = hs.detrender;
        hs.Q1 *= adjustedPrevPeriod;

        hilbertTempReal = HT_A * hs.I1ForOddPrev3;
        hs.jI = -hs.jI_Odd[hs.hilbertIdx];
        hs.jI_Odd[hs.hilbertIdx] = hilbertTempReal;
        hs.jI += hilbertTempReal - hs.prev_jI_Odd;
        hs.prev_jI_Odd = HT_B * hs.prev_jI_input_Odd;
        hs.jI += hs.prev_jI_Odd;
        hs.prev_jI_input_Odd = hs.I1ForOddPrev3;
        hs.jI *= adjustedPrevPeriod;

        hilbertTempReal = HT_A * hs.Q1;
        hs.jQ = -hs.jQ_Odd[hs.hilbertIdx];
        hs.jQ_Odd[hs.hilbertIdx] = hilbertTempReal;
        hs.jQ += hilbertTempReal - hs.prev_jQ_Odd;
        hs.prev_jQ_Odd = HT_B * hs.prev_jQ_input_Odd;
        hs.jQ += hs.prev_jQ_Odd;
        hs.prev_jQ_input_Odd = hs.Q1;
        hs.jQ *= adjustedPrevPeriod;

        hs.Q2 = (0.2 * (hs.Q1 + hs.jI)) + (0.8 * hs.prevQ2);
        hs.I2 = (0.2 * (hs.I1ForOddPrev3 - hs.jQ)) + (0.8 * hs.prevI2);

        outI1 = hs.I1ForOddPrev3;
        outQ1 = hs.Q1;

        if (++hs.hilbertIdx == 3) hs.hilbertIdx = 0;

        hs.I1ForEvenPrev3 = hs.I1ForEvenPrev2;
        hs.I1ForEvenPrev2 = hs.detrender;
    }
}

inline void updatePeriod(HilbertState& hs) {
    hs.Re = (0.2 * ((hs.I2 * hs.prevI2) + (hs.Q2 * hs.prevQ2))) + (0.8 * hs.Re);
    hs.Im = (0.2 * ((hs.I2 * hs.prevQ2) - (hs.Q2 * hs.prevI2))) + (0.8 * hs.Im);
    hs.prevQ2 = hs.Q2;
    hs.prevI2 = hs.I2;
    
    double tempPeriod = hs.period;
    if ((hs.Im != 0.0) && (hs.Re != 0.0))
        hs.period = 360.0 / (std::atan(hs.Im / hs.Re) * RAD2DEG);
    
    double temp2 = 1.5 * tempPeriod;
    if (hs.period > temp2) hs.period = temp2;
    temp2 = 0.67 * tempPeriod;
    if (hs.period < temp2) hs.period = temp2;
    if (hs.period < 6.0) hs.period = 6.0;
    else if (hs.period > 50.0) hs.period = 50.0;
    
    hs.period = (0.2 * hs.period) + (0.8 * tempPeriod);
    hs.smoothPeriod = (0.33 * hs.period) + (0.67 * hs.smoothPeriod);
}

// Update the running DC Phase — literal port of the ta_HT_DCPHASE.c block.
// TA-Lib's DCPhase is STATEFUL across bars: when |imagPart| is exactly zero
// the PREVIOUS phase is adjusted by +-90 degrees rather than recomputed;
// DCPeriodInt is not clamped, and the 360/smoothPeriod lag compensation is
// unconditional (smoothPeriod is never zero after the first main-loop bar).
inline void updateDCPhase(const double* smoothPrice, int smoothPrice_Idx,
                          double smoothPeriod, double& DCPhase) {
    double DCPeriod = smoothPeriod + 0.5;
    int DCPeriodInt = static_cast<int>(DCPeriod);
    
    double realPart = 0.0, imagPart = 0.0;
    int idx = smoothPrice_Idx;
    for (int i = 0; i < DCPeriodInt; ++i) {
        double phase = (static_cast<double>(i) * PI2) / static_cast<double>(DCPeriodInt);
        realPart += std::sin(phase) * smoothPrice[idx];
        imagPart += std::cos(phase) * smoothPrice[idx];
        idx = (idx == 0) ? 49 : idx - 1;
    }
    
    double absImag = std::fabs(imagPart);
    if (absImag > 0.0)
        DCPhase = std::atan(realPart / imagPart) * RAD2DEG;
    else if (absImag <= 0.01) {
        if (realPart < 0.0)
            DCPhase -= 90.0;
        else if (realPart > 0.0)
            DCPhase += 90.0;
        // realPart == 0: previous phase carried unchanged.
    }
    
    DCPhase += 90.0;
    DCPhase += 360.0 / smoothPeriod;
    if (imagPart < 0.0)
        DCPhase += 180.0;
    if (DCPhase > 315.0)
        DCPhase -= 360.0;
}

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
    if (size <= lookbackTotal) return;
    
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

