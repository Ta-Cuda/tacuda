/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef HILBERT_CORE_H
#define HILBERT_CORE_H

#include <cuda_runtime.h>
#include <cmath>

// ============================================================================
// Hilbert Transform Core - shared state and device functions for all HT_* indicators
// ============================================================================

namespace tacuda {
namespace hilbert {

// Constants
__device__ __constant__ constexpr double HT_A = 0.0962;
__device__ __constant__ constexpr double HT_B = 0.5769;

// Hilbert Transform state structure
struct HilbertState {
    // Detrender state (odd/even)
    double detrender_Odd[3];
    double detrender_Even[3];
    double prev_detrender_Odd;
    double prev_detrender_Even;
    double prev_detrender_input_Odd;
    double prev_detrender_input_Even;
    
    // Q1 state
    double Q1_Odd[3];
    double Q1_Even[3];
    double prev_Q1_Odd;
    double prev_Q1_Even;
    double prev_Q1_input_Odd;
    double prev_Q1_input_Even;
    
    // jI state
    double jI_Odd[3];
    double jI_Even[3];
    double prev_jI_Odd;
    double prev_jI_Even;
    double prev_jI_input_Odd;
    double prev_jI_input_Even;
    
    // jQ state
    double jQ_Odd[3];
    double jQ_Even[3];
    double prev_jQ_Odd;
    double prev_jQ_Even;
    double prev_jQ_input_Odd;
    double prev_jQ_input_Even;
    
    // I1 history
    double I1ForOddPrev2;
    double I1ForOddPrev3;
    double I1ForEvenPrev2;
    double I1ForEvenPrev3;
    
    // Period estimation state
    double Q2, I2, prevQ2, prevI2;
    double Re, Im;
    double period, smoothPeriod;
    
    // Hilbert index (cycles through 0,1,2)
    int hilbertIdx;
    
    // Current outputs
    double detrender;
    double Q1;
    double jI;
    double jQ;
    
    __device__ void init() {
        for (int i = 0; i < 3; ++i) {
            detrender_Odd[i] = 0.0;
            detrender_Even[i] = 0.0;
            Q1_Odd[i] = 0.0;
            Q1_Even[i] = 0.0;
            jI_Odd[i] = 0.0;
            jI_Even[i] = 0.0;
            jQ_Odd[i] = 0.0;
            jQ_Even[i] = 0.0;
        }
        prev_detrender_Odd = prev_detrender_Even = 0.0;
        prev_detrender_input_Odd = prev_detrender_input_Even = 0.0;
        prev_Q1_Odd = prev_Q1_Even = 0.0;
        prev_Q1_input_Odd = prev_Q1_input_Even = 0.0;
        prev_jI_Odd = prev_jI_Even = 0.0;
        prev_jI_input_Odd = prev_jI_input_Even = 0.0;
        prev_jQ_Odd = prev_jQ_Even = 0.0;
        prev_jQ_input_Odd = prev_jQ_input_Even = 0.0;
        
        I1ForOddPrev2 = I1ForOddPrev3 = 0.0;
        I1ForEvenPrev2 = I1ForEvenPrev3 = 0.0;
        
        Q2 = I2 = prevQ2 = prevI2 = 0.0;
        Re = Im = 0.0;
        period = smoothPeriod = 0.0;
        hilbertIdx = 0;
        detrender = Q1 = jI = jQ = 0.0;
    }
};

// WMA state for price smoothing
struct WMAState {
    double periodWMASub;
    double periodWMASum;
    double trailingWMAValue;
    int trailingWMAIdx;
    
    __device__ void init() {
        periodWMASub = 0.0;
        periodWMASum = 0.0;
        trailingWMAValue = 0.0;
        trailingWMAIdx = 0;
    }
};

// Initialize WMA with first 3 values
__device__ __forceinline__ void initWMA(WMAState& wma, const float* input, int& today) {
    double tempReal = static_cast<double>(input[today++]);
    wma.periodWMASub = tempReal;
    wma.periodWMASum = tempReal;
    tempReal = static_cast<double>(input[today++]);
    wma.periodWMASub += tempReal;
    wma.periodWMASum += tempReal * 2.0;
    tempReal = static_cast<double>(input[today++]);
    wma.periodWMASub += tempReal;
    wma.periodWMASum += tempReal * 3.0;
    wma.trailingWMAValue = 0.0;
}

// Compute WMA step
__device__ __forceinline__ double doWMAStep(WMAState& wma, const float* input, double newPrice) {
    wma.periodWMASub += newPrice;
    wma.periodWMASub -= wma.trailingWMAValue;
    wma.periodWMASum += newPrice * 4.0;
    wma.trailingWMAValue = static_cast<double>(input[wma.trailingWMAIdx++]);
    double smoothed = wma.periodWMASum * 0.1;
    wma.periodWMASum -= wma.periodWMASub;
    return smoothed;
}

// Process one Hilbert step (even or odd bar)
__device__ __forceinline__ void processHilbertStep(
    HilbertState& hs,
    double smoothedValue,
    int today,
    double& outI1,   // For PHASOR
    double& outQ1    // For PHASOR
) {
    constexpr double a = 0.0962;
    constexpr double b = 0.5769;
    
    double adjustedPrevPeriod = (0.075 * hs.period) + 0.54;
    double hilbertTempReal;
    
    if ((today & 1) == 0) {
        // Even bar
        hilbertTempReal = a * smoothedValue;
        hs.detrender = -hs.detrender_Even[hs.hilbertIdx];
        hs.detrender_Even[hs.hilbertIdx] = hilbertTempReal;
        hs.detrender += hilbertTempReal;
        hs.detrender -= hs.prev_detrender_Even;
        hs.prev_detrender_Even = b * hs.prev_detrender_input_Even;
        hs.detrender += hs.prev_detrender_Even;
        hs.prev_detrender_input_Even = smoothedValue;
        hs.detrender *= adjustedPrevPeriod;

        hilbertTempReal = a * hs.detrender;
        hs.Q1 = -hs.Q1_Even[hs.hilbertIdx];
        hs.Q1_Even[hs.hilbertIdx] = hilbertTempReal;
        hs.Q1 += hilbertTempReal;
        hs.Q1 -= hs.prev_Q1_Even;
        hs.prev_Q1_Even = b * hs.prev_Q1_input_Even;
        hs.Q1 += hs.prev_Q1_Even;
        hs.prev_Q1_input_Even = hs.detrender;
        hs.Q1 *= adjustedPrevPeriod;

        hilbertTempReal = a * hs.I1ForEvenPrev3;
        hs.jI = -hs.jI_Even[hs.hilbertIdx];
        hs.jI_Even[hs.hilbertIdx] = hilbertTempReal;
        hs.jI += hilbertTempReal;
        hs.jI -= hs.prev_jI_Even;
        hs.prev_jI_Even = b * hs.prev_jI_input_Even;
        hs.jI += hs.prev_jI_Even;
        hs.prev_jI_input_Even = hs.I1ForEvenPrev3;
        hs.jI *= adjustedPrevPeriod;

        hilbertTempReal = a * hs.Q1;
        hs.jQ = -hs.jQ_Even[hs.hilbertIdx];
        hs.jQ_Even[hs.hilbertIdx] = hilbertTempReal;
        hs.jQ += hilbertTempReal;
        hs.jQ -= hs.prev_jQ_Even;
        hs.prev_jQ_Even = b * hs.prev_jQ_input_Even;
        hs.jQ += hs.prev_jQ_Even;
        hs.prev_jQ_input_Even = hs.Q1;
        hs.jQ *= adjustedPrevPeriod;

        if (++hs.hilbertIdx == 3)
            hs.hilbertIdx = 0;

        hs.Q2 = (0.2 * (hs.Q1 + hs.jI)) + (0.8 * hs.prevQ2);
        hs.I2 = (0.2 * (hs.I1ForEvenPrev3 - hs.jQ)) + (0.8 * hs.prevI2);

        outI1 = hs.I1ForEvenPrev3;
        outQ1 = hs.Q1;

        hs.I1ForOddPrev3 = hs.I1ForOddPrev2;
        hs.I1ForOddPrev2 = hs.detrender;
    } else {
        // Odd bar
        hilbertTempReal = a * smoothedValue;
        hs.detrender = -hs.detrender_Odd[hs.hilbertIdx];
        hs.detrender_Odd[hs.hilbertIdx] = hilbertTempReal;
        hs.detrender += hilbertTempReal;
        hs.detrender -= hs.prev_detrender_Odd;
        hs.prev_detrender_Odd = b * hs.prev_detrender_input_Odd;
        hs.detrender += hs.prev_detrender_Odd;
        hs.prev_detrender_input_Odd = smoothedValue;
        hs.detrender *= adjustedPrevPeriod;

        hilbertTempReal = a * hs.detrender;
        hs.Q1 = -hs.Q1_Odd[hs.hilbertIdx];
        hs.Q1_Odd[hs.hilbertIdx] = hilbertTempReal;
        hs.Q1 += hilbertTempReal;
        hs.Q1 -= hs.prev_Q1_Odd;
        hs.prev_Q1_Odd = b * hs.prev_Q1_input_Odd;
        hs.Q1 += hs.prev_Q1_Odd;
        hs.prev_Q1_input_Odd = hs.detrender;
        hs.Q1 *= adjustedPrevPeriod;

        hilbertTempReal = a * hs.I1ForOddPrev3;
        hs.jI = -hs.jI_Odd[hs.hilbertIdx];
        hs.jI_Odd[hs.hilbertIdx] = hilbertTempReal;
        hs.jI += hilbertTempReal;
        hs.jI -= hs.prev_jI_Odd;
        hs.prev_jI_Odd = b * hs.prev_jI_input_Odd;
        hs.jI += hs.prev_jI_Odd;
        hs.prev_jI_input_Odd = hs.I1ForOddPrev3;
        hs.jI *= adjustedPrevPeriod;

        hilbertTempReal = a * hs.Q1;
        hs.jQ = -hs.jQ_Odd[hs.hilbertIdx];
        hs.jQ_Odd[hs.hilbertIdx] = hilbertTempReal;
        hs.jQ += hilbertTempReal;
        hs.jQ -= hs.prev_jQ_Odd;
        hs.prev_jQ_Odd = b * hs.prev_jQ_input_Odd;
        hs.jQ += hs.prev_jQ_Odd;
        hs.prev_jQ_input_Odd = hs.Q1;
        hs.jQ *= adjustedPrevPeriod;

        hs.Q2 = (0.2 * (hs.Q1 + hs.jI)) + (0.8 * hs.prevQ2);
        hs.I2 = (0.2 * (hs.I1ForOddPrev3 - hs.jQ)) + (0.8 * hs.prevI2);

        outI1 = hs.I1ForOddPrev3;
        outQ1 = hs.Q1;

        if (++hs.hilbertIdx == 3)
            hs.hilbertIdx = 0;

        hs.I1ForEvenPrev3 = hs.I1ForEvenPrev2;
        hs.I1ForEvenPrev2 = hs.detrender;
    }
}

// Update period estimation
__device__ __forceinline__ void updatePeriod(HilbertState& hs) {
    constexpr double rad2Deg = 57.29577951308232; // 180 / PI
    
    hs.Re = (0.2 * ((hs.I2 * hs.prevI2) + (hs.Q2 * hs.prevQ2))) + (0.8 * hs.Re);
    hs.Im = (0.2 * ((hs.I2 * hs.prevQ2) - (hs.Q2 * hs.prevI2))) + (0.8 * hs.Im);
    hs.prevQ2 = hs.Q2;
    hs.prevI2 = hs.I2;
    
    double tempPeriod = hs.period;
    if ((hs.Im != 0.0) && (hs.Re != 0.0))
        hs.period = 360.0 / (atan(hs.Im / hs.Re) * rad2Deg);
    
    double temp2 = 1.5 * tempPeriod;
    if (hs.period > temp2) hs.period = temp2;
    temp2 = 0.67 * tempPeriod;
    if (hs.period < temp2) hs.period = temp2;
    if (hs.period < 6.0) hs.period = 6.0;
    else if (hs.period > 50.0) hs.period = 50.0;
    
    hs.period = (0.2 * hs.period) + (0.8 * tempPeriod);
    hs.smoothPeriod = (0.33 * hs.period) + (0.67 * hs.smoothPeriod);
}

// Update the running DC Phase from the smooth price buffer — literal port
// of the ta_HT_DCPHASE.c block. NOTE: TA-Lib's DCPhase is STATEFUL across
// bars: when |imagPart| is exactly zero the PREVIOUS phase is adjusted by
// +-90 degrees rather than recomputed, and all subsequent offsets stack on
// top of the carried value. DCPeriodInt is intentionally not clamped and
// the 360/smoothPeriod lag compensation is unconditional, exactly like
// TA-Lib (smoothPeriod is never zero once the main loop has run one bar).
__device__ __forceinline__ void updateDCPhase(
    const double* smoothPrice,
    int smoothPrice_Idx,
    double smoothPeriod,
    double& DCPhase
) {
    constexpr double rad2Deg = 57.29577951308232;
    constexpr double constDeg2RadBy360 = 6.283185307179586; // 2*PI
    
    double DCPeriod = smoothPeriod + 0.5;
    int DCPeriodInt = static_cast<int>(DCPeriod);
    
    double realPart = 0.0, imagPart = 0.0;
    int idx = smoothPrice_Idx;
    for (int i = 0; i < DCPeriodInt; ++i) {
        double phase = (static_cast<double>(i) * constDeg2RadBy360) / static_cast<double>(DCPeriodInt);
        double tempPrice = smoothPrice[idx];
        realPart += sin(phase) * tempPrice;
        imagPart += cos(phase) * tempPrice;
        if (idx == 0)
            idx = 49; // SMOOTH_PRICE_SIZE - 1
        else
            idx--;
    }
    
    double absImag = fabs(imagPart);
    if (absImag > 0.0)
        DCPhase = atan(realPart / imagPart) * rad2Deg;
    else if (absImag <= 0.01) {
        if (realPart < 0.0)
            DCPhase -= 90.0;
        else if (realPart > 0.0)
            DCPhase += 90.0;
        // realPart == 0: previous phase carried unchanged.
    }
    
    DCPhase += 90.0;
    // Compensate for the one-bar lag of the weighted moving average.
    DCPhase += 360.0 / smoothPeriod;
    if (imagPart < 0.0)
        DCPhase += 180.0;
    if (DCPhase > 315.0)
        DCPhase -= 360.0;
}

} // namespace hilbert
} // namespace tacuda

#endif // HILBERT_CORE_H

