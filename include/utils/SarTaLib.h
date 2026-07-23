/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_SAR_TALIB_H
#define TACUDA_SAR_TALIB_H

/**
 * @file SarTaLib.h
 * @brief Exact ports of TA-Lib TA_SAR / TA_SAREXT usable from both host and
 *        device code (single source of truth for the CPU fast path in the
 *        host API and the CUDA kernels used by the device API).
 *
 * Reference: ta-lib/src/ta_func/ta_SAR.c and ta_SAREXT.c.
 * Both algorithms are inherently sequential (state carried across bars), so
 * the host API computes them on the CPU while the device API runs a single
 * GPU thread to keep data resident on the device.
 */

#include <cmath>

#if defined(__CUDACC__)
#define TACUDA_HD __host__ __device__
#else
#define TACUDA_HD
#endif

namespace tacuda {
namespace detail {

TACUDA_HD inline float sar_nan() {
#if defined(__CUDA_ARCH__)
    return nanf("");
#else
    return std::nanf("");
#endif
}

/**
 * @brief TA-Lib TA_SAR port. Writes NaN at index 0, values from index 1.
 *
 * Initial direction is derived from the first +DM/-DM pair (ties go long).
 */
TACUDA_HD inline void sar_talib_run(const float* high, const float* low,
                                    int size, double acceleration,
                                    double maximum, float* out) {
    out[0] = sar_nan();
    if (size <= 1) return;

    double af = acceleration;
    if (af > maximum) af = acceleration = maximum;

    // Initial direction from +DM/-DM between bar 0 and 1 (TA-Lib rule).
    double diffP = static_cast<double>(high[1]) - static_cast<double>(high[0]);
    double diffM = static_cast<double>(low[0]) - static_cast<double>(low[1]);
    bool isLong = !((diffM > 0.0) && (diffP < diffM));

    double ep, sar;
    if (isLong) {
        ep = static_cast<double>(high[1]);
        sar = static_cast<double>(low[0]);
    } else {
        ep = static_cast<double>(low[1]);
        sar = static_cast<double>(high[0]);
    }

    // Cheat on newLow/newHigh for the first iteration (TA-Lib does the same).
    double newLow = static_cast<double>(low[1]);
    double newHigh = static_cast<double>(high[1]);

    int outIdx = 1;
    int todayIdx = 1;
    while (todayIdx <= size - 1) {
        double prevLow = newLow;
        double prevHigh = newHigh;
        newLow = static_cast<double>(low[todayIdx]);
        newHigh = static_cast<double>(high[todayIdx]);
        todayIdx++;

        if (isLong) {
            if (newLow <= sar) {
                isLong = false;
                sar = ep;
                if (sar < prevHigh) sar = prevHigh;
                if (sar < newHigh) sar = newHigh;
                out[outIdx++] = static_cast<float>(sar);

                af = acceleration;
                ep = newLow;
                sar = sar + af * (ep - sar);
                if (sar < prevHigh) sar = prevHigh;
                if (sar < newHigh) sar = newHigh;
            } else {
                out[outIdx++] = static_cast<float>(sar);

                if (newHigh > ep) {
                    ep = newHigh;
                    af += acceleration;
                    if (af > maximum) af = maximum;
                }
                sar = sar + af * (ep - sar);
                if (sar > prevLow) sar = prevLow;
                if (sar > newLow) sar = newLow;
            }
        } else {
            if (newHigh >= sar) {
                isLong = true;
                sar = ep;
                if (sar > prevLow) sar = prevLow;
                if (sar > newLow) sar = newLow;
                out[outIdx++] = static_cast<float>(sar);

                af = acceleration;
                ep = newHigh;
                sar = sar + af * (ep - sar);
                if (sar > prevLow) sar = prevLow;
                if (sar > newLow) sar = newLow;
            } else {
                out[outIdx++] = static_cast<float>(sar);

                if (newLow < ep) {
                    ep = newLow;
                    af += acceleration;
                    if (af > maximum) af = maximum;
                }
                sar = sar + af * (ep - sar);
                if (sar < prevHigh) sar = prevHigh;
                if (sar < newHigh) sar = newHigh;
            }
        }
    }
}

/**
 * @brief TA-Lib TA_SAREXT port. Writes NaN at index 0, values from index 1.
 *
 * Semantics (matching TA-Lib exactly):
 *  - startValue == 0: initial direction from the first +DM/-DM pair;
 *  - startValue  > 0: start long with SAR = startValue;
 *  - startValue  < 0: start short with SAR = |startValue|;
 *  - offsetOnReverse is a relative offset applied on reversal
 *    (sar += sar * offset when flipping to short, sar -= sar * offset to long);
 *  - output is negated while in a short position.
 */
TACUDA_HD inline void sarext_talib_run(const float* high, const float* low,
                                       int size, double startValue,
                                       double offsetOnReverse,
                                       double accInitLong, double accLong,
                                       double accMaxLong, double accInitShort,
                                       double accShort, double accMaxShort,
                                       float* out) {
    out[0] = sar_nan();
    if (size <= 1) return;

    // Coherence clamps (TA-Lib mutates the parameters the same way).
    double afLong = accInitLong;
    double afShort = accInitShort;
    if (afLong > accMaxLong) afLong = accInitLong = accMaxLong;
    if (accLong > accMaxLong) accLong = accMaxLong;
    if (afShort > accMaxShort) afShort = accInitShort = accMaxShort;
    if (accShort > accMaxShort) accShort = accMaxShort;

    bool isLong;
    if (startValue == 0.0) {
        double diffP = static_cast<double>(high[1]) - static_cast<double>(high[0]);
        double diffM = static_cast<double>(low[0]) - static_cast<double>(low[1]);
        isLong = !((diffM > 0.0) && (diffP < diffM));
    } else {
        isLong = startValue > 0.0;
    }

    double ep, sar;
    if (startValue == 0.0) {
        if (isLong) {
            ep = static_cast<double>(high[1]);
            sar = static_cast<double>(low[0]);
        } else {
            ep = static_cast<double>(low[1]);
            sar = static_cast<double>(high[0]);
        }
    } else if (startValue > 0.0) {
        ep = static_cast<double>(high[1]);
        sar = startValue;
    } else {
        ep = static_cast<double>(low[1]);
        sar = -startValue; // fabs(startValue)
    }

    double newLow = static_cast<double>(low[1]);
    double newHigh = static_cast<double>(high[1]);

    int outIdx = 1;
    int todayIdx = 1;
    while (todayIdx <= size - 1) {
        double prevLow = newLow;
        double prevHigh = newHigh;
        newLow = static_cast<double>(low[todayIdx]);
        newHigh = static_cast<double>(high[todayIdx]);
        todayIdx++;

        if (isLong) {
            if (newLow <= sar) {
                isLong = false;
                sar = ep;
                if (sar < prevHigh) sar = prevHigh;
                if (sar < newHigh) sar = newHigh;
                if (offsetOnReverse != 0.0) sar += sar * offsetOnReverse;
                out[outIdx++] = static_cast<float>(-sar);

                afShort = accInitShort;
                ep = newLow;
                sar = sar + afShort * (ep - sar);
                if (sar < prevHigh) sar = prevHigh;
                if (sar < newHigh) sar = newHigh;
            } else {
                out[outIdx++] = static_cast<float>(sar);

                if (newHigh > ep) {
                    ep = newHigh;
                    afLong += accLong;
                    if (afLong > accMaxLong) afLong = accMaxLong;
                }
                sar = sar + afLong * (ep - sar);
                if (sar > prevLow) sar = prevLow;
                if (sar > newLow) sar = newLow;
            }
        } else {
            if (newHigh >= sar) {
                isLong = true;
                sar = ep;
                if (sar > prevLow) sar = prevLow;
                if (sar > newLow) sar = newLow;
                if (offsetOnReverse != 0.0) sar -= sar * offsetOnReverse;
                out[outIdx++] = static_cast<float>(sar);

                afLong = accInitLong;
                ep = newHigh;
                sar = sar + afLong * (ep - sar);
                if (sar > prevLow) sar = prevLow;
                if (sar > newLow) sar = newLow;
            } else {
                out[outIdx++] = static_cast<float>(-sar);

                if (newLow < ep) {
                    ep = newLow;
                    afShort += accShort;
                    if (afShort > accMaxShort) afShort = accMaxShort;
                }
                sar = sar + afShort * (ep - sar);
                if (sar < prevHigh) sar = prevHigh;
                if (sar < newHigh) sar = newHigh;
            }
        }
    }
}

} // namespace detail
} // namespace tacuda

#endif // TACUDA_SAR_TALIB_H
