/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef CLUBPACK_H
#define CLUBPACK_H

/**
 * @file ClubPack.h
 * @brief "Club pack" indicators: constructs from the Omega Research System
 * Trading & Development Club volumes (1997-2000) that have no TA-Lib
 * counterpart. Semantics follow the tacuda-opt EXPERIMENTS strategy pack
 * (tools/optimizer/src/strategy), which is the reference CPU implementation.
 *
 * All recursions run as single-thread device kernels (like SAR); windowed
 * indicators use one thread per output element. Multi-input calculate()
 * overloads take separate device pointers; the Indicator::calculate override
 * expects inputs stacked contiguously ([in0 | in1 | ...], each `size` long).
 */

#include "Indicator.h"

namespace tacuda {

/// Wilder's Accumulation Swing Index, limit-move term dropped (data is not
/// exchange-limited; the line is used for relative comparisons only).
/// Output: cumulative ASI, NaN at bar 0.
class ASI : public Indicator {
public:
    ASI() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;
};

/// EasyLanguage-style parabolic SAR: direction seeded from the first two
/// closes, acceleration starts at accFact and steps by accFact up to
/// maxAccel (the club systems use 0.2). Differs from TA-Lib ct_sar in the
/// seeding convention. Output: SAR level per bar, NaN at bar 0.
class SAREL : public Indicator {
public:
    SAREL(float accFact, float maxAccel);
    void calculate(const float* high, const float* low, const float* close,
                   float* output, int size,
                   cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;

private:
    float accFact;
    float maxAccel;
};

/// On Balance True Range: cumulative true range signed by the close-to-close
/// direction (up adds TR, down subtracts, unchanged keeps). NaN at bar 0.
class OBTR : public Indicator {
public:
    OBTR() = default;
    void calculate(const float* high, const float* low, const float* close,
                   float* output, int size,
                   cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;
};

/// Weighted OBV: cumulative (close-open)/(high-low) * volume; bars with an
/// empty range contribute nothing. Defined from bar 0.
class WOBV : public Indicator {
public:
    WOBV() = default;
    void calculate(const float* open, const float* high, const float* low,
                   const float* close, const float* volume, float* output,
                   int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;
};

/// Herrick Payoff Index proxy: exponentially smoothed mid-price change times
/// volume, acc += factor * (raw - acc). NaN at bar 0.
class HPI : public Indicator {
public:
    explicit HPI(float factor);
    void calculate(const float* high, const float* low, const float* volume,
                   float* output, int size,
                   cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;

private:
    float factor;
};

/// MarketStrength: net `period`-bar close change scaled by the same-signed
/// portion of the bar-to-bar movement, in [-100, 100]. NaN before bar
/// `period`.
class MarketStrength : public Indicator {
public:
    explicit MarketStrength(int period);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;

private:
    int period;
};

/// Chande's TrendScore: sum of +/-1 for the close compared against closes
/// lookback+1 .. lookback+10 bars back. NaN before bar lookback+10.
class TrendScore : public Indicator {
public:
    explicit TrendScore(int lookback);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;

private:
    int lookback;
};

/// Reference Deviation: RDV = 100 * sum(drd) / sum(|drd|) over the last
/// `sumPeriod` bars, drd = input - SMA(smaPeriod). NaN during warm-up and
/// when sum(|drd|) is zero.
class RefDev : public Indicator {
public:
    RefDev(int smaPeriod, int sumPeriod);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;

private:
    int smaPeriod;
    int sumPeriod;
};

/// Rolling population skewness m3 / m2^1.5 over `period` bars (0 when the
/// window variance is zero). NaN before bar period-1.
class RollingSkew : public Indicator {
public:
    explicit RollingSkew(int period);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;

private:
    int period;
};

/// Rolling population excess kurtosis m4 / m2^2 - 3 over `period` bars
/// (0 when the window variance is zero). NaN before bar period-1.
class RollingKurtosis : public Indicator {
public:
    explicit RollingKurtosis(int period);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;

private:
    int period;
};

/// Swing pivot high of `strength` bars, reported causally at the
/// confirmation bar: output[i] = input[i-strength] when that bar is strictly
/// higher than the `strength` bars before it and >= the `strength` bars
/// after it; NaN otherwise.
class PivotHigh : public Indicator {
public:
    explicit PivotHigh(int strength);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;

private:
    int strength;
};

/// Swing pivot low, mirror of PivotHigh (strictly lower before, <= after).
class PivotLow : public Indicator {
public:
    explicit PivotLow(int strength);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;

private:
    int strength;
};

/// EasyLanguage XAverage over a possibly NaN-prefixed line: seeded with the
/// first finite value, alpha = 2/(period+1); NaN inputs are skipped and
/// reported as NaN without disturbing the recursion.
class XAverage : public Indicator {
public:
    explicit XAverage(int period);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;

private:
    int period;
};

/// NaN-strict rolling mean over `period` values: the output is defined only
/// when every value in the window is finite. Unlike SMA's prefix-sum path,
/// a leading NaN prefix (typical for derived/chained lines) never poisons
/// the rest of the series.
class RollingMean : public Indicator {
public:
    explicit RollingMean(int period);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;

private:
    int period;
};

/// NaN-strict rolling maximum (see RollingMean for the NaN convention).
class RollingMax : public Indicator {
public:
    explicit RollingMax(int period);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;

private:
    int period;
};

/// NaN-strict rolling minimum (see RollingMean for the NaN convention).
class RollingMin : public Indicator {
public:
    explicit RollingMin(int period);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;

private:
    int period;
};

/// NaN-strict rolling population standard deviation.
class RollingStd : public Indicator {
public:
    explicit RollingStd(int period);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;

private:
    int period;
};

} // namespace tacuda

#endif // CLUBPACK_H
