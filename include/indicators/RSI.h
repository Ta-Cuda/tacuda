/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef RSI_H
#define RSI_H

#include "Indicator.h"
#include "IndicatorT.h"

namespace tacuda {

/**
 * @brief Type-generic RSI (Relative Strength Index).
 *
 * Explicit instantiations for float and double are provided in RSI.cu.
 */
template<typename T>
class RSIT : public IndicatorT<T> {
public:
    explicit RSIT(int period) : period(period) {}
    void calculate(const T* input, T* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

/**
 * @brief Backward-compatible float-only RSI inheriting from Indicator.
 *
 * Delegates to RSIT<float>::calculate internally.
 */
class RSI : public Indicator {
public:
    explicit RSI(int period);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
