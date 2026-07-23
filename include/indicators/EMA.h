/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef EMA_H
#define EMA_H

#include "Indicator.h"
#include "IndicatorT.h"

namespace tacuda {

/**
 * @brief Type-generic EMA (Exponential Moving Average).
 *
 * Explicit instantiations for float and double are provided in EMA.cu.
 */
template<typename T>
class EMAT : public IndicatorT<T> {
public:
    explicit EMAT(int period) : period(period) {}
    void calculate(const T* input, T* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

/**
 * @brief Backward-compatible float-only EMA inheriting from Indicator.
 *
 * Delegates to EMAT<float>::calculate internally.
 */
class EMA : public Indicator {
public:
    explicit EMA(int period);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
