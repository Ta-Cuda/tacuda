/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef SMA_H
#define SMA_H

#include "Indicator.h"
#include "IndicatorT.h"

namespace tacuda {

/**
 * @brief Type-generic SMA (Simple Moving Average).
 *
 * Explicit instantiations for float and double are provided in SMA.cu.
 */
template<typename T>
class SMAT : public IndicatorT<T> {
public:
    explicit SMAT(int period) : period(period) {}
    void calculate(const T* input, T* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

/**
 * @brief Backward-compatible float-only SMA inheriting from Indicator.
 *
 * Delegates to SMAT<float>::calculate internally.
 */
class SMA : public Indicator {
public:
    explicit SMA(int period);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;
private:
    int period;
};

} // namespace tacuda

#endif
