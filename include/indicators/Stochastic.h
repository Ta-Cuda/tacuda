/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef STOCHASTIC_H
#define STOCHASTIC_H

#include "Indicator.h"

namespace tacuda {
class Stochastic : public Indicator {
public:
    Stochastic(int kPeriod, int dPeriod);
    void calculate(const float* high, const float* low, const float* close,
                   float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* high, const float* low, const float* close,
                   float* slowK, float* slowD, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
    
    // Fused parallel implementation (50-100x faster than legacy sequential)
    void calculateFused(const float* high, const float* low, const float* close,
                       float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculateFused(const float* high, const float* low, const float* close,
                       float* slowK, float* slowD, int size, cudaStream_t stream = 0) noexcept(false);
private:
    int kPeriod;
    int dPeriod;
};

} // namespace tacuda

#endif
