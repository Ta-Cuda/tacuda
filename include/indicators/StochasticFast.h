/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef STOCHASTIC_FAST_H
#define STOCHASTIC_FAST_H

#include "Indicator.h"

namespace tacuda {
class StochasticFast : public Indicator {
public:
    StochasticFast(int kPeriod, int dPeriod);
    void calculate(const float* high, const float* low, const float* close,
                   float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
    
    // Fused parallel implementation (2-pass kernel fusion)
    void calculateFused(const float* high, const float* low, const float* close,
                       float* output, int size, cudaStream_t stream = 0) noexcept(false);
private:
    int kPeriod;
    int dPeriod;
};

} // namespace tacuda

#endif
