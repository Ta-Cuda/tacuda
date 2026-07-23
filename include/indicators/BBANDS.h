/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef BBANDS_H
#define BBANDS_H

#include "Indicator.h"

namespace tacuda {
class BBANDS : public Indicator {
public:
    BBANDS(int period, float upperMultiplier, float lowerMultiplier);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
    void calculate(const float* input, float* upper, float* middle, float* lower,
                   int size, cudaStream_t stream = 0) noexcept(false);
private:
    int period;
    float upperMultiplier;
    float lowerMultiplier;
};

} // namespace tacuda

#endif
