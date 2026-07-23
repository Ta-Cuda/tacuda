/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef ULTOSC_H
#define ULTOSC_H

#include "Indicator.h"

namespace tacuda {
class ULTOSC : public Indicator {
public:
    ULTOSC(int shortPeriod, int mediumPeriod, int longPeriod);
    void calculate(const float* high, const float* low, const float* close,
                   float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int shortPeriod;
    int mediumPeriod;
    int longPeriod;
};

} // namespace tacuda

#endif
