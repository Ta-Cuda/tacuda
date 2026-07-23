/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef ADOSC_H
#define ADOSC_H

#include "Indicator.h"

namespace tacuda {
class ADOSC : public Indicator {
public:
    ADOSC(int shortPeriod, int longPeriod);
    void calculate(const float* high, const float* low, const float* close,
                   const float* volume, float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int shortPeriod;
    int longPeriod;
};

} // namespace tacuda

#endif
