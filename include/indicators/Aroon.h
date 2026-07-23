/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef AROON_H
#define AROON_H

#include "Indicator.h"

namespace tacuda {
class Aroon : public Indicator {
public:
    Aroon(int upPeriod, int downPeriod);
    void calculate(const float* high, const float* low, float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int upPeriod;
    int downPeriod;
};

} // namespace tacuda

#endif
