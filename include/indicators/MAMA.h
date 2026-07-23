/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef MAMA_H
#define MAMA_H

#include "Indicator.h"

namespace tacuda {
class MAMA : public Indicator {
public:
    MAMA(float fastLimit, float slowLimit);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    float fastLimit;
    float slowLimit;
};

} // namespace tacuda

#endif
