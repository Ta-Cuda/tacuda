/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef INDICATOR_H
#define INDICATOR_H

#include <cuda_runtime.h>

namespace tacuda {
class Indicator {
public:
    virtual void calculate(const float* input, float* output, int size,
                           cudaStream_t stream = 0) noexcept(false) = 0;
    virtual ~Indicator() = default;
};

} // namespace tacuda

#endif
