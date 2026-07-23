/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef NVI_H
#define NVI_H

#include "Indicator.h"

namespace tacuda {
class NVI : public Indicator {
public:
    NVI() = default;
    void calculate(const float* close, const float* volume, float* output,
                   int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size,
                   cudaStream_t stream = 0) noexcept(false) override;
};

} // namespace tacuda

#endif
