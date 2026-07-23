/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef HT_DCPERIOD_H
#define HT_DCPERIOD_H

#include "Indicator.h"

namespace tacuda {
class HT_DCPERIOD : public Indicator {
public:
    HT_DCPERIOD() = default;
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
};

} // namespace tacuda

#endif
