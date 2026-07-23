/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef MACDFIX_H
#define MACDFIX_H

#include "Indicator.h"
#include "MA.h"

namespace tacuda {
class MACDFIX : public Indicator {
public:
    explicit MACDFIX(int signalPeriod);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    int signalPeriod;
};

} // namespace tacuda

#endif
