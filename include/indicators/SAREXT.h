/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef SAREXT_H
#define SAREXT_H

#include "Indicator.h"

namespace tacuda {
class SAREXT : public Indicator {
public:
    SAREXT(float startValue, float offsetOnReverse,
           float accInitLong, float accLong, float accMaxLong,
           float accInitShort, float accShort, float accMaxShort);
    void calculate(const float* high, const float* low,
                   float* output, int size, cudaStream_t stream = 0) noexcept(false);
    void calculate(const float* input, float* output, int size, cudaStream_t stream = 0) noexcept(false) override;
private:
    float startValue, offsetOnReverse;
    float accInitLong, accLong, accMaxLong;
    float accInitShort, accShort, accMaxShort;
};

} // namespace tacuda

#endif
