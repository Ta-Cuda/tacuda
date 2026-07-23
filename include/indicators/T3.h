/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef T3_H
#define T3_H

#include "Indicator.h"

namespace tacuda {
class T3 : public Indicator {
public:
  T3(int period, float vFactor);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int period;
  float vFactor;
};

} // namespace tacuda

#endif
