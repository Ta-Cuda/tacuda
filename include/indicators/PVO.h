/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef PVO_H
#define PVO_H

#include "Indicator.h"

namespace tacuda {
class PVO : public Indicator {
public:
  PVO(int fastPeriod, int slowPeriod);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int fastPeriod;
  int slowPeriod;
};

} // namespace tacuda

#endif
