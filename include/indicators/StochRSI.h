/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef STOCHRSI_H
#define STOCHRSI_H

#include "Indicator.h"

namespace tacuda {
class StochRSI : public Indicator {
public:
  StochRSI(int rsiPeriod, int kPeriod, int dPeriod);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int rsiPeriod;
  int kPeriod;
  int dPeriod;
};

} // namespace tacuda

#endif
