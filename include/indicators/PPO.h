/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef PPO_H
#define PPO_H

#include "Indicator.h"
#include "MA.h"

namespace tacuda {
class PPO : public Indicator {
public:
  // TA-Lib TA_PPO: selectable MA type (default EMA preserves the historical
  // ct_ppo behavior; TA-Lib's own default is SMA).
  PPO(int fastPeriod, int slowPeriod, MAType type = MAType::EMA);
  void calculate(const float *input, float *output,
                 int size, cudaStream_t stream = 0) noexcept(false) override;

private:
  int fastPeriod;
  int slowPeriod;
  MAType type;
};

} // namespace tacuda

#endif
