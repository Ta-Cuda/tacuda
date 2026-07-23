/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace tacuda::detail {

// Reference-style EMA-at-index used by existing APO/PPO tests.
// Note: This is intentionally the same definition as tests/cpp/test_utils.cpp
// (ema_at_ref), to keep ct_apo/ct_ppo parity with the test reference.
inline float ema_at_ref(const std::vector<float> &x, int idx, int period, int start = 0) {
  const float k = 2.0f / (static_cast<float>(period) + 1.0f);
  float weight = 1.0f;
  float weightedSum = x[idx];
  float weightSum = 1.0f;
  int steps = std::min(period, idx - start);
  for (int i = 1; i <= steps; ++i) {
    weight *= (1.0f - k);
    weightedSum += x[idx - i] * weight;
    weightSum += weight;
  }
  return weightedSum / weightSum;
}

inline void seed_apo_from_input(const std::vector<float> &in, int fastPeriod, int slowPeriod,
                               std::vector<float> &out) {
  const int n = static_cast<int>(in.size());
  if (static_cast<int>(out.size()) != n) {
    out.resize(static_cast<size_t>(n));
  }
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::fill(out.begin(), out.end(), nan);

  for (int i = slowPeriod; i < n; ++i) {
    float emaFast = ema_at_ref(in, i, fastPeriod);
    float emaSlow = ema_at_ref(in, i, slowPeriod);
    out[i] = emaFast - emaSlow;
  }
}

inline void seed_ppo_from_input(const std::vector<float> &in, int fastPeriod, int slowPeriod,
                               std::vector<float> &out) {
  const int n = static_cast<int>(in.size());
  if (static_cast<int>(out.size()) != n) {
    out.resize(static_cast<size_t>(n));
  }
  const float nan = std::numeric_limits<float>::quiet_NaN();
  std::fill(out.begin(), out.end(), nan);

  for (int i = slowPeriod; i < n; ++i) {
    float emaFast = ema_at_ref(in, i, fastPeriod);
    float emaSlow = ema_at_ref(in, i, slowPeriod);
    out[i] = (emaSlow == 0.0f) ? 0.0f : 100.0f * (emaFast - emaSlow) / emaSlow;
  }
}

} // namespace tacuda::detail

