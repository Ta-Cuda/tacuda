/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, EMA) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, std::numeric_limits<float>::quiet_NaN());

  int p = 5;
  ctStatus_t rc = ct_ema(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_ema failed";

  // First p-1 values should be NaN (warmup period)
  for (int i = 0; i < p - 1; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at warmup " << i;
  }

  // TA-Lib style: seed with SMA of first p values, then recursive EMA
  float sum = 0.0f;
  for (int i = 0; i < p; ++i) sum += x[i];
  ref[p - 1] = sum / p;
  const float k = 2.0f / (p + 1.0f);
  for (int i = p; i < N; ++i) {
    ref[i] = k * x[i] + (1.0f - k) * ref[i - 1];
  }
  expect_approx_equal(out, ref);
}
