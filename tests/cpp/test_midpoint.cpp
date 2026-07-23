/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <cmath>

TEST(Tacuda, MIDPOINT) {
  const int N = 40;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::cos(0.1f * i);
  std::vector<float> out(N, 0.0f), ref(N, std::numeric_limits<float>::quiet_NaN());
  int p = 7;
  ctStatus_t rc = ct_midpoint(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  // First p-1 values should be NaN (warmup period)
  for (int i = 0; i < p - 1; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at warmup " << i;
  }
  // Compute reference values starting from index p-1
  for (int i = 0; i <= N - p; ++i) {
    float mn = x[i], mx = x[i];
    for (int k = 1; k < p; ++k) {
      float v = x[i + k];
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
    ref[i + p - 1] = 0.5f * (mx + mn);  // Write at index i + p - 1
  }
  expect_approx_equal(out, ref);
}
