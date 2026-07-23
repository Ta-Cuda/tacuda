/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <cmath>

TEST(Tacuda, MIDPRICE) {
  const int N = 40;
  std::vector<float> high(N), low(N);
  for (int i = 0; i < N; ++i) {
    high[i] = 10.0f + std::sin(0.1f * i);
    low[i] = high[i] - 1.0f;
  }
  std::vector<float> out(N, 0.0f), ref(N, std::numeric_limits<float>::quiet_NaN());
  int p = 5;
  ctStatus_t rc = ct_midprice(high.data(), low.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  // First p-1 values should be NaN (warmup period)
  for (int i = 0; i < p - 1; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at warmup " << i;
  }
  // Compute reference values starting from index p-1
  for (int i = 0; i <= N - p; ++i) {
    float maxH = high[i];
    float minL = low[i];
    for (int k = 1; k < p; ++k) {
      maxH = std::max(maxH, high[i + k]);
      minL = std::min(minL, low[i + k]);
    }
    ref[i + p - 1] = 0.5f * (maxH + minL);  // Write at index i + p - 1
  }
  expect_approx_equal(out, ref);
}
