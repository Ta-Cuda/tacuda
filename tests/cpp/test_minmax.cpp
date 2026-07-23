/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <cmath>
#include <tacuda.h>

TEST(Tacuda, MINMAX) {
  const int N = 32;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.1f * i);
  std::vector<float> outMin(N, 0.0f), outMax(N, 0.0f);
  std::vector<float> refMin(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> refMax(N, std::numeric_limits<float>::quiet_NaN());
  int p = 5;
  ctStatus_t rc = ct_minmax(x.data(), outMin.data(), outMax.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  // TA-Lib convention: window [end-p+1, end] reported at `end`, leading
  // warm-up of p-1 NaNs.
  for (int end = p - 1; end < N; ++end) {
    float mn = x[end];
    float mx = x[end];
    for (int k = 1; k < p; ++k) {
      mn = std::min(mn, x[end - k]);
      mx = std::max(mx, x[end - k]);
    }
    refMin[end] = mn;
    refMax[end] = mx;
  }
  expect_approx_equal(outMin, refMin);
  expect_approx_equal(outMax, refMax);
  for (int i = 0; i < p - 1; ++i) {
    EXPECT_TRUE(std::isnan(outMin[i]));
    EXPECT_TRUE(std::isnan(outMax[i]));
  }
}
