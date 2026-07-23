/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <cmath>
#include <tacuda.h>

TEST(Tacuda, MINMAXINDEX) {
  const int N = 32;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.1f * i);
  std::vector<float> outMin(N, 0.0f), outMax(N, 0.0f);
  std::vector<float> refMin(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> refMax(N, std::numeric_limits<float>::quiet_NaN());
  int p = 5;
  ctStatus_t rc = ct_minmaxindex(x.data(), outMin.data(), outMax.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  for (int i = 0; i <= N - p; ++i) {
    float mn = x[i];
    float mx = x[i];
    int minIdx = 0;
    int maxIdx = 0;
    // Ties pick the most recent bar (TA-Lib sliding-update semantics).
    for (int k = 1; k < p; ++k) {
      float v = x[i + k];
      if (v <= mn) {
        mn = v;
        minIdx = k;
      }
      if (v >= mx) {
        mx = v;
        maxIdx = k;
      }
    }
    refMin[i + p - 1] = static_cast<float>(i + minIdx);
    refMax[i + p - 1] = static_cast<float>(i + maxIdx);
  }
  expect_approx_equal(outMin, refMin);
  expect_approx_equal(outMax, refMax);
  for (int i = 0; i < p - 1; ++i) {
    EXPECT_TRUE(std::isnan(outMin[i])) << "expected NaN at warmup " << i;
    EXPECT_TRUE(std::isnan(outMax[i])) << "expected NaN at warmup " << i;
  }
}
