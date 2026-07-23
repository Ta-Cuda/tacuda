/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <cmath>

TEST(Tacuda, MAXINDEX) {
  const int N = 32;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.1f * i);
  std::vector<float> out(N, 0.0f), ref(N, std::numeric_limits<float>::quiet_NaN());
  int p = 5;
  ctStatus_t rc = ct_maxindex(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  // First p-1 values should be NaN (warmup period)
  for (int i = 0; i < p - 1; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at warmup " << i;
  }
  // TA-Lib-compatible MAXINDEX returns absolute index in input series;
  // ties pick the most recent bar (sliding-update semantics).
  for (int i = 0; i <= N - p; ++i) {
    float maxVal = x[i];
    int maxIdx = 0;
    for (int k = 1; k < p; ++k) {
      if (x[i + k] >= maxVal) {
        maxVal = x[i + k];
        maxIdx = k;
      }
    }
    ref[i + p - 1] = static_cast<float>(i + maxIdx);
  }
  expect_approx_equal(out, ref);
}
