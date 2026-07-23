/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, Correl) {
  const int N = 128;
  std::vector<float> x(N), y(N);
  for (int i = 0; i < N; ++i) {
    x[i] = std::sin(0.01f * i);
    y[i] = std::cos(0.03f * i);
  }
  std::vector<float> out(N, 0.0f), ref(N, std::numeric_limits<float>::quiet_NaN());
  int p = 5;
  ctStatus_t rc = ct_correl(x.data(), y.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_correl failed";
  // First p-1 values should be NaN (warmup period)
  for (int i = 0; i < p - 1; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at warmup " << i;
  }
  // Compute reference values starting from index p-1
  for (int i = 0; i <= N - p; ++i) {
    float sumX = 0.0f, sumY = 0.0f;
    for (int j = 0; j < p; ++j) {
      sumX += x[i + j];
      sumY += y[i + j];
    }
    float meanX = sumX / p;
    float meanY = sumY / p;
    float cov = 0.0f, varX = 0.0f, varY = 0.0f;
    for (int j = 0; j < p; ++j) {
      float dx = x[i + j] - meanX;
      float dy = y[i + j] - meanY;
      cov += dx * dy;
      varX += dx * dx;
      varY += dy * dy;
    }
    float denom = std::sqrt(varX * varY);
    ref[i + p - 1] = (denom == 0.0f) ? 0.0f : cov / denom;  // Write at index i + p - 1
  }
  expect_approx_equal(out, ref);
}
