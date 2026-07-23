/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, StdDev) {
  const int N = 100;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::cos(0.1f * i);

  std::vector<float> out(N, 0.0f), ref(N, std::numeric_limits<float>::quiet_NaN());

  int p = 5;
  ctStatus_t rc = ct_stddev(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_stddev failed";
  for (int i = p - 1; i < N; ++i) {
    float mean = 0.0f;
    for (int j = 0; j < p; ++j)
      mean += x[i - j];
    mean /= p;
    float sumsq = 0.0f;
    for (int j = 0; j < p; ++j) {
      float d = x[i - j] - mean;
      sumsq += d * d;
    }
    ref[i] = std::sqrt(sumsq / p);
  }
  expect_approx_equal(out, ref);
  for (int i = 0; i < p - 1; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
}
