/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <tacuda.h>

TEST(Tacuda, TRIMA) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f),
      ref(N, std::numeric_limits<float>::quiet_NaN()),
      tmp(N, std::numeric_limits<float>::quiet_NaN());

  int p = 5;
  ctStatus_t rc = ct_trima(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_trima failed";
  int p1 = (p + 1) / 2;
  int p2 = (p % 2 == 0) ? (p1 + 1) : p1;

  // TA-Lib semantics: first valid value at index (p-1)
  for (int i = 0; i < p - 1; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at warmup " << i;
  }

  // Triangular MA = SMA(SMA(x, p1), p2), end-of-window aligned (like TA-Lib)
  for (int t = p1 - 1; t < N; ++t) {
    float s = 0.0f;
    for (int j = 0; j < p1; ++j) {
      s += x[t - j];
    }
    tmp[t] = s / p1;
  }
  const int firstValid = (p1 - 1) + (p2 - 1);
  for (int t = firstValid; t < N; ++t) {
    float s = 0.0f;
    for (int j = 0; j < p2; ++j) {
      s += tmp[t - j];
    }
    ref[t] = s / p2;
  }
  expect_approx_equal(out, ref);
}
