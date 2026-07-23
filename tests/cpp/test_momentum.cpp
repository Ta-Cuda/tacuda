/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, Momentum) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f), ref(N, std::numeric_limits<float>::quiet_NaN());

  int p = 5;
  ctStatus_t rc = ct_momentum(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_momentum failed";
  // First p values should be NaN (warmup period)
  for (int i = 0; i < p; i++) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at warmup " << i;
  }
  // Compute reference values starting from index p
  for (int i = 0; i < N - p; i++)
    ref[i + p] = x[i + p] - x[i];  // Write at index i + p
  expect_approx_equal(out, ref);
}
