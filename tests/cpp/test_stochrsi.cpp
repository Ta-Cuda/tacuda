/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <tacuda.h>

TEST(Tacuda, StochRSI) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> k(N, 0.0f), d(N, 0.0f);

  int rsiP = 14, kP = 5, dP = 3;
  ctStatus_t rc = ct_stochrsi(x.data(), k.data(), d.data(), N, rsiP, kP, dP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_stochrsi failed";

  auto [refK, refD] = stochrsi_ref(x, rsiP, kP, dP);
  expect_approx_equal(k, refK);
  expect_approx_equal(d, refD);
  // TA-Lib publishes K and D with a common outBegIdx:
  // both start at rsiP + kP + dP - 2.
  for (int i = 0; i < rsiP + kP + dP - 2; ++i) {
    EXPECT_TRUE(std::isnan(k[i])) << "expected NaN for K at head " << i;
    EXPECT_TRUE(std::isnan(d[i])) << "expected NaN for D at head " << i;
  }
}
