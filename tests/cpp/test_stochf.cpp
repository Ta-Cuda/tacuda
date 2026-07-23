/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, StochF) {
  std::vector<float> high = {48.70f, 48.72f, 48.90f, 48.87f, 48.82f,
                             49.05f, 49.20f, 49.35f, 49.92f, 50.19f,
                             50.12f, 49.66f, 49.88f, 50.19f, 50.36f};
  std::vector<float> low = {47.79f, 48.14f, 48.39f, 48.37f, 48.24f,
                            48.64f, 48.94f, 48.86f, 49.50f, 49.87f,
                            49.20f, 48.90f, 49.43f, 49.73f, 49.26f};
  std::vector<float> close = {48.16f, 48.61f, 48.75f, 48.63f, 48.74f,
                              49.03f, 49.07f, 49.32f, 49.91f, 49.91f,
                              49.40f, 49.50f, 49.75f, 49.87f, 50.13f};
  const int N = high.size();
  std::vector<float> k(N, 0.0f), d(N, 0.0f),
      refK(N, std::numeric_limits<float>::quiet_NaN()),
      refD(N, std::numeric_limits<float>::quiet_NaN());

  int kP = 5, dP = 3;
  ctStatus_t rc = ct_stochf(high.data(), low.data(), close.data(), k.data(), d.data(), N, kP, dP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_stochf failed";

  std::vector<float> rawK(N, std::numeric_limits<float>::quiet_NaN());
  for (int i = kP - 1; i < N; ++i) {
    float highest = high[i];
    float lowest = low[i];
    for (int j = 1; j < kP; ++j) {
      float h = high[i - j];
      float l = low[i - j];
      if (h > highest) highest = h;
      if (l < lowest)  lowest = l;
    }
    float denom = highest - lowest;
    rawK[i] = denom == 0.0f ? 0.0f : (close[i] - lowest) / denom * 100.0f;
  }
  // TA-Lib publishes K and D with a common outBegIdx: the K line loses the
  // D warm-up as well.
  int start = kP + dP - 2;
  for (int i = start; i < N; ++i) {
    refK[i] = rawK[i];
    float sum = 0.0f;
    for (int j = 0; j < dP; ++j)
      sum += rawK[i - j];
    refD[i] = sum / dP;
  }

  expect_approx_equal(k, refK);
  expect_approx_equal(d, refD);
  for (int i = 0; i < start; ++i) {
    EXPECT_TRUE(std::isnan(k[i])) << "expected NaN at head " << i;
    EXPECT_TRUE(std::isnan(d[i])) << "expected NaN at head " << i;
  }
}
