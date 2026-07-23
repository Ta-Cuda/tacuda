/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <cmath>
#include <tacuda.h>
#include <vector>
#include <gtest/gtest.h>

TEST(Tacuda, WILLRRange) {
  const int N = 100;
  const int P = 6;
  std::vector<float> high(N), low(N), close(N);
  for (int i = 0; i < N; ++i) {
    high[i] = 10.0f + i * 0.1f;
    low[i] = high[i] - 2.0f;
    close[i] = low[i] + 1.0f;
  }
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc =
      ct_willr(high.data(), low.data(), close.data(), out.data(), N, P);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  for (int i = 0; i < N; ++i) {
    if (i < P - 1) {
      EXPECT_TRUE(std::isnan(out[i]));
    } else {
      EXPECT_LE(out[i], 0.0f);
      EXPECT_GE(out[i], -100.0f);
    }
  }
}
