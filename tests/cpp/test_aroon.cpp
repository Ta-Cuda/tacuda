/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, Aroon) {
  std::vector<float> high = {1.f, 2.f, 3.f, 2.f, 3.f, 4.f, 5.f, 4.f, 6.f, 7.f};
  std::vector<float> low = {0.5f, 1.5f, 2.5f, 1.5f, 2.5f,
                            3.5f, 4.5f, 3.5f, 5.5f, 6.5f};
  const int N = high.size();
  std::vector<float> up(N, 0.0f), down(N, 0.0f), osc(N, 0.0f);
  int pUp = 5, pDown = 5;
  ctStatus_t rc = ct_aroon(high.data(), low.data(), up.data(), down.data(),
                           osc.data(), N, pUp, pDown);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_aroon failed";

  std::vector<float> refUp(N, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> refDown(N, std::numeric_limits<float>::quiet_NaN());
  for (int i = pUp; i < N; ++i) {
    int sinceHigh = 0;
    float maxVal = high[i];
    for (int j = 1; j <= pUp; ++j) {
      float val = high[i - j];
      if (val > maxVal) {  // strict: ties pick the newest bar (TA-Lib)
        maxVal = val;
        sinceHigh = j;
      }
    }
    refUp[i] = 100.0f * (pUp - sinceHigh) / pUp;
  }
  for (int i = pDown; i < N; ++i) {
    int sinceLow = 0;
    float minVal = low[i];
    for (int j = 1; j <= pDown; ++j) {
      float val = low[i - j];
      if (val < minVal) {
        minVal = val;
        sinceLow = j;
      }
    }
    refDown[i] = 100.0f * (pDown - sinceLow) / pDown;
  }
  std::vector<float> refOsc(N, std::numeric_limits<float>::quiet_NaN());
  for (int i = std::max(pUp, pDown); i < N; ++i)
    refOsc[i] = refUp[i] - refDown[i];

  expect_approx_equal(up, refUp);
  expect_approx_equal(down, refDown);
  expect_approx_equal(osc, refOsc);

  for (int i = 0; i < pUp; ++i)
    EXPECT_TRUE(std::isnan(up[i])) << "expected NaN at head " << i;
  for (int i = 0; i < pDown; ++i)
    EXPECT_TRUE(std::isnan(down[i])) << "expected NaN at head " << i;
  for (int i = 0; i < std::max(pUp, pDown); ++i)
    EXPECT_TRUE(std::isnan(osc[i])) << "expected NaN at head " << i;
}
