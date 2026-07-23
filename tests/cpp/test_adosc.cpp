/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, ADOSC) {
  std::vector<float> high = {12.f, 12.5f, 13.f, 13.5f, 14.f, 14.5f};
  std::vector<float> low = {11.f, 11.5f, 12.f, 12.5f, 13.f, 13.5f};
  std::vector<float> close = {11.5f, 12.f, 12.5f, 13.f, 13.5f, 14.f};
  std::vector<float> volume = {100.f, 110.f, 120.f, 130.f, 140.f, 150.f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  int shortP = 3, longP = 5;
  ctStatus_t rc = ct_adosc(high.data(), low.data(), close.data(), volume.data(),
                           out.data(), N, shortP, longP);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_adosc failed";
  auto ref = adosc_ref(high, low, close, volume, shortP, longP);
  expect_approx_equal(out, ref);
  // TA-Lib ADOSC lookback = slowest EMA lookback = longP - 1.
  for (int i = 0; i < longP - 1; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
  }
  EXPECT_FALSE(std::isnan(out[longP - 1])) << "first valid bar must be longP-1";
}
