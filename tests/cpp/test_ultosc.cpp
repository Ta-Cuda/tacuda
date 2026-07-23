/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, ULTOSC) {
  std::vector<float> high = {30.0f, 32.0f, 31.0f, 33.0f, 34.0f, 35.0f,
                             36.0f, 37.0f, 36.0f, 38.0f, 39.0f, 40.0f};
  std::vector<float> low = {29.0f, 30.0f, 30.0f, 31.0f, 32.0f, 33.0f,
                            34.0f, 35.0f, 34.0f, 35.0f, 36.0f, 37.0f};
  std::vector<float> close = {29.5f, 31.0f, 30.5f, 32.0f, 33.0f, 34.0f,
                              35.0f, 36.0f, 35.0f, 37.0f, 38.0f, 39.0f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  int sp = 3, mp = 5, lp = 7;
  ctStatus_t rc =
      ct_ultosc(high.data(), low.data(), close.data(), out.data(), N, sp, mp, lp);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_ultosc failed";
  auto ref = ultosc_ref(high, low, close, sp, mp, lp);
  expect_approx_equal(out, ref);
  for (int i = 0; i < lp; ++i) {
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
  }
}
