/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <tacuda.h>

TEST(Tacuda, MINUS_DI) {
  std::vector<float> high = {30.0f, 32.0f, 31.0f, 33.0f, 34.0f, 35.0f,
                             36.0f, 37.0f, 36.0f, 38.0f, 39.0f, 40.0f};
  std::vector<float> low = {29.0f, 30.0f, 30.0f, 31.0f, 32.0f, 33.0f,
                            34.0f, 35.0f, 34.0f, 35.0f, 36.0f, 37.0f};
  std::vector<float> close = {29.5f, 31.0f, 30.5f, 32.0f, 33.0f, 34.0f,
                              35.0f, 36.0f, 35.0f, 37.0f, 38.0f, 39.0f};
  const int N = high.size();
  std::vector<float> out(N, 0.0f);
  int p = 3;
  ctStatus_t rc =
      ct_minus_di(high.data(), low.data(), close.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_minus_di failed";
  auto ref = minus_di_ref(high, low, close, p);
  expect_approx_equal(out, ref);
  for (int i = 0; i < p; ++i)
    EXPECT_TRUE(std::isnan(out[i])) << "expected NaN at head " << i;
}
