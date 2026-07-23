/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <tacuda.h>

TEST(Tacuda, WclPrice) {
  const int N = 100;
  std::vector<float> high(N), low(N), close(N);
  for (int i = 0; i < N; ++i) {
    high[i] = 10.0f + std::sin(0.05f * i);
    low[i] = high[i] - 1.0f;
    close[i] = low[i] + 0.5f;
  }
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc =
      ct_wclprice(high.data(), low.data(), close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_wclprice failed";
  auto ref = wclprice_ref(high, low, close);
  expect_approx_equal(out, ref);
}
