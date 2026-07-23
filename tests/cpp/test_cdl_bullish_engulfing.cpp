/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLBullishEngulfing) {
  const int N = 4;
  std::vector<float> open{10.0f, 9.5f, 9.1f, 9.3f};
  std::vector<float> high{10.2f, 9.7f, 9.9f, 9.5f};
  std::vector<float> low{9.7f, 9.1f, 9.0f, 8.9f};
  std::vector<float> close{9.8f, 9.2f, 9.8f, 9.1f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_bullish_engulfing(open.data(), high.data(), low.data(),
                                           close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_bullish_engulfing failed";
  auto ref = cdl_ref(open, high, low, close, "CDLENGULFING");
  expect_approx_equal(out, ref);
}
