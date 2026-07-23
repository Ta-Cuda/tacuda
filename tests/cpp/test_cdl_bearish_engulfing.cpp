/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLBearishEngulfing) {
  const int N = 4;
  std::vector<float> open{10.0f, 10.5f, 10.9f, 10.8f};
  std::vector<float> high{10.4f, 11.0f, 11.1f, 11.0f};
  std::vector<float> low{9.8f, 10.4f, 10.0f, 10.5f};
  std::vector<float> close{10.2f, 10.8f, 10.2f, 10.6f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_bearish_engulfing(open.data(), high.data(), low.data(),
                                           close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_bearish_engulfing failed";
  auto ref = cdl_ref(open, high, low, close, "CDLENGULFING");
  expect_approx_equal(out, ref);
}
