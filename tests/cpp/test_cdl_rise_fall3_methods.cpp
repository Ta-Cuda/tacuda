/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLRISEFALL3METHODS) {
  const int N = 5;
  std::vector<float> open{1.0f, 2.0f, 1.9f, 1.8f, 1.9f};
  std::vector<float> high{2.2f, 2.1f, 2.0f, 1.9f, 2.4f};
  std::vector<float> low{0.8f, 1.7f, 1.6f, 1.5f, 1.8f};
  std::vector<float> close{2.0f, 1.9f, 1.8f, 1.7f, 2.3f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_rise_fall3_methods(
      open.data(), high.data(), low.data(), close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_rise_fall3_methods failed";
  auto ref = cdl_ref(open, high, low, close, "CDLRISEFALL3METHODS");
  expect_approx_equal(out, ref);
}
