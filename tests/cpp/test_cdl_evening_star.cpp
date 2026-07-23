/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <limits>
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLEveningStar) {
  const int N = 5;
  std::vector<float> open{1.0f, 1.7f, 1.7f, 1.5f, 1.5f};
  std::vector<float> high{1.6f, 1.75f, 1.8f, 1.6f, 1.6f};
  std::vector<float> low{0.9f, 1.65f, 1.1f, 1.4f, 1.4f};
  std::vector<float> close{1.5f, 1.72f, 1.2f, 1.55f, 1.55f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_evening_star(open.data(), high.data(), low.data(),
                                      close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_evening_star failed";
  auto ref = cdl_ref(open, high, low, close, "CDLEVENINGSTAR");
  expect_approx_equal(out, ref);
}
