/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <limits>
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLDojiStar) {
  const int N = 4;
  std::vector<float> open{1.0f, 1.0f, 1.6f, 1.5f};
  std::vector<float> high{1.1f, 1.5f, 1.65f, 1.6f};
  std::vector<float> low{0.9f, 0.95f, 1.55f, 1.45f};
  std::vector<float> close{1.05f, 1.4f, 1.6f, 1.55f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_doji_star(open.data(), high.data(), low.data(),
                                   close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_doji_star failed";
  auto ref = cdl_ref(open, high, low, close, "CDLDOJISTAR");
  expect_approx_equal(out, ref);
}
