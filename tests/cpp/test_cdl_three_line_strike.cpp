/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLThreeLineStrike) {
  const int N = 5;
  std::vector<float> open{10.0f, 10.8f, 11.3f, 12.2f, 9.6f};
  std::vector<float> high{11.1f, 11.6f, 12.1f, 12.3f, 9.7f};
  std::vector<float> low{9.9f, 10.7f, 11.2f, 9.4f, 9.5f};
  std::vector<float> close{11.0f, 11.5f, 12.0f, 9.5f, 9.6f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_three_line_strike(open.data(), high.data(), low.data(),
                                           close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_three_line_strike failed";
  auto ref = cdl_ref(open, high, low, close, "CDL3LINESTRIKE");
  expect_approx_equal(out, ref);
}

