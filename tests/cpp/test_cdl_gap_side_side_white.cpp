/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLGapSideSideWhite) {
  const int N = 4;
  std::vector<float> open{10.0f, 11.0f, 11.0f, 12.0f};
  std::vector<float> high{10.5f, 11.6f, 11.4f, 12.2f};
  std::vector<float> low{9.5f, 11.0f, 11.0f, 11.8f};
  std::vector<float> close{10.4f, 11.5f, 11.3f, 11.9f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_gap_side_side_white(open.data(), high.data(), low.data(),
                                             close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_gap_side_side_white failed";
  auto ref = cdl_ref(open, high, low, close, "CDLGAPSIDESIDEWHITE");
  expect_approx_equal(out, ref);
}

