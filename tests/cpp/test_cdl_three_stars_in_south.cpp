/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLThreeStarsInSouth) {
  const int N = 5;
  std::vector<float> open{10.0f, 8.5f, 7.8f, 7.8f, 7.8f};
  std::vector<float> high{10.1f, 8.6f, 7.9f, 7.9f, 7.9f};
  std::vector<float> low{6.0f, 7.0f, 7.2f, 7.7f, 7.7f};
  std::vector<float> close{8.0f, 7.5f, 7.4f, 7.8f, 7.8f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_three_stars_in_south(open.data(), high.data(), low.data(),
                                              close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_three_stars_in_south failed";
  auto ref = cdl_ref(open, high, low, close, "CDL3STARSINSOUTH");
  expect_approx_equal(out, ref);
}

