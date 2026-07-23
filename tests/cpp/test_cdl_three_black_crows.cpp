/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLThreeBlackCrows) {
  const int N = 5;
  std::vector<float> open{10.0f, 9.5f, 9.0f, 7.6f, 7.6f};
  std::vector<float> high{10.1f, 9.6f, 9.1f, 7.7f, 7.7f};
  std::vector<float> low{8.9f, 8.4f, 7.4f, 7.5f, 7.5f};
  std::vector<float> close{9.0f, 8.5f, 7.5f, 7.6f, 7.6f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_three_black_crows(open.data(), high.data(), low.data(),
                                           close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_three_black_crows failed";
  auto ref = cdl_ref(open, high, low, close, "CDL3BLACKCROWS");
  expect_approx_equal(out, ref);
}

