/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <limits>
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLConcealBabySwallow) {
  const int N = 5;
  std::vector<float> open{10.0f, 9.0f, 8.0f, 7.0f, 6.0f};
  std::vector<float> high{10.2f, 9.2f, 8.2f, 7.2f, 6.2f};
  std::vector<float> low{9.5f, 8.5f, 7.5f, 6.5f, 5.5f};
  std::vector<float> close{9.7f, 8.7f, 7.7f, 6.7f, 6.3f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_conceal_baby_swallow(
      open.data(), high.data(), low.data(), close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_conceal_baby_swallow failed";
  auto ref = cdl_ref(open, high, low, close, "CDLCONCEALBABYSWALLOW");
  expect_approx_equal(out, ref);
}
