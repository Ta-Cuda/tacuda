/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLBreakaway) {
  const int N = 5;
  std::vector<float> open{10.0f, 8.8f, 8.0f, 7.4f, 7.1f};
  std::vector<float> high{10.1f, 8.9f, 8.1f, 7.5f, 10.3f};
  std::vector<float> low{8.9f, 8.0f, 7.3f, 6.8f, 7.0f};
  std::vector<float> close{9.0f, 8.2f, 7.5f, 7.0f, 10.2f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_breakaway(open.data(), high.data(), low.data(),
                                   close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_breakaway failed";
  auto ref = cdl_ref(open, high, low, close, "CDLBREAKAWAY");
  expect_approx_equal(out, ref);
}

