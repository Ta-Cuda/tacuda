/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLRickshawMan) {
  const int N = 3;
  std::vector<float> open{1.0f, 2.0f, 3.0f};
  std::vector<float> high{1.1f, 3.0f, 3.2f};
  std::vector<float> low{0.9f, 1.0f, 2.8f};
  std::vector<float> close{1.05f, 2.05f, 3.1f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_rickshaw_man(open.data(), high.data(), low.data(),
                                      close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_rickshaw_man failed";
  auto ref = cdl_ref(open, high, low, close, "CDLRICKSHAWMAN");
  expect_approx_equal(out, ref);
}
