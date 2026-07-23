/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLDoji) {
  const int N = 5;
  std::vector<float> open{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> high{1.2f, 2.2f, 3.2f, 4.2f, 5.2f};
  std::vector<float> low{0.8f, 1.8f, 2.8f, 3.8f, 4.8f};
  std::vector<float> close{1.1f, 2.1f, 3.01f, 4.3f, 4.9f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_doji(open.data(), high.data(), low.data(), close.data(),
                              out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_doji failed";
  auto ref = cdl_ref(open, high, low, close, "CDLDOJI");
  expect_approx_equal(out, ref);
}
