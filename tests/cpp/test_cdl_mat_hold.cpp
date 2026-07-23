/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <limits>
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLMatHold) {
  const int N = 5;
  std::vector<float> open{10.0f, 11.2f, 11.0f, 10.9f, 10.86f};
  std::vector<float> high{11.0f, 11.4f, 11.1f, 11.0f, 11.5f};
  std::vector<float> low{9.5f, 10.9f, 10.7f, 10.6f, 10.7f};
  std::vector<float> close{10.8f, 11.0f, 10.9f, 10.85f, 11.4f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_mat_hold(open.data(), high.data(), low.data(),
                                  close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_mat_hold failed";
  auto ref = cdl_ref(open, high, low, close, "CDLMATHOLD");
  expect_approx_equal(out, ref);
}
