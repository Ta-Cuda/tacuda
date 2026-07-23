/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <limits>
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLInNeck) {
  const int N = 3;
  std::vector<float> open{10.0f, 8.8f, 9.0f};
  std::vector<float> high{10.1f, 9.1f, 9.2f};
  std::vector<float> low{8.9f, 8.7f, 8.8f};
  std::vector<float> close{9.0f, 9.05f, 9.1f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_in_neck(open.data(), high.data(), low.data(),
                                 close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_in_neck failed";
  auto ref = cdl_ref(open, high, low, close, "CDLINNECK");
  expect_approx_equal(out, ref);
}
