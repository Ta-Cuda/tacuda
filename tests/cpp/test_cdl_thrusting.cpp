/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLThrusting) {
  const int N = 3;
  std::vector<float> open{10.0f, 8.8f, 9.5f};
  std::vector<float> high{10.5f, 9.6f, 9.7f};
  std::vector<float> low{9.0f, 8.5f, 9.3f};
  std::vector<float> close{9.2f, 9.4f, 9.6f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_thrusting(open.data(), high.data(), low.data(),
                                   close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_thrusting failed";
  auto ref = cdl_ref(open, high, low, close, "CDLTHRUSTING");
  expect_approx_equal(out, ref);
}

