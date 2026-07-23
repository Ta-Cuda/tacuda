/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLMarubozu) {
  const int N = 2;
  std::vector<float> open{1.0f, 10.0f};
  std::vector<float> high{1.2f, 12.0f};
  std::vector<float> low{0.9f, 10.0f};
  std::vector<float> close{1.1f, 12.0f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_marubozu(open.data(), high.data(), low.data(),
                                  close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_marubozu failed";
  auto ref = cdl_ref(open, high, low, close, "CDLMARUBOZU");
  expect_approx_equal(out, ref);
}

