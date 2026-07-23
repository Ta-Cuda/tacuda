/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLInvertedHammer) {
  const int N = 3;
  std::vector<float> open{1.0f, 2.0f, 3.0f};
  std::vector<float> high{1.2f, 2.7f, 3.4f};
  std::vector<float> low{0.8f, 1.98f, 2.9f};
  std::vector<float> close{1.1f, 2.05f, 3.2f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_inverted_hammer(open.data(), high.data(), low.data(),
                                         close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_inverted_hammer failed";
  auto ref = cdl_ref(open, high, low, close, "CDLINVERTEDHAMMER");
  expect_approx_equal(out, ref);
}
