/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLMachingLow) {
  const int N = 2;
  std::vector<float> open{5.0f, 4.5f};
  std::vector<float> high{5.2f, 4.6f};
  std::vector<float> low{3.0f, 3.2f};
  std::vector<float> close{3.5f, 3.5f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_matching_low(open.data(), high.data(), low.data(),
                                      close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_matching_low failed";
  auto ref = cdl_ref(open, high, low, close, "CDLMATCHINGLOW");
  expect_approx_equal(out, ref);
}

