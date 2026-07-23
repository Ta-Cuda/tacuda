/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLStalledPattern) {
  const int N = 4;
  std::vector<float> open{1.0f, 2.1f, 3.3f, 4.0f};
  std::vector<float> high{2.2f, 3.5f, 3.6f, 4.6f};
  std::vector<float> low{0.8f, 2.0f, 3.2f, 3.8f};
  std::vector<float> close{2.0f, 3.2f, 3.4f, 4.5f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_stalled_pattern(open.data(), high.data(), low.data(),
                                         close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_stalled_pattern failed";
  auto ref = cdl_ref(open, high, low, close, "CDLSTALLEDPATTERN");
  expect_approx_equal(out, ref);
}

