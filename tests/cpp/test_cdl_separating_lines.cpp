/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "test_utils.hpp"
#include <tacuda.h>
#include <vector>

TEST(Tacuda, CDLSeparatingLines) {
  const int N = 3;
  std::vector<float> open{2.0f, 2.0f, 3.0f};
  std::vector<float> high{2.1f, 2.3f, 3.2f};
  std::vector<float> low{1.8f, 1.95f, 2.8f};
  std::vector<float> close{1.9f, 2.2f, 3.1f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_separating_lines(open.data(), high.data(), low.data(),
                                          close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_separating_lines failed";
  auto ref = cdl_ref(open, high, low, close, "CDLSEPARATINGLINES");
  expect_approx_equal(out, ref);
}
