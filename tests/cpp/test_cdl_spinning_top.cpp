/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>

TEST(Tacuda, CDLSpinningTop) {
  const int N = 3;
  std::vector<float> open{1.0f, 2.0f, 3.0f};
  std::vector<float> high{1.4f, 2.6f, 3.5f};
  std::vector<float> low{0.8f, 1.4f, 2.6f};
  std::vector<float> close{1.3f, 2.1f, 3.4f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_spinning_top(open.data(), high.data(), low.data(),
                                      close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_spinning_top failed";
  auto ref = cdl_ref(open, high, low, close, "CDLSPINNINGTOP");
  expect_approx_equal(out, ref);
}

