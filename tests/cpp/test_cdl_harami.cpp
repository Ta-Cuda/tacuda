/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLHarami) {
  const int N = 4;
  std::vector<float> open{10.0f, 8.5f, 9.3f, 9.5f};
  std::vector<float> high{10.5f, 9.3f, 9.7f, 9.8f};
  std::vector<float> low{7.0f, 8.4f, 9.0f, 9.3f};
  std::vector<float> close{8.0f, 9.1f, 9.4f, 9.4f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_harami(open.data(), high.data(), low.data(),
                                 close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_harami failed";
  auto ref = cdl_ref(open, high, low, close, "CDLHARAMI");
  expect_approx_equal(out, ref);
}

