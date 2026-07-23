/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>
#include <limits>

TEST(Tacuda, CDLUnique3River) {
  const int N = 5;
  std::vector<float> open{10.0f, 9.6f, 8.8f, 9.1f, 9.2f};
  std::vector<float> high{10.5f, 9.7f, 9.0f, 9.3f, 9.4f};
  std::vector<float> low{8.9f, 8.5f, 8.6f, 8.9f, 9.0f};
  std::vector<float> close{9.2f, 9.4f, 9.0f, 9.2f, 9.1f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_unique_3_river(open.data(), high.data(), low.data(),
                                        close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_unique_3_river failed";
  auto ref = cdl_ref(open, high, low, close, "CDLUNIQUE3RIVER");
  expect_approx_equal(out, ref);
}

