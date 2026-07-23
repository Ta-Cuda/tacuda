/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <vector>

TEST(Tacuda, CDLGravestoneDoji) {
  const int N = 4;
  std::vector<float> open{1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> high{1.2f, 2.2f, 3.5f, 4.3f};
  std::vector<float> low{0.9f, 1.8f, 3.0f, 3.7f};
  std::vector<float> close{1.1f, 2.1f, 3.0f, 4.1f};
  std::vector<float> out(N, 0.0f);
  ctStatus_t rc = ct_cdl_gravestone_doji(open.data(), high.data(), low.data(),
                                         close.data(), out.data(), N);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_cdl_gravestone_doji failed";
  auto ref = cdl_ref(open, high, low, close, "CDLGRAVESTONEDOJI");
  expect_approx_equal(out, ref);
}

