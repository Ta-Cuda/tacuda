/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <cuda_runtime.h>

TEST(Tacuda, MACD_LineLargePeriod) {
  const int N = 200000;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.001f * i);

  int fastP = 5000, slowP = 10000;
  std::vector<float> out(N, 0.0f);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  ctStatus_t rc = ct_macd_line(x.data(), out.data(), N, fastP, slowP);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  EXPECT_LT(ms, 200.0f);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);

  for (int i = 0; i < slowP - 1; ++i)
    EXPECT_TRUE(std::isnan(out[i]));
  for (int i = slowP - 1; i < N; ++i)
    EXPECT_TRUE(std::isfinite(out[i]));
}

