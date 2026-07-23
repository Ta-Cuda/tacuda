/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"

TEST(Tacuda, MACD_Ext) {
  const int N = 64;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);
  std::vector<float> macd(N, 0.0f), signal(N, 0.0f), hist(N, 0.0f);
  int fastP = 3, slowP = 8, signalP = 4;
  ctStatus_t rc = ct_macd(x.data(), macd.data(), signal.data(), hist.data(), N,
                          fastP, slowP, signalP, CT_MA_EMA);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  
  auto [macdRef, sigRef, histRef] = macd_ref(x, fastP, slowP, signalP);
  
  expect_approx_equal(macd, macdRef);
  expect_approx_equal(signal, sigRef);
  expect_approx_equal(hist, histRef);
}

TEST(Tacuda, MACD_ExtLargePeriod) {
  const int N = 200000;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.001f * i);

  int fastP = 5000, slowP = 10000, signalP = 4000;
  std::vector<float> macd(N, 0.0f), signal(N, 0.0f), hist(N, 0.0f);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  ctStatus_t rc = ct_macd(x.data(), macd.data(), signal.data(), hist.data(), N,
                          fastP, slowP, signalP, CT_MA_EMA);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  EXPECT_LT(ms, 400.0f);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);

  auto [macdRef, sigRef, histRef] = macd_ref(x, fastP, slowP, signalP);
  expect_approx_equal(macd, macdRef, 1e-3f);
  expect_approx_equal(signal, sigRef, 1e-3f);
  expect_approx_equal(hist, histRef, 1e-3f);
  for (int i = 0; i < slowP - 1; ++i) {
    EXPECT_TRUE(std::isnan(macd[i]));
    EXPECT_TRUE(std::isnan(signal[i]));
    EXPECT_TRUE(std::isnan(hist[i]));
  }
}

TEST(Tacuda, MACD_ExtBoundaryEqualsSlowPeriod) {
  const int slowP = 12;
  const int fastP = 5;
  const int signalP = 4;
  const int N = slowP;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::cos(0.2f * i);

  std::vector<float> macd(N, 0.0f), signal(N, 0.0f), hist(N, 0.0f);
  ctStatus_t rc = ct_macd(x.data(), macd.data(), signal.data(), hist.data(), N,
                          fastP, slowP, signalP, CT_MA_EMA);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);
  auto [macdRef, sigRef, histRef] = macd_ref(x, fastP, slowP, signalP);
  expect_approx_equal(macd, macdRef);
  expect_approx_equal(signal, sigRef);
  expect_approx_equal(hist, histRef);
}
