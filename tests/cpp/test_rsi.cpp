/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include "test_utils.hpp"
#include <cuda_runtime.h>

TEST(Tacuda, RSI) {
  const int N = 128;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.05f * i);

  std::vector<float> out(N, 0.0f);

  int p = 14;
  ctStatus_t rc = ct_rsi(x.data(), out.data(), N, p);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS) << "ct_rsi failed";

  auto ref = rsi_ref(x, p);
  expect_approx_equal(out, ref);
}

TEST(Tacuda, RSI_LargePeriod) {
  const int N = 200000;
  std::vector<float> x(N);
  for (int i = 0; i < N; ++i)
    x[i] = std::sin(0.001f * i);

  std::vector<float> out(N, 0.0f);
  int p = 10000;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  ctStatus_t rc = ct_rsi(x.data(), out.data(), N, p);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  EXPECT_LT(ms, 200.0f);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  ASSERT_EQ(rc, CT_STATUS_SUCCESS);

  // Wilder RSI reference (matches TA-Lib)
  std::vector<float> ref(N, std::numeric_limits<float>::quiet_NaN());
  if (N > p) {
    float avgGain = 0.0f, avgLoss = 0.0f;
    for (int i = 1; i <= p; ++i) {
      float diff = x[i] - x[i - 1];
      if (diff > 0.0f) avgGain += diff;
      else avgLoss -= diff;
    }
    avgGain /= static_cast<float>(p);
    avgLoss /= static_cast<float>(p);

    auto rsi_from = [](float g, float l) -> float {
      // TA-Lib: rsi = 100*g/(g+l); a fully flat window yields 0.
      if (l == 0.0f) return (g == 0.0f) ? 0.0f : 100.0f;
      float rs = g / l;
      return 100.0f - 100.0f / (1.0f + rs);
    };

    ref[p] = rsi_from(avgGain, avgLoss);
    for (int i = p + 1; i < N; ++i) {
      float diff = x[i] - x[i - 1];
      float g = (diff > 0.0f) ? diff : 0.0f;
      float l = (diff < 0.0f) ? -diff : 0.0f;
      avgGain = (avgGain * (p - 1) + g) / static_cast<float>(p);
      avgLoss = (avgLoss * (p - 1) + l) / static_cast<float>(p);
      ref[i] = rsi_from(avgGain, avgLoss);
    }
  }

  expect_approx_equal(out, ref, 2e-3f);
}
