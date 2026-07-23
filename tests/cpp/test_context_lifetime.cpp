/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// Lifetime contracts of the Context / DeviceBuffer stream control block:
//  - buffers (float and dtype-aware) may safely outlive their Context;
//  - buffers created by another Context are rejected (their release is
//    ordered on the foreign stream, so mixing streams could recycle the
//    memory while kernels are still writing).
#include "test_utils.hpp"

#include <cuda_runtime.h>
#include <tacuda/Context.h>

#include <cmath>
#include <stdexcept>
#include <vector>

TEST(ContextLifetime, DeviceBufferOutlivesContext) {
  const int N = 512;
  std::vector<float> data(N);
  for (int i = 0; i < N; ++i) data[static_cast<size_t>(i)] = std::sin(0.05f * i);

  tacuda::DeviceBuffer result;
  {
    tacuda::Context ctx;
    auto buf = ctx.upload(data.data(), N);
    result = ctx.sma(buf, 14);
    // No explicit synchronize: the buffer must carry everything it needs.
  }  // ctx destroyed here; `result` still owns the stream via its guard

  std::vector<float> host(N, -1.0f);
  // The guard keeps the stream alive, so a sync on the default stream after
  // a plain device->host copy is enough to read the values.
  ASSERT_EQ(cudaMemcpy(host.data(), result.data(), N * sizeof(float),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int i = 0; i < 13; ++i) {
    EXPECT_TRUE(std::isnan(host[static_cast<size_t>(i)])) << "warm-up at " << i;
  }
  for (int i = 13; i < N; ++i) {
    EXPECT_TRUE(std::isfinite(host[static_cast<size_t>(i)])) << "value at " << i;
  }
}

TEST(ContextLifetime, DeviceBufferBaseOutlivesContext) {
  const int N = 256;
  std::vector<double> data(N);
  for (int i = 0; i < N; ++i) data[static_cast<size_t>(i)] = 100.0 + i;

  tacuda::DeviceBufferBase result;
  {
    tacuda::Context ctx;
    result = ctx.upload(data.data(), N, CT_DTYPE_F64);
  }  // ctx destroyed; the dtype-aware buffer must hold the stream guard too

  std::vector<double> host(N, -1.0);
  ASSERT_EQ(cudaMemcpy(host.data(), result.data(), N * sizeof(double),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int i = 0; i < N; ++i) {
    EXPECT_DOUBLE_EQ(host[static_cast<size_t>(i)], 100.0 + i);
  }
}

TEST(ContextLifetime, CrossContextBufferRejected) {
  const int N = 128;
  std::vector<float> data(N, 1.0f);

  tacuda::Context a;
  tacuda::Context b;
  auto bufA = a.upload(data.data(), N);

  // Same-context use is fine.
  EXPECT_NO_THROW(a.sma(bufA, 14));
  // Foreign-context use must be rejected for every consuming method.
  EXPECT_THROW(b.sma(bufA, 14), std::invalid_argument);
  EXPECT_THROW(b.rsi(bufA, 14), std::invalid_argument);
  auto bufB = b.upload(data.data(), N);
  EXPECT_THROW(b.atr(bufA, bufB, bufB, 14), std::invalid_argument);
  std::vector<float> host(N);
  EXPECT_THROW(b.download(bufA, host.data()), std::invalid_argument);
}

TEST(ContextLifetime, ViewInheritsGuardIdentity) {
  const int N = 64;
  std::vector<float> data(N, 2.0f);

  tacuda::Context a;
  tacuda::Context b;
  auto bufA = a.upload(data.data(), N);
  auto viewA = bufA.view(0, 32);

  EXPECT_NO_THROW(a.sma(viewA, 4));
  EXPECT_THROW(b.sma(viewA, 4), std::invalid_argument);
}
