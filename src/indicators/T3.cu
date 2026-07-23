/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/EMA.h>
#include <indicators/T3.h>
#include <stdexcept>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/TacudaConfig.h>

namespace {

enum class T3Algo { Auto = 0, Sequential = 1, EmaHybrid = 2 };

struct T3KernelConfig {
  int blockSize = 256;
  int sequentialThreshold = 50000;
  T3Algo algo = T3Algo::Auto;
};

static T3KernelConfig& getCfg() {
  static T3KernelConfig cfg = []() {
    auto& gcfg = tacuda::Config::instance();
    T3KernelConfig c;
    c.blockSize = gcfg.t3.blockSize;
    c.sequentialThreshold = gcfg.t3.sequentialThreshold;
    int v = gcfg.t3.algo;
    if (v >= 0 && v <= 2) c.algo = static_cast<T3Algo>(v);
    return c;
  }();
  return cfg;
}

// T3 combine kernel: output starts at 6*(period-1), aligned with TA-Lib lookback
// For output index o = 6*p1 + idx (where p1 = period-1):
//   T3[o] = c1*e6[o-5*p1] + c2*e5[o-4*p1] + c3*e4[o-3*p1] + c4*e3[o-2*p1]
// Since e3..e6 are "compacted" buffers (each shifted by their respective warmup),
// we access: e6[idx+p1], e5[idx+2*p1], e4[idx+3*p1], e3[idx+4*p1]
__global__ void t3CombineKernel(const float *__restrict__ e3,
                                const float *__restrict__ e4,
                                const float *__restrict__ e5,
                                const float *__restrict__ e6,
                                float *__restrict__ output, float c1, float c2,
                                float c3, float c4, int period, int valid) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < valid) {
    int p1 = period - 1;
    // Output starts at 6*p1, TA-Lib lookback
    int outIdx = 6 * p1 + idx;
    // Access compacted buffers with correct offsets
    output[outIdx] = c1 * e6[idx + p1] + c2 * e5[idx + 2 * p1] +
                     c3 * e4[idx + 3 * p1] + c4 * e3[idx + 4 * p1];
  }
}

} // namespace

namespace tacuda {

T3::T3(int period, float vFactor) : period(period), vFactor(vFactor) {}

void T3::calculate(const float *input, float *output,
                   int size, cudaStream_t stream) noexcept(false) {
  if (period < 2 || size <= 0) {
    throw std::invalid_argument("T3: invalid args");
  }
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

  // Always use EMA Hybrid path: compute e1..e6 using parallel EMA
  // T3 lookback = 6*(period-1), matching TA-Lib
  int lookback = 6 * (period - 1);
  if (size <= lookback) {
    // Not enough data, output already filled with NaN
    return;
  }

  auto e1 = acquireDeviceBufferAsync<float>(size, stream);
  auto e2 = acquireDeviceBufferAsync<float>(size - (period - 1), stream);
  auto e3 = acquireDeviceBufferAsync<float>(size - 2 * (period - 1), stream);
  auto e4 = acquireDeviceBufferAsync<float>(size - 3 * (period - 1), stream);
  auto e5 = acquireDeviceBufferAsync<float>(size - 4 * (period - 1), stream);
  auto e6 = acquireDeviceBufferAsync<float>(size - 5 * (period - 1), stream);

  tacuda::EMA ema(period);
  ema.calculate(input, e1.get(), size, stream);

  const float *e1Valid = e1.get() + (period - 1);
  int size2 = size - (period - 1);
  ema.calculate(e1Valid, e2.get(), size2, stream);

  const float *e2Valid = e2.get() + (period - 1);
  int size3 = size2 - (period - 1);
  ema.calculate(e2Valid, e3.get(), size3, stream);

  const float *e3Valid = e3.get() + (period - 1);
  int size4 = size3 - (period - 1);
  ema.calculate(e3Valid, e4.get(), size4, stream);

  const float *e4Valid = e4.get() + (period - 1);
  int size5 = size4 - (period - 1);
  ema.calculate(e4Valid, e5.get(), size5, stream);

  const float *e5Valid = e5.get() + (period - 1);
  int size6 = size5 - (period - 1);
  ema.calculate(e5Valid, e6.get(), size6, stream);

  // T3 coefficients from vFactor (Tillson formula)
  float b = vFactor;
  float c1 = -b * b * b;
  float c2 = 3 * b * b + 3 * b * b * b;
  float c3 = -3 * b - 6 * b * b - 3 * b * b * b;
  float c4 = 1 + 3 * b + 3 * b * b + b * b * b;

  // Valid elements: start at 6*(period-1), matching TA-Lib lookback
  int valid = size - lookback;
  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(valid);
  t3CombineKernel<<<grid, block, 0, stream>>>(e3.get(), e4.get(), e5.get(), e6.get(), output, c1, c2, c3, c4, period, valid);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace tacuda
