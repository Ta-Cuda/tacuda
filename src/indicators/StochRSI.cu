/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/StochRSI.h>
#include <indicators/RSI.h>
#include <math.h>
#include <stdexcept>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>

__global__ void stochRsiFromRsiKernel(const float *__restrict__ rsi,
                               float *__restrict__ kOut,
                               float *__restrict__ dOut, int rsiPeriod,
                               int kPeriod, int dPeriod, int size) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float nan = nanf("");
    for (int i = 0; i < size; ++i) {
      kOut[i] = nan;
      dOut[i] = nan;
    }

    const int kStart = rsiPeriod + kPeriod - 1;
    for (int i = kStart; i < size; ++i) {
      float highest = -INFINITY;
      float lowest = INFINITY;
      bool valid = true;
      for (int j = 0; j < kPeriod; ++j) {
        float v = rsi[i - j];
        if (isnan(v)) {
          valid = false;
          break;
        }
        if (v > highest) highest = v;
        if (v < lowest) lowest = v;
      }
      if (!valid) continue;

      float denom = highest - lowest;
      kOut[i] = (denom == 0.0f) ? 0.0f : (rsi[i] - lowest) / denom * 100.0f;
    }

    const int dStart = kStart + dPeriod - 1;
    for (int i = dStart; i < size; ++i) {
      float sum = 0.0f;
      bool valid = true;
      for (int j = 0; j < dPeriod; ++j) {
        float v = kOut[i - j];
        if (isnan(v)) {
          valid = false;
          break;
        }
        sum += v;
      }
      if (!valid) continue;
      dOut[i] = sum / dPeriod;
    }

    // TA-Lib publishes FASTK and FASTD with a common outBegIdx: the K line
    // loses the D warm-up as well.
    for (int i = kStart; i < dStart && i < size; ++i) {
      kOut[i] = nan;
    }
  }
}

tacuda::StochRSI::StochRSI(int rsiPeriod, int kPeriod, int dPeriod)
    : rsiPeriod(rsiPeriod), kPeriod(kPeriod), dPeriod(dPeriod) {}

void tacuda::StochRSI::calculate(const float *input, float *output,
                         int size, cudaStream_t stream) noexcept(false) {
  if (rsiPeriod <= 0 || kPeriod <= 0 || dPeriod <= 0 || size <= 0) {
    throw std::invalid_argument("StochRSI: invalid parameters");
  }
  if (size <= rsiPeriod + kPeriod + dPeriod - 2) {
    // TA-Lib succeeds with zero output elements when the input is shorter
    // than the shared K/D lookback; both lines are NaNs.
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, 2 * static_cast<size_t>(size) * sizeof(float), stream));
    return;
  }
  auto rsi = acquireDeviceBufferAsync<float>(size, stream);
  float *kOut = output;
  float *dOut = output + size;

  // Reuse RSI implementation for TA-Lib-compatible Wilder smoothing.
  RSI rsiIndicator(rsiPeriod);
  rsiIndicator.calculate(input, rsi.get(), size, stream);

  stochRsiFromRsiKernel<<<1, 1, 0, stream>>>(rsi.get(), kOut, dOut,
                                             rsiPeriod, kPeriod, dPeriod, size);
  CUDA_CHECK(cudaGetLastError());
}
