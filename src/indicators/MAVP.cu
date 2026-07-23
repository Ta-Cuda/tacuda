/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/MAVP.h>

#include <indicators/MA.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

__host__ __device__ inline int mavpLookback(tacuda::MAType type, int period) {
  // TA-Lib semantics: one global outBegIdx based on maxPeriod and MA type.
  // For MAVP, lookback corresponds to the MA lookback at maxPeriod.
  switch (type) {
    case tacuda::MAType::SMA:
    case tacuda::MAType::EMA:
    case tacuda::MAType::WMA:
    case tacuda::MAType::TRIMA:
      return period - 1;
    case tacuda::MAType::DEMA:
      return 2 * (period - 1);
    case tacuda::MAType::TEMA:
      return 3 * (period - 1);
    case tacuda::MAType::KAMA:
      return period;
    case tacuda::MAType::T3:
      return 6 * (period - 1);
    case tacuda::MAType::MAMA:
      // TA-Lib MAVP with MAMA returns NaNs (period is not applicable).
      return std::numeric_limits<int>::max();
    default:
      return period - 1;
  }
}

__global__ void mavpSelectKernel(const float* __restrict__ periods,
                                const float* __restrict__ maForPeriod,
                                float* __restrict__ output,
                                int size,
                                int minPeriod,
                                int maxPeriod,
                                int curPeriod,
                                int lookback) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;
  if (idx < lookback) return;  // output stays NaN

  float pRaw = periods[idx];
  if (!isfinite(pRaw)) return;
  int p = static_cast<int>(pRaw);  // TA-Lib truncation semantics
  if (p < minPeriod) p = minPeriod;
  if (p > maxPeriod) p = maxPeriod;

  if (p == curPeriod) {
    output[idx] = maForPeriod[idx];
  }
}

// TA-Lib compatible EMA starting at an explicit start index. TA-Lib MAVP
// calls TA_MA(startIdx=lookback(maxPeriod), ...) for every distinct period,
// and TA_INT_EMA seeds with the SMA of the `period` bars ENDING at startIdx —
// not with the SMA at the beginning of the series. A whole-series EMA phases
// in differently and stays visibly off for thousands of bars.
__global__ void mavpEmaFromStartKernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int size,
                                       int period,
                                       int startIdx) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (size <= 0 || period <= 0 || startIdx >= size) return;
  if (startIdx < period - 1) startIdx = period - 1;

  const double k = 2.0 / (static_cast<double>(period) + 1.0);
  int today = startIdx - (period - 1);
  double prev = 0.0;
  for (int i = 0; i < period; ++i) {
    prev += static_cast<double>(input[today++]);
  }
  prev /= static_cast<double>(period);
  output[startIdx] = static_cast<float>(prev);  // today == startIdx + 1
  for (int t = today; t < size; ++t) {
    prev = (static_cast<double>(input[t]) - prev) * k + prev;
    output[t] = static_cast<float>(prev);
  }
}

// TA-Lib compatible KAMA computation starting at an explicit start index.
// This is required for MAVP+KAMA because TA-Lib calls TA_MA(startIdx, endIdx, ...)
// and TA_KAMA output depends on startIdx.
__global__ void mavpKamaFromStartKernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int size,
                                       int period,
                                       int startIdx) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (size <= 0) return;
  if (period <= 0) return;

  // TA_KAMA lookback
  const int lookbackTotal = period;
  if (startIdx < lookbackTotal) startIdx = lookbackTotal;
  if (startIdx >= size) return;

  // TA-Lib KAMA constants: fast=2, slow=30 (fixed)
  const double constMax = 2.0 / (30.0 + 1.0);
  const double constDiff = 2.0 / (2.0 + 1.0) - constMax;

  // Initialize the variables by going through the lookback period.
  double sumROC1 = 0.0;
  int today = startIdx - lookbackTotal;
  int trailingIdx = today;
  for (int i = 0; i < period; ++i) {
    double t = static_cast<double>(input[today]) - static_cast<double>(input[today + 1]);
    sumROC1 += fabs(t);
    ++today;
  }

  // Calculate the first KAMA.
  double prevKAMA = static_cast<double>(input[today - 1]);  // yesterday

  double tempReal = static_cast<double>(input[today]);
  double tempReal2 = static_cast<double>(input[trailingIdx++]);
  double periodROC = tempReal - tempReal2;

  // Save the trailing value (see TA-Lib note about aliasing).
  double trailingValue = tempReal2;

  // Efficiency ratio.
  double er;
  if ((sumROC1 <= periodROC) || (sumROC1 == 0.0)) er = 1.0;
  else er = fabs(periodROC / sumROC1);

  // Smoothing constant.
  double sc = (er * constDiff) + constMax;
  sc *= sc;

  // First output at index startIdx.
  prevKAMA = ((static_cast<double>(input[today]) - prevKAMA) * sc) + prevKAMA;
  output[today] = static_cast<float>(prevKAMA);
  ++today;

  // Unstable period is assumed 0 (TA-Lib default), so nothing to skip here.

  // Remaining range.
  while (today < size) {
    tempReal = static_cast<double>(input[today]);
    tempReal2 = static_cast<double>(input[trailingIdx++]);
    periodROC = tempReal - tempReal2;

    sumROC1 -= fabs(trailingValue - tempReal2);
    sumROC1 += fabs(tempReal - static_cast<double>(input[today - 1]));
    trailingValue = tempReal2;

    if ((sumROC1 <= periodROC) || (sumROC1 == 0.0)) er = 1.0;
    else er = fabs(periodROC / sumROC1);

    sc = (er * constDiff) + constMax;
    sc *= sc;

    prevKAMA = ((static_cast<double>(input[today]) - prevKAMA) * sc) + prevKAMA;
    output[today] = static_cast<float>(prevKAMA);
    ++today;
  }
}

} // namespace

tacuda::MAVP::MAVP(int minPeriod, int maxPeriod, MAType type)
    : minPeriod(minPeriod), maxPeriod(maxPeriod), type(type) {
    if (minPeriod <= 0 || maxPeriod <= 0 || minPeriod > maxPeriod) {
        throw std::invalid_argument("MAVP: invalid period range");
    }
}

void tacuda::MAVP::calculate(const float* values, const float* periods, float* output,
                             int size, cudaStream_t stream) noexcept(false) {
    if (size <= 0) {
        throw std::invalid_argument("MAVP: invalid size");
    }

    // Initialize output to NaN (TA-Lib style padding to full series length)
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    // TA-Lib MAVP: output begins at lookback(type, maxPeriod) regardless of per-bar periods.
    const int lookback = mavpLookback(type, maxPeriod);
    if (lookback >= size) return;

    // TA-Lib returns NaNs for MAMA in MAVP (period not applicable).
    if (type == tacuda::MAType::MAMA) return;

    auto tmp = acquireDeviceBufferAsync<float>(static_cast<size_t>(size), stream);
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);

    for (int p = minPeriod; p <= maxPeriod; ++p) {
        if (type == tacuda::MAType::KAMA) {
            // TA-Lib MAVP+KAMA depends on the caller startIdx; compute KAMA starting at
            // the global MAVP start index (lookback of maxPeriod).
            CUDA_CHECK(cudaMemsetAsync(tmp.get(), 0xFF, size * sizeof(float), stream));
            mavpKamaFromStartKernel<<<1, 1, 0, stream>>>(values, tmp.get(), size, p, lookback);
            CUDA_CHECK(cudaGetLastError());
        } else if (type == tacuda::MAType::EMA) {
            // TA-Lib MAVP+EMA seeds each per-period EMA at the global MAVP
            // start index (see mavpEmaFromStartKernel).
            CUDA_CHECK(cudaMemsetAsync(tmp.get(), 0xFF, size * sizeof(float), stream));
            mavpEmaFromStartKernel<<<1, 1, 0, stream>>>(values, tmp.get(), size, p, lookback);
            CUDA_CHECK(cudaGetLastError());
        } else {
            tacuda::MA ma(p, type);
            ma.calculate(values, tmp.get(), size, stream);
        }

        mavpSelectKernel<<<grid, block, 0, stream>>>(
            periods, tmp.get(), output, size, minPeriod, maxPeriod, p, lookback);
        CUDA_CHECK(cudaGetLastError());
    }
}

void tacuda::MAVP::calculate(const float* input, float* output, int size,
                             cudaStream_t stream) noexcept(false) {
    (void)input;
    (void)output;
    (void)size;
    (void)stream;
    throw std::logic_error(
        "MAVP::calculate requires separate value and period buffers");
}
