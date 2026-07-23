/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/SMA.h>
#include <indicators/TRIMA.h>
#include <stdexcept>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>

tacuda::TRIMA::TRIMA(int period) : period(period) {}

void tacuda::TRIMA::calculate(const float *input, float *output,
                      int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0) {
      throw std::invalid_argument("TRIMA: invalid period");
  }
  if (size < period) {
      // TA-Lib succeeds with zero output elements when the input is
      // shorter than the lookback; the whole line is warm-up NaNs.
      CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
      return;
  }
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

  // Triangular MA as SMA(SMA(x, p1), p2) with end-of-window alignment
  int p1 = (period + 1) / 2;
  int p2 = (period % 2 == 0) ? (p1 + 1) : p1;

  auto tmp = acquireDeviceBufferAsync<float>(size, stream);

  // First SMA over input
  tacuda::SMA sma1(p1);
  sma1.calculate(input, tmp.get(), size, stream);

  // Second SMA over valid tail of first SMA and write aligned into global output
  const float* tmpValid = tmp.get() + (p1 - 1);
  int size2 = size - (p1 - 1);
  tacuda::SMA sma2(p2);
  sma2.calculate(tmpValid, output + (p1 - 1), size2, stream);
}
