/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/ROCR100.h>
#include <stdexcept>
#include <utils/CudaUtils.h>

__global__ void rocr100Kernel(const float *__restrict__ input,
                              float *__restrict__ output, int period,
                              int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size - period) {
    float prev = input[idx];
    float curr = input[idx + period];
    // Write result at idx + period to match TA-Lib behavior (first period
    // values remain NaN as warmup); a zero denominator yields 0, not NaN.
    output[idx + period] = (prev != 0.0f) ? (curr / prev) * 100.0f : 0.0f;
  }
}

tacuda::ROCR100::ROCR100(int period) : period(period) {}

void tacuda::ROCR100::calculate(const float *input, float *output,
                        int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0) {
      throw std::invalid_argument("ROCR100: invalid period");
  }
  if (period >= size) {
      // TA-Lib succeeds with zero output elements when the input is
      // shorter than the lookback; the whole line is warm-up NaNs.
      CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
      return;
  }
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(size);
  rocr100Kernel<<<grid, block, 0, stream>>>(input, output, period, size);
  CUDA_CHECK(cudaGetLastError());
}
