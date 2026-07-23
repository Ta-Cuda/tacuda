/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/SUM.h>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>

__global__ void castToDoubleKernel(const float *__restrict__ input,
                                   double *__restrict__ out,
                                   int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = static_cast<double>(input[idx]);
  }
}

// Double-precision variant to reduce accumulation error in long windows.
// TA-Lib convention: the sum over [end-period+1, end] is reported at `end`,
// with a leading warm-up of period-1 NaNs.
__global__ void sumKernelPrefixD(const double *__restrict__ prefix,
                                 float *__restrict__ output, int period,
                                 int size) {
  int end = blockIdx.x * blockDim.x + threadIdx.x;
  if (end >= size || end < period - 1) return;
  int j = end - period + 1;
  double prev = (j == 0) ? 0.0 : prefix[j - 1];
  double sum = prefix[end] - prev;
  output[end] = static_cast<float>(sum);
}

tacuda::SUM::SUM(int period) : period(period) {}

void tacuda::SUM::calculate(const float *input, float *output,
                    int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0) {
      throw std::invalid_argument("SUM: invalid period");
  }
  if (period > size) {
      // TA-Lib succeeds with zero output elements when the input is
      // shorter than the lookback; the whole line is warm-up NaNs.
      CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
      return;
  }
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

  // Compute prefix in double via explicit cast to avoid float-accumulation drift
  auto prefixD = acquireDeviceBufferAsync<double>(size, stream);
  {
    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    castToDoubleKernel<<<grid, block, 0, stream>>>(input, prefixD.get(), size);
    CUDA_CHECK(cudaGetLastError());
  }
  thrust::device_ptr<double> preD(prefixD.get());
  thrust::inclusive_scan(thrust::cuda::par.on(stream), preD, preD + size, preD);

  dim3 block = defaultBlock();
  dim3 grid = defaultGrid(size);
  sumKernelPrefixD<<<grid, block, 0, stream>>>(prefixD.get(), output, period, size);
  CUDA_CHECK(cudaGetLastError());
}
