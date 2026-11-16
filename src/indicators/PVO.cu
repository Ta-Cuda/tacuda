/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <algorithm>
#include <stdexcept>
#include <indicators/PVO.h>
#include <utils/CudaUtils.h>
#include <utils/AffineScan.cuh>
#include <utils/DeterministicScan.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace {
using Pair = tacuda::detail::AffineTransform1<double>;

__global__ void pvoSeeds(const float* volume, double* seed, int fast, int slow) {
  __shared__ double fastSum[256], slowSum[256];
  const int t = threadIdx.x;
  double f = 0.0, s = 0.0;
  for (int i = t; i < slow; i += blockDim.x) {
    const double value = volume[i];
    s += value;
    if (i < fast) f += value;
  }
  fastSum[t] = f; slowSum[t] = s;
  __syncthreads();
  for (int stride = 128; stride; stride >>= 1) {
    if (t < stride) {
      fastSum[t] += fastSum[t + stride];
      slowSum[t] += slowSum[t + stride];
    }
    __syncthreads();
  }
  if (t == 0) { seed[0] = fastSum[0] / fast; seed[1] = slowSum[0] / slow; }
}

struct EmaTerms {
  const float* volume;
  const double* seed;
  int period, seedIndex;
  double alpha;
  __device__ Pair operator()(int i) const {
    if (i < period - 1) return {1.0, 0.0};
    if (i == period - 1) return {0.0, seed[seedIndex]};
    return {1.0 - alpha, alpha * static_cast<double>(volume[i])};
  }
};

__global__ void pvoKernel(const Pair* __restrict__ fast,
                          const Pair* __restrict__ slow,
                          float* __restrict__ output,
                          int slowP, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= slowP - 1 && idx < size) {
    const double f = fast[idx].b, s = slow[idx].b;
    output[idx] = fabs(s) < 1e-14 ? 0.0f : static_cast<float>(100.0 * (f - s) / s);
  }
}
}

tacuda::PVO::PVO(int fastPeriod, int slowPeriod)
    : fastPeriod(fastPeriod), slowPeriod(slowPeriod) {}

void tacuda::PVO::calculate(const float* input, float* output,
                    int size, cudaStream_t stream) noexcept(false) {
  if (fastPeriod <= 0 || slowPeriod <= 0) {
    throw std::invalid_argument("PVO: invalid periods");
  }
  if (fastPeriod >= slowPeriod) {
    throw std::invalid_argument("PVO: fastPeriod must be < slowPeriod");
  }
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
  if (slowPeriod > size) return;
  auto seed = acquireDeviceBufferAsync<double>(2, stream);
  pvoSeeds<<<1, 256, 0, stream>>>(input, seed.get(), fastPeriod, slowPeriod);
  CUDA_CHECK(cudaGetLastError());
  auto fast = acquireDeviceBufferAsync<Pair>(size, stream);
  auto slow = acquireDeviceBufferAsync<Pair>(size, stream);
  auto indices = thrust::make_counting_iterator<int>(0);
  auto fastTerms = thrust::make_transform_iterator(indices,
      EmaTerms{input, seed.get(), fastPeriod, 0, 2.0 / (fastPeriod + 1.0)});
  auto slowTerms = thrust::make_transform_iterator(indices,
      EmaTerms{input, seed.get(), slowPeriod, 1, 2.0 / (slowPeriod + 1.0)});
  using Compose = tacuda::detail::AffineCompose1<double>;
  tacuda::detail::deterministic_inclusive_scan<Pair>(fastTerms, fast.get(), size,
      stream, Compose{}, Pair{1.0, 0.0});
  tacuda::detail::deterministic_inclusive_scan<Pair>(slowTerms, slow.get(), size,
      stream, Compose{}, Pair{1.0, 0.0});
  pvoKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(
      fast.get(), slow.get(), output, slowPeriod, size);
  CUDA_CHECK(cudaGetLastError());
}
