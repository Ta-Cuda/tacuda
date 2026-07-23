/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/KAMA.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/TacudaConfig.h>
#include <utils/AffineScan.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <stdexcept>
#include <cmath>

namespace {

struct KAMAKernelConfig {
  int blockSize = 256;
  int sequentialThreshold = 2048;
  int warpThreshold = 4096;
};
static KAMAKernelConfig cfg = []() {
  auto& gcfg = tacuda::Config::instance();
  KAMAKernelConfig c;
  c.sequentialThreshold = gcfg.kama.sequentialThreshold;
  c.warpThreshold = gcfg.kama.warpThreshold;
  return c;
}();

struct AbsDiffFunctor {
  const float* data;
  __host__ __device__ double operator()(int i) const {
    if (i <= 0) return 0.0;
    double cur = static_cast<double>(data[i]);
    double prev = static_cast<double>(data[i - 1]);
    double d = cur - prev;
    return d < 0.0 ? -d : d;
  }
};

// Compute sc[i] for i>=period, else 0
__global__ void kamaComputeScKernel(const float* __restrict__ input,
                                    const double* __restrict__ prefixAbs,
                                    float* __restrict__ scBuf,
                                    int period,
                                    int size,
                                    float fastSC,
                                    float slowSC) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    float sc = 0.0f;
    if (i >= period) {
      // TA-Lib efficiency ratio: signed periodROC; er forced to 1 when the
      // volatility sum is <= the signed direction or (almost) zero
      // (TA_IS_ZERO uses 1e-8).
      double periodROC = static_cast<double>(input[i]) - static_cast<double>(input[i - period]);
      double sumAbs = prefixAbs[i] - prefixAbs[i - period];
      double er;
      if (sumAbs <= periodROC || (sumAbs > -1e-8 && sumAbs < 1e-8)) {
        er = 1.0;
      } else {
        er = fabs(periodROC) / sumAbs;
      }
      double scd = er * (static_cast<double>(fastSC) - static_cast<double>(slowSC)) +
                   static_cast<double>(slowSC);
      sc = static_cast<float>(scd * scd);
    }
    scBuf[i] = sc;
  }
}

using KamaPair = tacuda::detail::AffineTransform1<double>;
using KamaCompose = tacuda::detail::AffineCompose1<double>;

// Legacy compose helper for block-scan kernels that use double2 directly
struct KamaComposeOp {
  __host__ __device__ static inline double2 compose(const double2& leftAccum, const double2& rightCurr) {
    return make_double2(rightCurr.x * leftAccum.x, rightCurr.x * leftAccum.y + rightCurr.y);
  }
  __host__ __device__ double2 operator()(const double2& accum, const double2& curr) const {
    return compose(accum, curr);
  }
};

// Build per-index affine transforms (a_i, b_i) for KAMA: y_i = a_i*y_{i-1} + b_i
__global__ void kamaBuildPairsFromScKernel(const float* __restrict__ input,
                                           const float* __restrict__ scBuf,
                                           double2* __restrict__ pairs,
                                           int period,
                                           int size) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int tailLen = size - (period - 1);
  if (j >= tailLen) return;
  if (j == 0) { pairs[0] = make_double2(1.0, 0.0); return; }
  int i = (period - 1) + j;
  double sc = static_cast<double>(scBuf[i]);
  double a = 1.0 - sc;
  double b = sc * static_cast<double>(input[i]);
  pairs[j] = make_double2(a, b);
}

// In-block inclusive scan for pairs; writes scanned pairs and block totals
__global__ void kamaBlockScanKernel(const double2* __restrict__ inPairs,
                                    double2* __restrict__ outPairs,
                                    double2* __restrict__ blockTotals,
                                    int tailLen) {
  extern __shared__ double2 s[];
  int global = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  double2 val = make_double2(1.0, 0.0);
  if (global < tailLen) val = inPairs[global];
  s[tid] = val;
  __syncthreads();
  // Hillis-Steele inclusive scan with custom compose
  for (int offset = 1; offset < blockDim.x; offset <<= 1) {
    double2 prev = make_double2(1.0, 0.0);
    if (tid >= offset) prev = s[tid - offset];
    __syncthreads();
    if (tid >= offset) s[tid] = KamaComposeOp::compose(prev, s[tid]);
    __syncthreads();
  }
  if (global < tailLen) outPairs[global] = s[tid];
  if (blockTotals && tid == blockDim.x - 1) blockTotals[blockIdx.x] = s[tid];
}

// Apply block prefixes to scanned pairs (compose with prefix of previous blocks)
__global__ void kamaApplyBlockPrefixesKernel(double2* __restrict__ pairs,
                                             const double2* __restrict__ blockPrefixes,
                                             int tailLen) {
  int global = blockIdx.x * blockDim.x + threadIdx.x;
  if (global >= tailLen) return;
  int b = blockIdx.x;
  if (b == 0) return; // first block already correct
  double2 prefix = blockPrefixes[b - 1];
  pairs[global] = KamaComposeOp::compose(prefix, pairs[global]);
}

// Finalize KAMA from scanned pairs; skip j==0 to keep output[period-1] as NaN
__global__ void kamaFinalizeFromPairsKernel(const double2* __restrict__ pairs,
                                            float* __restrict__ output,
                                            float y0,
                                            int period,
                                            int size) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int tailLen = size - (period - 1);
  if (j >= tailLen) return;
  if (j == 0) return; // keep output[period-1] as NaN for parity
  double2 t = pairs[j];
  double y = t.x * static_cast<double>(y0) + t.y;
  int outIdx = (period - 1) + j;
  output[outIdx] = static_cast<float>(y);
}

} // namespace

namespace tacuda {

KAMA::KAMA(int period, int fastPeriod, int slowPeriod)
    : period(period), fastSC(2.0f / (fastPeriod + 1.0f)),
      slowSC(2.0f / (slowPeriod + 1.0f)) {}

void KAMA::calculate(const float *input, float *output,
                     int size, cudaStream_t stream) noexcept(false) {
  if (period <= 0) {
      throw std::invalid_argument("KAMA: invalid period");
  }
  if (period > size) {
      // TA-Lib succeeds with zero output elements when the input is
      // shorter than the lookback; the whole line is warm-up NaNs.
      CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
      return;
  }
  // Initialize output with NaNs so warmup remain NaN
  CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

  // Fused build of prefix |Δ| via transform iterator
  auto absPrefix = acquireDeviceBufferAsync<double>(size, stream);
  {
    thrust::counting_iterator<int> begin(0), end(size);
    AbsDiffFunctor f{input};
    auto it = thrust::make_transform_iterator(begin, f);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), it, it + size, thrust::device_pointer_cast(absPrefix.get()));
  }

  int tailLen = size - (period - 1);
  if (tailLen <= 1) return;

  // Small tails: single-block scan without thrust. Correct only when the
  // whole tail fits into one block — kamaBlockScanKernel<<<1, blockSize>>>
  // scans exactly blockSize elements. (A larger threshold used to leave
  // every element past blockSize unscanned: garbage output for sizes in
  // (blockSize, warpThreshold].)
  if (tailLen <= cfg.blockSize) {
    auto scBuf = acquireDeviceBufferAsync<float>(size, stream);
    kamaComputeScKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(input, absPrefix.get(), scBuf.get(), period, size, fastSC, slowSC);
    CUDA_CHECK(cudaGetLastError());

    auto pairs = acquireDeviceBufferAsync<double2>(tailLen, stream);
    kamaBuildPairsFromScKernel<<<defaultGrid(tailLen), defaultBlock(), 0, stream>>>(input, scBuf.get(), pairs.get(), period, size);
    CUDA_CHECK(cudaGetLastError());

    int blockSize = cfg.blockSize;
    int gridSize = 1; // single block
    size_t shm = static_cast<size_t>(blockSize) * sizeof(double2);
    kamaBlockScanKernel<<<gridSize, blockSize, shm, stream>>>(pairs.get(), pairs.get(), nullptr, tailLen);
    CUDA_CHECK(cudaGetLastError());

    float y0 = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&y0, input + (period - 1), sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    kamaFinalizeFromPairsKernel<<<defaultGrid(tailLen), defaultBlock(), 0, stream>>>(pairs.get(), output, y0, period, size);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  // Larger tails: segmented block scan + block prefix scan (reduces thrust to small array)
  auto scBuf = acquireDeviceBufferAsync<float>(size, stream);
  kamaComputeScKernel<<<defaultGrid(size), defaultBlock(), 0, stream>>>(input, absPrefix.get(), scBuf.get(), period, size, fastSC, slowSC);
  CUDA_CHECK(cudaGetLastError());

  auto pairs = acquireDeviceBufferAsync<double2>(tailLen, stream);
  kamaBuildPairsFromScKernel<<<defaultGrid(tailLen), defaultBlock(), 0, stream>>>(input, scBuf.get(), pairs.get(), period, size);
  CUDA_CHECK(cudaGetLastError());

  int blockSize = cfg.blockSize;
  int numBlocks = (tailLen + blockSize - 1) / blockSize;
  auto blockTotals = acquireDeviceBufferAsync<double2>(numBlocks, stream);
  size_t shm = static_cast<size_t>(blockSize) * sizeof(double2);
  kamaBlockScanKernel<<<numBlocks, blockSize, shm, stream>>>(pairs.get(), pairs.get(), blockTotals.get(), tailLen);
  CUDA_CHECK(cudaGetLastError());

  // Scan block totals with thrust (small)
  thrust::inclusive_scan(thrust::cuda::par.on(stream),
                         thrust::device_pointer_cast(blockTotals.get()),
                         thrust::device_pointer_cast(blockTotals.get() + numBlocks),
                         thrust::device_pointer_cast(blockTotals.get()),
                         KamaComposeOp{});

  // Apply block prefixes
  kamaApplyBlockPrefixesKernel<<<numBlocks, blockSize, 0, stream>>>(pairs.get(), blockTotals.get(), tailLen);
  CUDA_CHECK(cudaGetLastError());

  // Finalize values
  float y0 = 0.0f;
  CUDA_CHECK(cudaMemcpyAsync(&y0, input + (period - 1), sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  kamaFinalizeFromPairsKernel<<<defaultGrid(tailLen), defaultBlock(), 0, stream>>>(pairs.get(), output, y0, period, size);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace tacuda
