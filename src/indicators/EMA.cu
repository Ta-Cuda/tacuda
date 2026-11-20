/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/EMA.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/AffineScan.cuh>
#include <utils/LeadingNaN.cuh>
#include <utils/TacudaConfig.h>
#include <stdexcept>
#include <cstdlib>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

// Affine pair type — always double for accumulation precision
using AffinePairD = tacuda::detail::AffineTransform1<double>;
// Share the transform allocation: its leading slot holds seed index and value.

namespace {

enum class EmaAlgo {
    Auto = 0,
    Sequential = 1,
    WarpOptimized = 2,
    ParallelScan = 3,
    Hybrid = 4
};

struct EMAKernelConfig {
    int blockSize = 256;
    int sequentialThreshold = 10000;
    int warpThreshold = 100000;
    EmaAlgo algo = EmaAlgo::Auto;
};

static EMAKernelConfig& getCfg() {
    static EMAKernelConfig cfg = []() {
        auto& gcfg = tacuda::Config::instance();
        EMAKernelConfig c;
        c.blockSize = gcfg.ema.blockSize;
        c.sequentialThreshold = gcfg.ema.sequentialThreshold;
        c.warpThreshold = gcfg.ema.warpThreshold;
        int v = gcfg.ema.algo;
        if (v >= 0 && v <= 4) c.algo = static_cast<EmaAlgo>(v);
        return c;
    }();
    return cfg;
}

// ===================== Templated kernels =====================

// Compute the SMA seed and its position without reading the input on the host.
template<typename T>
__global__ void emaInitSmaSeedKernel(const T* __restrict__ input,
                                     AffinePairD* __restrict__ seed,
                                     int period,
                                     int size) {
    extern __shared__ char smem_raw[];
    double* ssum = reinterpret_cast<double*>(smem_raw);
    __shared__ int first;
    int tid = threadIdx.x;
    if (tid == 0) first = tacuda::detail::leading_nan_count(input, size);
    __syncthreads();
    if (period > size - first) {
        if (tid == 0) *seed = {static_cast<double>(size), 0.0};
        return;
    }

    double partial = 0.0;
    for (int i = tid; i < period; i += blockDim.x) {
        partial += static_cast<double>(input[first + i]);
    }
    ssum[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *seed = {static_cast<double>(first + period - 1),
                 static_cast<T>(ssum[0] / static_cast<double>(period))};
    }
}

// Keep transforms indexed from the nominal lookback for the parallel scan.
// Until the row's actual seed index the transform is the identity.
template<typename T>
__global__ void emaBuildTransformsKernel(const T* __restrict__ input,
                                         AffinePairD* __restrict__ pairs,
                                         const AffinePairD* __restrict__ seed,
                                         int period,
                                         int size,
                                         double alpha) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tailLen = size - (period - 1);
    if (tailLen <= 0) return;
    if (tid >= tailLen) return;
    const int i = period - 1 + tid;
    if (i <= static_cast<int>(seed->A)) {
        pairs[tid] = {1.0, 0.0};
        return;
    }
    double A = 1.0 - alpha;
    double b = alpha * static_cast<double>(input[i]);
    pairs[tid] = {A, b};
}

// After inclusive scan over pairs with composition, finalize outputs
template<typename T>
__global__ void emaFinalizeFromPairsKernel(const AffinePairD* __restrict__ pairs,
                                           const AffinePairD* __restrict__ seed,
                                           T* __restrict__ output,
                                           int period,
                                           int size) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<size_t>(size)) return;
    if (i < static_cast<size_t>(seed->A)) {
        output[i] = static_cast<T>(nanf(""));
        return;
    }
    const AffinePairD pc = pairs[i - (period - 1)];
    output[i] = static_cast<T>(pc.A * seed->b + pc.b);
}

// ===================== Templated implementation =====================

template<typename T>
void ema_calculate_impl(int period, const T* input, T* output, int size, cudaStream_t stream) {
    if (period <= 0) {
        throw std::invalid_argument("EMA: invalid period");
    }
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(T), stream));
        return;
    }

    if (period == 1) {
        CUDA_CHECK(cudaMemcpyAsync(output, input, size * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        return;
    }

    const int tailLen = size - (period - 1);
    auto storage = acquireDeviceBufferAsync<AffinePairD>(static_cast<size_t>(tailLen) + 1, stream);
    auto* seed = storage.get();
    auto* pairs = seed + 1;

    {
        dim3 block(getCfg().blockSize);
        dim3 grid(1);
        size_t shm = static_cast<size_t>(block.x) * sizeof(double);
        emaInitSmaSeedKernel<T><<<grid, block, shm, stream>>>(input, seed, period, size);
        CUDA_CHECK(cudaGetLastError());
    }

    // Always use parallel scan (works correctly for all sizes)
    double alpha = 2.0 / (static_cast<double>(period) + 1.0);

    {
        dim3 b = defaultBlock();
        dim3 g = defaultGrid(tailLen);
        emaBuildTransformsKernel<T><<<g, b, 0, stream>>>(input, pairs, seed, period, size, alpha);
        CUDA_CHECK(cudaGetLastError());

        tacuda::detail::affine_inclusive_scan<AffinePairD, tacuda::detail::AffineCompose1<double>>(
            pairs, pairs, tailLen, stream);

        emaFinalizeFromPairsKernel<T><<<defaultGrid(size), b, 0, stream>>>(pairs, seed, output, period, size);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

// ===================== Explicit instantiations =====================

namespace tacuda {

// --- EMAT<float> ---
template<>
void EMAT<float>::calculate(const float* input, float* output, int size, cudaStream_t stream) {
    ema_calculate_impl<float>(period, input, output, size, stream);
}

// --- EMAT<double> ---
template<>
void EMAT<double>::calculate(const double* input, double* output, int size, cudaStream_t stream) {
    ema_calculate_impl<double>(period, input, output, size, stream);
}

// --- Backward-compatible EMA (float, inherits from Indicator) ---
EMA::EMA(int period) : period(period) {}

void EMA::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    ema_calculate_impl<float>(period, input, output, size, stream);
}

template class EMAT<float>;
template class EMAT<double>;

} // namespace tacuda
