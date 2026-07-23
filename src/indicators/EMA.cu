/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/EMA.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/AffineScan.cuh>
#include <utils/TacudaConfig.h>
#include <stdexcept>
#include <cstdlib>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

// Affine pair type — always double for accumulation precision
using AffinePairD = tacuda::detail::AffineTransform1<double>;

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

// TA-Lib compatible EMA seed: SMA of the first `period` samples written at
// index (period-1). Subsequent samples follow the standard EMA recurrence.
template<typename T>
__global__ void emaInitSmaSeedKernel(const T* __restrict__ input,
                                     T* __restrict__ output,
                                     int period,
                                     int size) {
    extern __shared__ char smem_raw[];
    double* ssum = reinterpret_cast<double*>(smem_raw);
    int tid = threadIdx.x;

    double partial = 0.0;
    for (int i = tid; i < period && i < size; i += blockDim.x) {
        partial += static_cast<double>(input[i]);
    }
    ssum[tid] = partial;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0 && period <= size) {
        output[period - 1] = static_cast<T>(ssum[0] / static_cast<double>(period));
    }
}

// Build transforms (A,b) for EMA recurrence per index starting at (period-1)
// pairs[0] = (1,0) identity keeps SMA seed; for j>=1, i = period-1 + j
// pairs[j] = (1-alpha, alpha * x[i])
template<typename T>
__global__ void emaBuildTransformsKernel(const T* __restrict__ input,
                                         AffinePairD* __restrict__ pairs,
                                         int period,
                                         int size,
                                         double alpha) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tailLen = size - (period - 1);
    if (tailLen <= 0) return;
    if (tid >= tailLen) return;
    if (tid == 0) {
        pairs[0] = {1.0, 0.0};
        return;
    }
    int i = (period - 1) + tid;
    double A = 1.0 - alpha;
    double b = alpha * static_cast<double>(input[i]);
    pairs[tid] = {A, b};
}

// After inclusive scan over pairs with composition, finalize outputs
template<typename T>
__global__ void emaFinalizeFromPairsKernel(const AffinePairD* __restrict__ pairs,
                                           T* __restrict__ output,
                                           int period,
                                           int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tailLen = size - (period - 1);
    if (tailLen <= 0) return;
    if (tid >= tailLen) return;
    double y0 = static_cast<double>(output[period - 1]);
    AffinePairD pc = pairs[tid];
    double y = pc.A * y0 + pc.b;
    int i = (period - 1) + tid;
    output[i] = static_cast<T>(y);
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
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }

    // Initialize warmup to NaN
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(T), stream));

    if (period == 1) {
        CUDA_CHECK(cudaMemcpyAsync(output, input, size * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        return;
    }

    // Seed with SMA of first 'period' values at index (period-1) (TA-Lib compatible)
    {
        dim3 block(getCfg().blockSize);
        dim3 grid(1);
        size_t shm = static_cast<size_t>(block.x) * sizeof(double);
        emaInitSmaSeedKernel<T><<<grid, block, shm, stream>>>(input, output, period, size);
        CUDA_CHECK(cudaGetLastError());
    }

    // Always use parallel scan (works correctly for all sizes)
    double alpha = 2.0 / (static_cast<double>(period) + 1.0);

    {
        int tailLen = size - (period - 1);
        if (tailLen <= 0) return;
        auto pairsBuf = acquireDeviceBufferAsync<AffinePairD>(tailLen, stream);
        dim3 b = defaultBlock();
        dim3 g = defaultGrid(tailLen);
        emaBuildTransformsKernel<T><<<g, b, 0, stream>>>(input, pairsBuf.get(), period, size, alpha);
        CUDA_CHECK(cudaGetLastError());

        tacuda::detail::affine_inclusive_scan<AffinePairD, tacuda::detail::AffineCompose1<double>>(
            pairsBuf.get(), pairsBuf.get(), tailLen, stream);

        emaFinalizeFromPairsKernel<T><<<g, b, 0, stream>>>(pairsBuf.get(), output, period, size);
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
