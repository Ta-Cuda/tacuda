/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/WMA.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/TacudaConfig.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

namespace {

// WMA weights are simply 1..period: computing (k + 1) in-register is free,
// so no weight buffers exist at all. (The previous constant-memory /
// global-buffer caches were shared mutable state — racy across concurrent
// streams/threads and wrong across multiple devices.)
__device__ __forceinline__ float wmaWeight(int k) {
    return static_cast<float>(k + 1);
}

__global__ void castFloatToDoubleKernel(const float* __restrict__ input,
                                        double* __restrict__ out,
                                        int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = static_cast<double>(input[idx]);
}

__global__ void buildIndexTimesKernel(const float* __restrict__ input,
                                      double* __restrict__ out,
                                      int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = static_cast<double>(idx) * static_cast<double>(input[idx]);
}

// Finalize WMA from prefix sums with end-of-window alignment (weights 1..P oldest->newest)
__global__ void wmaFinalizeFromPrefixKernel(const double* __restrict__ prefixX,
                                            const double* __restrict__ prefixIX,
                                            float* __restrict__ output,
                                            int period,
                                            int size) {
    int end = blockIdx.x * blockDim.x + threadIdx.x;
    if (end >= size) return;
    if (end < period - 1) return; // warmup stays NaN

    int j = end - period + 1;
    double sumX = prefixX[end] - (j > 0 ? prefixX[j - 1] : 0.0);
    double sumIX = prefixIX[end] - (j > 0 ? prefixIX[j - 1] : 0.0);
    double weighted = sumIX + (1.0 - static_cast<double>(j)) * sumX;
    double denom = 0.5 * static_cast<double>(period) * static_cast<double>(period + 1);
    output[end] = static_cast<float>(weighted / denom);
}

// Small-window kernel (shared memory); weights 1..window computed inline.
__global__ void wmaKernelSmallWindow(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     float invSum,
                                     int size,
                                     int window) {
    extern __shared__ float tile[];
    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    int gid = blockStart + tid;

    int sharedCount = blockDim.x + window - 1;
    int tileStart = blockStart - (window - 1);
    for (int i = tid; i < sharedCount; i += blockDim.x) {
        int src = tileStart + i;
        tile[i] = (src >= 0 && src < size) ? input[src] : 0.0f;
    }
    __syncthreads();

    if (gid >= window - 1 && gid < size) {
        double sum = 0.0;
        // weights index 0..window-1 equals 1..window
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            if (k >= window) break;
            sum += static_cast<double>(tile[tid + k]) * static_cast<double>(wmaWeight(k));
        }
        for (int k = 32; k < window; ++k) {
            sum += static_cast<double>(tile[tid + k]) * static_cast<double>(wmaWeight(k));
        }
        output[gid] = static_cast<float>(sum * static_cast<double>(invSum));
    }
}

__device__ __forceinline__ float ldgf(const float* ptr) {
    #if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
    #else
    return *ptr;
    #endif
}

// Warp-optimized kernel: each warp computes one output using weights 1..W
__global__ void wmaKernelWarpOptimized(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       float invSum,
                                       int size,
                                       int window) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warpsPerBlock = blockDim.x >> 5;
    int warpInBlock = tid >> 5;
    int warpGlobal = blockIdx.x * warpsPerBlock + warpInBlock;
    int gid = warpGlobal;

    if (gid >= window - 1 && gid < size) {
        int start = gid - window + 1;
        double sum = 0.0;
        for (int k = lane; k < window; k += 32) {
            float x = ldgf(input + start + k);
            sum += static_cast<double>(x) * static_cast<double>(wmaWeight(k));
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            output[gid] = static_cast<float>(sum * static_cast<double>(invSum));
        }
    }
}

// Unrolled-by-4 kernel. Scalar loads only: the window start is arbitrary,
// so a reinterpret-cast float4 load would be misaligned (UB on CUDA).
__global__ void wmaKernelVectorized(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    float invSum,
                                    int size,
                                    int window) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    if (gid < window - 1) return;

    int start = gid - window + 1;
    double sum = 0.0;

    int vecIters = window >> 2; // /4

    for (int it = 0; it < vecIters; ++it) {
        int base = it << 2;
        const float* p = input + start + base;
        sum += static_cast<double>(p[0]) * static_cast<double>(wmaWeight(base + 0))
             + static_cast<double>(p[1]) * static_cast<double>(wmaWeight(base + 1))
             + static_cast<double>(p[2]) * static_cast<double>(wmaWeight(base + 2))
             + static_cast<double>(p[3]) * static_cast<double>(wmaWeight(base + 3));
    }
    for (int k = (vecIters << 2); k < window; ++k) {
        sum += static_cast<double>(input[start + k]) * static_cast<double>(wmaWeight(k));
    }

    output[gid] = static_cast<float>(sum * static_cast<double>(invSum));
}

struct WMAKernelConfig {
    int blockSize = 256;
    int smallWindowThreshold = 64;
    int warpWindowThreshold = 256;
    int gridStrideFactor = 4;
    int maxGridSize = 65535;
    int maxSharedBytes = 48 * 1024;
    int prefixPreferThreshold = 96; // prefer prefix for >= threshold
    bool preferPrefixForLarge = true;
    bool enableWarp = true;
    bool enableVector = true;
};

static WMAKernelConfig cfg = []() {
    WMAKernelConfig c;
    cudaDeviceProp prop{};
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        c.maxSharedBytes = static_cast<int>(prop.sharedMemPerBlock);
    }
    auto& gcfg = tacuda::Config::instance();
    c.blockSize = gcfg.wma.blockSize;
    c.smallWindowThreshold = gcfg.wma.smallWindowThreshold;
    c.warpWindowThreshold = gcfg.wma.warpWindowThreshold;
    c.gridStrideFactor = gcfg.wma.gridStrideFactor;
    if (gcfg.wma.maxSharedBytes > 0) c.maxSharedBytes = gcfg.wma.maxSharedBytes;
    c.preferPrefixForLarge = gcfg.wma.preferPrefixForLarge;
    c.enableWarp = gcfg.wma.enableWarp;
    c.prefixPreferThreshold = gcfg.wma.prefixPreferThreshold;
    c.enableVector = gcfg.wma.enableVector;
    return c;
}();

} // namespace

namespace tacuda {

WMA::WMA(int period) : period(period) {}

void WMA::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("WMA: invalid period");
    }
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }

    // Initialize warmup with NaN
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    if (period == 1) {
        CUDA_CHECK(cudaMemcpyAsync(output, input, size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        return;
    }

    const int window = period;
    const int blockSize = cfg.blockSize;

    // invSum = 1 / (period*(period+1)/2); weights 1..P are computed in-kernel.
    const double denom = 0.5 * static_cast<double>(window) * static_cast<double>(window + 1);
    const float invSum = static_cast<float>(1.0 / denom);

    bool launched = false;

    // Force prefix for windows >= threshold if preferred
    bool forcePrefix = (window >= cfg.prefixPreferThreshold) && cfg.preferPrefixForLarge;

    // Strategy 1: Small-window shared-memory
    if (!launched && !forcePrefix && window <= cfg.smallWindowThreshold) {
        size_t shm = static_cast<size_t>(blockSize + window - 1) * sizeof(float);
        if (static_cast<int>(shm) <= cfg.maxSharedBytes) {
            int grid = (size + blockSize - 1) / blockSize;
            wmaKernelSmallWindow<<<grid, blockSize, shm, stream>>>(input, output, invSum, size, window);
            CUDA_CHECK(cudaGetLastError());
            launched = true;
        }
    }

    // Strategy 2: Warp-optimized for medium windows
    if (!launched && !forcePrefix && cfg.enableWarp && window <= cfg.warpWindowThreshold) {
        int warpsPerBlock = blockSize >> 5;
        int grid = (warpsPerBlock > 0) ? ((size + warpsPerBlock - 1) / warpsPerBlock) : 0;
        if (grid > 0) {
            wmaKernelWarpOptimized<<<grid, blockSize, 0, stream>>>(input, output, invSum, size, window);
            CUDA_CHECK(cudaGetLastError());
            launched = true;
        }
    }

    // Strategy 3: Unrolled scalar path when prefix disabled
    if (!launched && !forcePrefix && cfg.enableVector && window >= 16) {
        int grid = (size + blockSize - 1) / blockSize;
        wmaKernelVectorized<<<grid, blockSize, 0, stream>>>(input, output, invSum, size, window);
        CUDA_CHECK(cudaGetLastError());
        launched = true;
    }

    // Strategy 4: Prefix-sum O(N) for large windows or fallback exact path
    if (!launched) {
        auto prefixX = acquireDeviceBufferAsync<double>(size, stream);
        auto prefixIX = acquireDeviceBufferAsync<double>(size, stream);
        dim3 b = defaultBlock();
        dim3 g = defaultGrid(size);
        castFloatToDoubleKernel<<<g, b, 0, stream>>>(input, prefixX.get(), size);
        CUDA_CHECK(cudaGetLastError());
        buildIndexTimesKernel<<<g, b, 0, stream>>>(input, prefixIX.get(), size);
        CUDA_CHECK(cudaGetLastError());
        {
            thrust::device_ptr<double> px(prefixX.get());
            thrust::inclusive_scan(thrust::cuda::par.on(stream), px, px + size, px);
        }
        {
            thrust::device_ptr<double> pix(prefixIX.get());
            thrust::inclusive_scan(thrust::cuda::par.on(stream), pix, pix + size, pix);
        }
        wmaFinalizeFromPrefixKernel<<<g, b, 0, stream>>>(prefixX.get(), prefixIX.get(), output, window, size);
    CUDA_CHECK(cudaGetLastError());
}
}

} // namespace tacuda
