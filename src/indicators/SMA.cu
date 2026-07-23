/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/SMA.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/TacudaConfig.h>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/functional.h>
#include <algorithm>

namespace {

// ===================== Templated kernels =====================

// Finalize SMA from prefix-sum in double with end-of-window alignment
template<typename T>
__global__ void smaFinalizeFromPrefixKernel(const double* __restrict__ prefix,
                                            T* __restrict__ output,
                                            int period,
                                            int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    if (idx < period - 1) return; // warmup region stays NaN

    double sum = prefix[idx];
    if (idx >= period) sum -= prefix[idx - period];
    output[idx] = static_cast<T>(sum / static_cast<double>(period));
}

// Shared-memory tiled kernel for small windows
template<typename T>
__global__ void smaKernelSmallWindow(const T* __restrict__ input,
                                     T* __restrict__ output,
                                     int size,
                                     int window) {
    extern __shared__ char smem_raw[];
    T* tile = reinterpret_cast<T*>(smem_raw);

    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    int gid = blockStart + tid;

    int sharedCount = blockDim.x + window - 1;
    int tileStart = blockStart - (window - 1);
    for (int i = tid; i < sharedCount; i += blockDim.x) {
        int src = tileStart + i;
        tile[i] = (src >= 0 && src < size) ? input[src] : T(0);
    }
    __syncthreads();

    if (gid >= window - 1 && gid < size) {
        double sum = 0.0;
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
            if (k >= window) break;
            sum += static_cast<double>(tile[tid + k]);
        }
        for (int k = 32; k < window; ++k) {
            sum += static_cast<double>(tile[tid + k]);
        }
        output[gid] = static_cast<T>(sum / static_cast<double>(window));
    }
}

// Helpers for warp kernel
template<typename T>
__device__ __forceinline__ T ldg_typed(const T* ptr) {
    #if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
    #else
    return *ptr;
    #endif
}

__device__ __forceinline__ void prefetchL2_addr(const void* ptr) {
    #if __CUDA_ARCH__ >= 600
    asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));
    #endif
}

// Warp-optimized kernel for medium windows with L2 prefetch and 4x32 unrolling
template<typename T>
__global__ void smaKernelWarpOptimized(const T* __restrict__ input,
                                       T* __restrict__ output,
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

        for (int base = lane; base < window; base += 32 * 4) {
            int nextBase = base + 32 * 4;
            if (nextBase < window) {
                int pfStart = start + nextBase;
                if (pfStart + lane < size) prefetchL2_addr(input + pfStart + lane);
                if (pfStart + lane + 32 < size && nextBase + 32 < window) prefetchL2_addr(input + pfStart + lane + 32);
                if (pfStart + lane + 64 < size && nextBase + 64 < window) prefetchL2_addr(input + pfStart + lane + 64);
                if (pfStart + lane + 96 < size && nextBase + 96 < window) prefetchL2_addr(input + pfStart + lane + 96);
            }

            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                int ofs = base + u * 32;
                if (ofs < window) {
                    sum += static_cast<double>(ldg_typed(input + start + ofs));
                }
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            output[gid] = static_cast<T>(sum / static_cast<double>(window));
        }
    }
}

// Grid-stride fallback kernel for very large windows when prefix-sum is disabled
template<typename T>
__global__ void smaKernelLargeWindow(const T* __restrict__ input,
                                     T* __restrict__ output,
                                     int size,
                                     int window) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = gid; idx < size; idx += stride) {
        if (idx >= window - 1) {
            int start = idx - window + 1;
            double sum = 0.0;
            #pragma unroll 8
            for (int k = 0; k < 256; ++k) {
                if (k >= window) break;
                sum += static_cast<double>(input[start + k]);
            }
            for (int k = 256; k < window; ++k) {
                sum += static_cast<double>(input[start + k]);
            }
            output[idx] = static_cast<T>(sum / static_cast<double>(window));
        }
    }
}

template<typename T>
struct ToDoubleFunctor {
    __host__ __device__ double operator()(T v) const { return static_cast<double>(v); }
};

struct SMAKernelConfig {
    int blockSize = 256;
    int smallWindowThreshold = 64;
    int warpWindowThreshold = 256;
    int gridStrideFactor = 4;
    int maxGridSize = 65535;
    int maxSharedBytes = 48 * 1024;
    int prefixPreferThreshold = 96;
    bool preferPrefixForLarge = true;
    bool enableWarp = true;
};

static SMAKernelConfig cfg = []() {
    SMAKernelConfig c;
    cudaDeviceProp prop{};
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
        c.maxSharedBytes = static_cast<int>(prop.sharedMemPerBlock);
    }
    auto& gcfg = tacuda::Config::instance();
    c.blockSize = gcfg.sma.blockSize;
    c.smallWindowThreshold = gcfg.sma.smallWindowThreshold;
    c.warpWindowThreshold = gcfg.sma.warpWindowThreshold;
    c.gridStrideFactor = gcfg.sma.gridStrideFactor;
    if (gcfg.sma.maxSharedBytes > 0) c.maxSharedBytes = gcfg.sma.maxSharedBytes;
    c.preferPrefixForLarge = gcfg.sma.preferPrefixForLarge;
    c.enableWarp = gcfg.sma.enableWarp;
    c.prefixPreferThreshold = gcfg.sma.prefixPreferThreshold;
    return c;
}();

// ===================== Templated implementation =====================

template<typename T>
void sma_calculate_impl(int period, const T* input, T* output, int size, cudaStream_t stream) {
    if (period <= 0) {
        throw std::invalid_argument("SMA: invalid period");
    }
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }

    // Initialize with NaN for warmup
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(T), stream));

    if (period == 1) {
        CUDA_CHECK(cudaMemcpyAsync(output, input, size * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        return;
    }

    const int window = period;
    const int blockSize = cfg.blockSize;

    bool launched = false;

    bool forcePrefix = (window >= cfg.prefixPreferThreshold) && cfg.preferPrefixForLarge;

    // Strategy 1: Shared-memory tiled for small windows
    if (!launched && !forcePrefix && window <= cfg.smallWindowThreshold) {
        size_t shm = static_cast<size_t>(blockSize + window - 1) * sizeof(T);
        if (static_cast<int>(shm) <= cfg.maxSharedBytes) {
            int grid = (size + blockSize - 1) / blockSize;
            smaKernelSmallWindow<T><<<grid, blockSize, shm, stream>>>(input, output, size, window);
            CUDA_CHECK(cudaGetLastError());
            launched = true;
        }
    }

    // Strategy 2: Warp-optimized for medium windows
    if (!launched && !forcePrefix && cfg.enableWarp && window <= cfg.warpWindowThreshold) {
        int warpsPerBlock = blockSize >> 5;
        int grid = (warpsPerBlock > 0) ? ((size + warpsPerBlock - 1) / warpsPerBlock) : 0;
        if (grid > 0) {
            smaKernelWarpOptimized<T><<<grid, blockSize, 0, stream>>>(input, output, size, window);
            CUDA_CHECK(cudaGetLastError());
            launched = true;
        }
    }

    // Strategy 3: Grid-stride O(N*W) when prefix is disabled
    if (!launched && !forcePrefix && !cfg.preferPrefixForLarge && window < cfg.prefixPreferThreshold) {
        int grid = std::min(cfg.maxGridSize, std::max(1, (size + blockSize * cfg.gridStrideFactor - 1) / (blockSize * cfg.gridStrideFactor)));
        smaKernelLargeWindow<T><<<grid, blockSize, 0, stream>>>(input, output, size, window);
        CUDA_CHECK(cudaGetLastError());
        launched = true;
    }

    // Strategy 4: Double-precision prefix-sum (O(N))
    if (!launched) {
        auto prefixD = acquireDeviceBufferAsync<double>(size, stream);
        thrust::device_ptr<const T> inPtr(input);
        auto inBegin = thrust::make_transform_iterator(inPtr, ToDoubleFunctor<T>{});
        thrust::device_ptr<double> preD(prefixD.get());
        thrust::inclusive_scan(thrust::cuda::par.on(stream), inBegin, inBegin + size, preD);
        dim3 block = defaultBlock();
        dim3 grid = defaultGrid(size);
        smaFinalizeFromPrefixKernel<T><<<grid, block, 0, stream>>>(prefixD.get(), output, window, size);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

// ===================== Explicit instantiations =====================

namespace tacuda {

// --- SMAT<float> ---
template<>
void SMAT<float>::calculate(const float* input, float* output, int size, cudaStream_t stream) {
    sma_calculate_impl<float>(period, input, output, size, stream);
}

// --- SMAT<double> ---
template<>
void SMAT<double>::calculate(const double* input, double* output, int size, cudaStream_t stream) {
    sma_calculate_impl<double>(period, input, output, size, stream);
}

// --- Backward-compatible SMA (float, inherits from Indicator) ---
SMA::SMA(int period) : period(period) {}

void SMA::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    sma_calculate_impl<float>(period, input, output, size, stream);
}

// Force template class instantiation so vtable & typeinfo are emitted
template class SMAT<float>;
template class SMAT<double>;

} // namespace tacuda
