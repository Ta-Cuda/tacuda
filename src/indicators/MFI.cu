/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/MFI.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <stdexcept>
#include <cmath>

// --- Helper kernel for final step: prefix difference -> MFI ---
__global__ void mfiFromPrefixesKernel(const double* __restrict__ posPrefix,
                                      const double* __restrict__ negPrefix,
                                      float* __restrict__ output,
                                      int period,
                                      int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    // warm start as in TA-Lib: first period elements = NaN
    if (i < period) {
        output[i] = nanf("");
        return;
    }

    double leftPos = (i >= period) ? posPrefix[i - period] : 0.0;
    double leftNeg = (i >= period) ? negPrefix[i - period] : 0.0;
    double posSum = posPrefix[i] - leftPos;
    double negSum = negPrefix[i] - leftNeg;

    double total = posSum + negSum;
    float mfi = (total < 1.0)
                    ? 0.0f
                    : static_cast<float>(100.0 * (posSum / total));
    output[i] = mfi;
}

// --- Functors for transform_iterator over indices ---
struct PosMFFunctor {
    const double* __restrict__ tp;
    const float* __restrict__ vol;

    __host__ __device__ double operator()(int i) const {
        if (i <= 0) return 0.0; // no comparison on first bar
        double tp_curr = tp[i];
        double tp_prev = tp[i - 1];
        if (!isfinite(tp_curr) || !isfinite(tp_prev)) return 0.0;
        double mf = tp_curr * static_cast<double>(vol[i]);
        return (tp_curr > tp_prev) ? mf : 0.0;
    }
};

struct NegMFFunctor {
    const double* __restrict__ tp;
    const float* __restrict__ vol;

    __host__ __device__ double operator()(int i) const {
        if (i <= 0) return 0.0;
        double tp_curr = tp[i];
        double tp_prev = tp[i - 1];
        if (!isfinite(tp_curr) || !isfinite(tp_prev)) return 0.0;
        double mf = tp_curr * static_cast<double>(vol[i]);
        return (tp_curr < tp_prev) ? mf : 0.0;
    }
};

// --- Compute Typical Price = (H+L+C)/3 in separate pass ---
__global__ void typicalPriceKernel(const float* __restrict__ high,
                                   const float* __restrict__ low,
                                   const float* __restrict__ close,
                                   double* __restrict__ tp,
                                   int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double h = static_cast<double>(high[i]);
    double l = static_cast<double>(low[i]);
    double c = static_cast<double>(close[i]);
    // For NaN/Inf can return NaN — it will zero out contribution via functors
    tp[i] = (h + l + c) / 3.0;
}

namespace tacuda {

MFI::MFI(int period) : period(period) {}

void MFI::calculate(const float* high, const float* low, const float* close,
                    const float* volume, float* output, int size,
                    cudaStream_t stream) noexcept(false)
{
    if (period < 2) {
        throw std::invalid_argument("MFI: invalid period");
    }
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }

    // 1) Typical Price on device
    auto tp = acquireDeviceBufferAsync<double>(size, stream);
    {
        dim3 block = defaultBlock();
        dim3 grid  = defaultGrid(size);
        typicalPriceKernel<<<grid, block, 0, stream>>>(high, low, close, tp.get(), size);
        CUDA_CHECK(cudaGetLastError());
    }

    // 2) Prefix sums of positive/negative money flows (double for stability)
    auto posPrefix = acquireDeviceBufferAsync<double>(size, stream);
    auto negPrefix = acquireDeviceBufferAsync<double>(size, stream);

    // transform_iterator over counting_iterator -> mf(i)
    auto pol = thrust::cuda::par.on(stream);

    auto c0   = thrust::make_counting_iterator<int>(0);
    auto cend = thrust::make_counting_iterator<int>(size);

    auto posIt = thrust::make_transform_iterator(c0,  PosMFFunctor{tp.get(), volume});
    auto negIt = thrust::make_transform_iterator(c0,  NegMFFunctor{tp.get(), volume});

    // inclusive_scan(posIt) -> posPrefix
    thrust::inclusive_scan(pol, posIt, posIt + size, thrust::device_pointer_cast(posPrefix.get()));
    thrust::inclusive_scan(pol, negIt, negIt + size, thrust::device_pointer_cast(negPrefix.get()));

    // 3) Final kernel: prefix difference over window -> MFI
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream)); // NaN at start
    {
        dim3 block = defaultBlock();
        dim3 grid  = defaultGrid(size);
        mfiFromPrefixesKernel<<<grid, block, 0, stream>>>(posPrefix.get(), negPrefix.get(),
                                                          output, period, size);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Combined overload: [H | L | C | V] consecutively
void MFI::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    const float* high   = input;
    const float* low    = input + size;
    const float* close  = input + 2 * size;
    const float* volume = input + 3 * size;
    calculate(high, low, close, volume, output, size, stream);
}

} // namespace tacuda
