/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/ADOSC.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/AffineScan.cuh>

#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <stdexcept>
#include <cmath>

// ============================================================
// AD Term Logic (same as AD.cu)
// ============================================================
struct ADTerm {
    const float* __restrict__ h;
    const float* __restrict__ l;
    const float* __restrict__ c;
    const float* __restrict__ v;

    __device__ double operator()(int i) const {
        double hi = static_cast<double>(h[i]);
        double lo = static_cast<double>(l[i]);
        double cl = static_cast<double>(c[i]);
        double vol = static_cast<double>(v[i]);

        if (!isfinite(hi) || !isfinite(lo) || !isfinite(cl) || !isfinite(vol)) {
            return 0.0;
        }

        double range = hi - lo;
        if (!(range > 0.0)) {
            return 0.0;
        }

        double moneyFlowMultiplier = ((cl - lo) - (hi - cl)) / range;
        return moneyFlowMultiplier * vol;
    }
};

// ============================================================
// EMA Parallel Scan Logic
// ============================================================
using AffinePair = tacuda::detail::AffineTransform1<double>;
using AffineCompose = tacuda::detail::AffineCompose1<double>;

// Generates AffinePairs for the scan input
// y[i] = alpha * x[i] + (1-alpha) * y[i-1]
// For i=0: y[0] = x[0] => A=0, b=x[0].
// For i>0: A=1-alpha, b=alpha*x[i].
struct EmaGenerator {
    const double* __restrict__ input; // The AD line
    double alpha;

    __device__ AffinePair operator()(int i) const {
        double val = input[i];
        
        if (i == 0) {
            return {0.0, val};
        } else {
            return {1.0 - alpha, alpha * val};
        }
    }
};

// ============================================================
// Final Combination Kernel
// ============================================================
__global__ void adoscDiffKernel(const AffinePair* __restrict__ emaFast,
                                const AffinePair* __restrict__ emaSlow,
                                float* __restrict__ output,
                                int size,
                                int lookback)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    if (i < lookback) {
        output[i] = nanf("");
    } else {
        double fastVal = emaFast[i].b;
        double slowVal = emaSlow[i].b;
        output[i] = static_cast<float>(fastVal - slowVal);
    }
}

namespace tacuda {

ADOSC::ADOSC(int shortPeriod, int longPeriod)
  : shortPeriod(shortPeriod), longPeriod(longPeriod) {}

void ADOSC::calculate(const float* high, const float* low, const float* close,
                      const float* volume, float* output,
                      int size, cudaStream_t stream) noexcept(false)
{
    // TA-Lib bounds for both ADOSC periods: 2..100000 (BAD_PARAM otherwise).
    if (shortPeriod < 2 || longPeriod < 2 ||
        shortPeriod > 100000 || longPeriod > 100000) {
        throw std::invalid_argument("ADOSC: periods must be in [2, 100000]");
    }
    if (size <= 0) {
        throw std::invalid_argument("ADOSC: invalid size");
    }
    // TA-Lib keeps the period ROLES as given: ADOSC(10,3) = EMA(10) - EMA(3).
    // Only the lookback uses the slowest period (ta_ADOSC.c implementation
    // note gives users the freedom of "unusual parameter settings").
    const int fastP = shortPeriod;
    const int slowP = longPeriod;
    const int slowestP = shortPeriod > longPeriod ? shortPeriod : longPeriod;

    // 1. Calculate AD Line (Accumulation/Distribution)
    // Reuse AD logic: scan (mfm * vol)
    auto adLine = acquireDeviceBufferAsync<double>(size, stream);
    {
        auto idx = thrust::make_counting_iterator<int>(0);
        auto termIt = thrust::make_transform_iterator(idx, ADTerm{high, low, close, volume});
        
        // Inclusive scan to get cumulative AD
        thrust::inclusive_scan(thrust::cuda::par.on(stream),
                               termIt, termIt + size,
                               adLine.get());
    }

    // 2. Calculate EMA(Fast) and EMA(Slow) using parallel scan
    double alphaShort = 2.0 / (fastP + 1);
    double alphaLong  = 2.0 / (slowP + 1);

    // We need buffers for scan results (AffinePair)
    // We can't easily scan directly into 'b' because we need 'a' state during scan.
    auto emaFast = acquireDeviceBufferAsync<AffinePair>(size, stream);
    auto emaSlow = acquireDeviceBufferAsync<AffinePair>(size, stream);

    auto idx = thrust::make_counting_iterator<int>(0);

    // Fast EMA Scan
    {
        auto genFast = thrust::make_transform_iterator(idx, EmaGenerator{adLine.get(), alphaShort});
        thrust::inclusive_scan(thrust::cuda::par.on(stream),
                               genFast, genFast + size,
                               thrust::device_pointer_cast(emaFast.get()),
                               AffineCompose());
    }

    // Slow EMA Scan
    {
        auto genSlow = thrust::make_transform_iterator(idx, EmaGenerator{adLine.get(), alphaLong});
        thrust::inclusive_scan(thrust::cuda::par.on(stream),
                               genSlow, genSlow + size,
                               thrust::device_pointer_cast(emaSlow.get()),
                               AffineCompose());
    }

    // 3. Compute Difference
    // TA-Lib lookback: EMA lookback of the slowest period.
    int lookback = slowestP - 1;
    {
        dim3 block = defaultBlock();
        dim3 grid = defaultGrid(size);
        adoscDiffKernel<<<grid, block, 0, stream>>>(
            emaFast.get(),
            emaSlow.get(),
            output,
            size,
            lookback
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

void ADOSC::calculate(const float* input, float* output, int size,
                      cudaStream_t stream) noexcept(false)
{
    const float* high   = input;
    const float* low    = input + size;
    const float* close  = input + 2 * size;
    const float* volume = input + 3 * size;
    calculate(high, low, close, volume, output, size, stream);
}

} // namespace tacuda
