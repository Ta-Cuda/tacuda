/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// indicators/AD.cu
#include <indicators/AD.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>

#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <stdexcept>
#include <cmath>

// Term generator t[i] = CLV[i] * Volume[i], exactly as in TA-Lib (double accu)
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

namespace tacuda {

void AD::calculate(const float* high, const float* low, const float* close,
                   const float* volume, float* output, int size,
                   cudaStream_t stream) noexcept(false)
{
    if (size <= 0) throw std::invalid_argument("AD: invalid size");

    // TA-Lib accumulates in double → we repeat, then cast down to float
    auto tmpD = acquireDeviceBufferAsync<double>(size, stream);

    {
        auto idx0   = thrust::make_counting_iterator<int>(0);
        auto termIt = thrust::make_transform_iterator(idx0, ADTerm{high, low, close, volume});

        thrust::inclusive_scan(thrust::cuda::par.on(stream),
                               termIt, termIt + size,
                               tmpD.get());
    }

    // Convert to float in output buffer
    thrust::transform(thrust::cuda::par.on(stream),
                      tmpD.get(), tmpD.get() + size,
                      output,
                      [] __device__ (double x) { return static_cast<float>(x); });
}

void AD::calculate(const float* input, float* output, int size,
                   cudaStream_t stream) noexcept(false)
{
    // Layout: [High | Low | Close | Volume], each block of length size
    const float* high   = input;
    const float* low    = input + size;
    const float* close  = input + 2 * size;
    const float* volume = input + 3 * size;
    calculate(high, low, close, volume, output, size, stream);
}

} // namespace tacuda
