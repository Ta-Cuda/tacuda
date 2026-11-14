/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_DEVICE_SEED_CUH
#define TACUDA_DEVICE_SEED_CUH

/**
 * @file DeviceSeed.cuh
 * @brief Recurrence seeds computed on the device.
 *
 * Wilder/EMA recurrences start from a seed: the sum or mean of the first
 * window. Computing it with thrust::transform_reduce returns the value to the
 * host (a synchronizing device-to-host copy), and writing it back into the
 * output with cudaMemcpyAsync from a host variable is pageable and therefore
 * not capturable into a CUDA graph either. Here the seed stays on the device:
 * one warp sums the window and the kernels that consume it read it through a
 * pointer.
 */

#include <cuda_runtime.h>

#include <utils/CudaUtils.h>

namespace tacuda {
namespace detail {

/// seed = sum(x[begin .. begin+count)) / divisor, accumulated in double.
/// Writes the double to *seedOut and, when y != nullptr, float(seed) to
/// y[yIdx]. A non-positive count yields 0 (an empty window).
template <typename T>
__global__ void seedSumKernel(const T* __restrict__ x, int begin, int count,
                              double divisor, double* __restrict__ seedOut,
                              float* __restrict__ y, int yIdx) {
    double s = 0.0;
    for (int i = threadIdx.x; i < count; i += warpSize) {
        s += static_cast<double>(x[begin + i]);
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, offset);
    }
    if (threadIdx.x == 0) {
        const double seed = s / divisor;
        *seedOut = seed;
        if (y) y[yIdx] = static_cast<float>(seed);
    }
}

inline void launch_seed_sum(const float* x, int begin, int count, double divisor,
                            double* seedOut, float* y, int yIdx,
                            cudaStream_t stream) {
    seedSumKernel<float><<<1, 32, 0, stream>>>(x, begin, count, divisor, seedOut, y, yIdx);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace detail
}  // namespace tacuda

#endif  // TACUDA_DEVICE_SEED_CUH
