/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
/**
 * @file BatchRows.cu
 * @brief Shared helpers of the row-batched legacy families (see BatchRows.h).
 */

#include <utils/BatchRows.h>
#include <utils/CudaUtils.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace {

constexpr int kMaxPrefixOutputs = 3;

struct PrefixOutputs {
    float* out[kMaxPrefixOutputs] = {};
};

// Element k of the (output, row >= 1, t < count) grid copies row 0's value.
__global__ void copyRowPrefixKernel(PrefixOutputs o, int nOut, int length,
                                    int series, int count) {
    const size_t perOutput = static_cast<size_t>(series - 1) * count;
    const size_t total = perOutput * nOut;
    for (size_t k = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         k < total; k += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t j = k / perOutput;
        const size_t r = k - j * perOutput;
        const size_t row = 1 + r / count;
        const size_t t = r % count;
        float* out = o.out[j];
        out[row * static_cast<size_t>(length) + t] = out[t];
    }
}

}  // namespace

namespace tacuda {
namespace detail {

RowLaunch divergent_row_launch(int series, bool fp64) {
    // Device limits are queried per call: cheap, and correct when the caller
    // switches devices. RTX 3090 Ti, SAR (double), rows x bars = 2M:
    // one row per warp is 2.3x faster up to 250 rows, 1.2x at 350-500, and
    // 1.3x slower from 700 rows (4.5x at 4000) than 64 rows per block.
    int device = 0, sms = 0, threadsPerSm = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&threadsPerSm, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    const long long sparseLimit = fp64 ? 6LL * sms
                                       : static_cast<long long>(sms) * (threadsPerSm / 32);
    if (series <= sparseLimit) {
        constexpr int threads = 128;  // four warps, one row each
        const long long blocks = (static_cast<long long>(series) * 32 + threads - 1) / threads;
        return {static_cast<int>(std::max<long long>(1, std::min<long long>(blocks, 65535))),
                threads, 32};
    }
    return {row_blocks(series), kRowThreads, 1};
}

void copy_row_prefix(float* const* outs, int nOut, int length, int series,
                     int count, cudaStream_t stream) {
    if (series < 2 || count <= 0 || nOut <= 0) return;
    if (nOut > kMaxPrefixOutputs || count > length) {
        throw std::invalid_argument("copy_row_prefix: unsupported shape");
    }
    PrefixOutputs o;
    for (int j = 0; j < nOut; ++j) o.out[j] = outs[j];
    const size_t total = static_cast<size_t>(series - 1) * count * nOut;
    const int threads = 256;
    const int blocks = static_cast<int>(std::min<size_t>((total + threads - 1) / threads, 65535));
    copyRowPrefixKernel<<<blocks, threads, 0, stream>>>(o, nOut, length, series, count);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace detail
}  // namespace tacuda
