/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#pragma once

// A fixed tree within each tile, followed by a recursively scanned array of
// tile totals. Unlike device-wide decoupled look-back, the arithmetic order
// never depends on which CUDA block happens to finish first. Scratch remains
// stream ordered and CUDA Graph capture uses the normal graph-owned pool.
#include <cub/block/block_scan.cuh>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <thrust/functional.h>

namespace tacuda { namespace detail {
namespace deterministic_scan_detail {
constexpr int kTile = 256;

template <typename T, typename Iterator, typename Op>
__global__ void tiles(Iterator input, T* output, T* totals, int count, Op op, T identity) {
    using Scan = cub::BlockScan<T, kTile, cub::BLOCK_SCAN_RAKING>;
    __shared__ typename Scan::TempStorage storage;
    const size_t i = static_cast<size_t>(blockIdx.x) * kTile + threadIdx.x;
    const T value = i < static_cast<size_t>(count) ? input[i] : identity;
    T result, aggregate;
    Scan(storage).InclusiveScan(value, result, op, aggregate);
    if (i < static_cast<size_t>(count)) output[i] = result;
    if (totals && threadIdx.x == 0) totals[blockIdx.x] = aggregate;
}

template <typename T, typename Op>
__global__ void offsets(T* output, const T* totals, int count, Op op) {
    const size_t i = static_cast<size_t>(blockIdx.x) * kTile + threadIdx.x;
    if (blockIdx.x > 0 && i < static_cast<size_t>(count)) {
        output[i] = op(totals[blockIdx.x - 1], output[i]);
    }
}
}

template <typename T, typename Iterator, typename Op>
inline void deterministic_inclusive_scan(Iterator input, T* output, int count,
                                          cudaStream_t stream, Op op, T identity) {
    if (count <= 0) return;
    using namespace deterministic_scan_detail;
    const int blocks = static_cast<int>((static_cast<size_t>(count) + kTile - 1) / kTile);
    if (blocks == 1) {
        tiles<T><<<1, kTile, 0, stream>>>(input, output, nullptr, count, op, identity);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    auto totals = acquireDeviceBufferAsync<T>(blocks, stream);
    tiles<T><<<blocks, kTile, 0, stream>>>(input, output, totals.get(), count, op, identity);
    CUDA_CHECK(cudaGetLastError());
    deterministic_inclusive_scan<T>(totals.get(), totals.get(), blocks, stream, op, identity);
    offsets<T><<<blocks, kTile, 0, stream>>>(output, totals.get(), count, op);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Iterator>
inline void deterministic_double_scan(Iterator input, double* output, int count, cudaStream_t stream) {
    deterministic_inclusive_scan<double>(input, output, count, stream, thrust::plus<double>{}, 0.0);
}

}}  // namespace tacuda::detail
