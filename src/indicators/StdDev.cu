/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/StdDev.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/DeterministicScan.cuh>
#include <utils/WindowStdDev.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cfloat>
#include <stdexcept>

namespace {

__global__ void directStdDev(const float* input, float* output, int size, int period) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(size)) {
        output[i] = i < static_cast<size_t>(period - 1)
            ? __int_as_float(0xFFFFFFFF)
            : tacuda::detail::direct_window_stddev(input + i - (period - 1), period);
    }
}

struct Moments {
    double mean;
    double m2;
    int count;
    int segment;
};

__host__ __device__ Moments merge(Moments a, Moments b) {
    if (a.count == 0) return b;
    if (b.count == 0) return a;
    const int count = a.count + b.count;
    const double delta = b.mean - a.mean;
    const double fraction = static_cast<double>(b.count) / count;
    return {a.mean + delta * fraction,
            a.m2 + b.m2 + delta * delta * (a.count * fraction),
            count, b.segment};
}

struct MergeWithinSegment {
    __host__ __device__ Moments operator()(Moments a, Moments b) const {
        return a.count && b.count && a.segment != b.segment ? b : merge(a, b);
    }
};

template <bool Backwards>
struct SegmentInput {
    const float* input;
    int size;
    int period;

    __host__ __device__ Moments operator()(int i) const {
        const int segment = i / period;
        int index = i;
        if constexpr (Backwards) {
            const int begin = segment * period;
            const int available = size - begin;
            const int width = period < available ? period : available;
            index = begin + (width - 1 - (i - begin));
        }
        const double value = input[index];
        // Preserve the missing sample in every moment that contains it.
        const double m2 = value >= -FLT_MAX && value <= FLT_MAX ? 0.0 : value - value;
        return {value, m2, 1, segment};
    }
};

__global__ void finishWideStdDev(const Moments* prefixes, const Moments* suffixes,
                                  float* output, int size, int period) {
    const size_t end = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (end >= static_cast<size_t>(size)) return;
    if (end < static_cast<size_t>(period - 1)) {
        output[end] = __int_as_float(0xFFFFFFFF);
        return;
    }
    const int last = static_cast<int>(end);
    const int first = last - period + 1;
    // A window the size of a segment occupies either one whole segment
    // or a suffix of one and a prefix of the next.
    const int within = first % period;
    Moments window = prefixes[last];
    if (within != 0) {
        const int suffixIndex = first - within + (period - 1 - within);
        window = merge(suffixes[suffixIndex], window);
    }
    output[end] = static_cast<float>(sqrt(window.m2 / period));
}

}  // namespace

tacuda::StdDev::StdDev(int period) : period(period) {}

void tacuda::StdDev::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) throw std::invalid_argument("StdDev: invalid period");
    if (period > size) {
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    // The fused batch dispatcher uses direct windows for its single launch.
    // On a long standalone row the scans also pay off for medium periods.
    constexpr int directLimit = 24;
    if (period <= directLimit) {
        directStdDev<<<defaultGrid(size), defaultBlock(), 0, stream>>>(input, output, size, period);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    auto workspace = acquireDeviceBufferAsync<Moments>(static_cast<size_t>(size) * 2, stream);
    auto* prefixes = workspace.get();
    auto* suffixes = prefixes + size;
    const thrust::counting_iterator<int> indices(0);
    const auto forward = thrust::make_transform_iterator(indices, SegmentInput<false>{input, size, period});
    const auto backward = thrust::make_transform_iterator(indices, SegmentInput<true>{input, size, period});
    detail::deterministic_inclusive_scan<Moments>(forward, prefixes, size, stream, MergeWithinSegment{}, Moments{});
    detail::deterministic_inclusive_scan<Moments>(backward, suffixes, size, stream, MergeWithinSegment{}, Moments{});
    finishWideStdDev<<<defaultGrid(size), defaultBlock(), 0, stream>>>(prefixes, suffixes, output, size, period);
    CUDA_CHECK(cudaGetLastError());
}
