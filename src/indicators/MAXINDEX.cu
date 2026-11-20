/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/MAXINDEX.h>
#include <utils/BatchRows.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <cmath>

// TA-Lib 0.8.3 names the newest tie, including a stale-window recompute.
__device__ __forceinline__ void max_index_row(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              int period, int size) {
    int highestIdx = -1;
    float highest = 0.0f;
    int trailingIdx = 0;
    for (int today = period - 1; today < size; ++today, ++trailingIdx) {
        float tmp = input[today];
        if (highestIdx < trailingIdx) {
            highestIdx = today;
            highest = input[highestIdx];
            for (int i = today - 1; i >= trailingIdx; --i) {
                float v = input[i];
                if (v > highest) {
                    highestIdx = i;
                    highest = v;
                }
            }
        } else if (tmp >= highest) {
            highestIdx = today;
            highest = tmp;
        }
        // TA-Lib MAXINDEX returns the absolute index in the input series.
        output[today] = static_cast<float>(highestIdx);
    }
}

__global__ void maxIndexKernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int period, int size) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    max_index_row(input, output, period, size);
}

// Batch path: one GPU thread per row; indices stay relative to the row.
__global__ void maxIndexRowsKernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int period, int length, int series, int spread) {
    size_t r = 0, step = 0;
    if (!tacuda::detail::row_slot(spread, r, step)) return;
    for (; r < static_cast<size_t>(series); r += step) {
        const size_t off = r * static_cast<size_t>(length);
        max_index_row(input + off, output + off, period, length);
    }
}

void tacuda::detail::max_index_rows(const float* in, float* out, int length,
                                    int series, int period, cudaStream_t stream) {
    if (period <= 0) {
        throw std::invalid_argument("MAXINDEX: invalid period");
    }
    CUDA_CHECK(cudaMemsetAsync(out, 0xFF, static_cast<size_t>(length) * series * sizeof(float), stream));
    if (period > length) return;  // every row is warm-up
    const RowLaunch launch = divergent_row_launch(series, /*fp64=*/false);
    maxIndexRowsKernel<<<launch.blocks, launch.threads, 0, stream>>>(in, out, period, length, series, launch.spread);
    CUDA_CHECK(cudaGetLastError());
}

tacuda::MAXINDEX::MAXINDEX(int period) : period(period) {}

void tacuda::MAXINDEX::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (period <= 0) {
        throw std::invalid_argument("MAXINDEX: invalid period");
    }
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));
    // Sequential state machine: single thread reproduces TA-Lib exactly
    // (see kernel comment); batch paths parallelize across series.
    maxIndexKernel<<<1, 1, 0, stream>>>(input, output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

