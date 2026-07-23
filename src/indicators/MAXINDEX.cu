/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/MAXINDEX.h>
#include <utils/CudaUtils.h>
#include <stdexcept>
#include <cmath>

// Exact TA-Lib state machine (ta_MAXINDEX.c). Tie handling is
// path-dependent there: a stale extremum triggers a left-to-right recompute
// where the OLDEST tie wins (`tmp > highest`), while an incoming bar
// replaces on equality (`tmp >= highest`, newest wins). A stateless
// per-window kernel cannot reproduce this, so the series is walked
// sequentially by a single thread.
__global__ void maxIndexKernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int period, int size) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int highestIdx = -1;
    float highest = 0.0f;
    int trailingIdx = 0;
    for (int today = period - 1; today < size; ++today, ++trailingIdx) {
        float tmp = input[today];
        if (highestIdx < trailingIdx) {
            highestIdx = trailingIdx;
            highest = input[highestIdx];
            for (int i = highestIdx + 1; i <= today; ++i) {
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

