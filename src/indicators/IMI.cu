/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <indicators/IMI.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <stdexcept>

namespace {
// Double accumulation mirrors TA-Lib's double math over the same
// float32-quantized inputs.
__global__ void intradayDiffKernel(const float* __restrict__ open,
                                   const float* __restrict__ close,
                                   double* __restrict__ up,
                                   double* __restrict__ down,
                                   int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double diff = static_cast<double>(close[idx]) - static_cast<double>(open[idx]);
        if (diff > 0.0) {
            up[idx] = diff;
            down[idx] = 0.0;
        } else {
            up[idx] = 0.0;
            down[idx] = -diff;
        }
    }
}

// TA-Lib-style alignment: the window [end-period+1, end] is reported at
// `end`, with a leading warm-up of period-1 NaNs. Like ta_IMI.c the
// division is unguarded: a fully flat window yields 0/0 = NaN.
__global__ void imiKernel(const double* __restrict__ prefixUp,
                          const double* __restrict__ prefixDown,
                          float* __restrict__ output,
                          int period, int size) {
    int end = blockIdx.x * blockDim.x + threadIdx.x;
    if (end >= size || end < period - 1) return;
    int start = end - period + 1;
    double upPrev = (start == 0) ? 0.0 : prefixUp[start - 1];
    double downPrev = (start == 0) ? 0.0 : prefixDown[start - 1];
    double sumUp = prefixUp[end] - upPrev;
    double sumDown = prefixDown[end] - downPrev;
    output[end] = static_cast<float>(100.0 * (sumUp / (sumUp + sumDown)));
}
} // namespace

tacuda::IMI::IMI(int period) : period(period) {}

void tacuda::IMI::calculate(const float* open, const float* close, float* output,
                            int size, cudaStream_t stream) noexcept(false) {
    // TA-Lib bounds for optInTimePeriod: 2..100000 (BAD_PARAM otherwise).
    if (period < 2 || period > 100000) {
        throw std::invalid_argument("IMI: period must be in [2, 100000]");
    }
    if (period > size) {
        // TA-Lib succeeds with zero output elements when the input is
        // shorter than the lookback; the whole line is warm-up NaNs.
        CUDA_CHECK(cudaMemsetAsync(output, 0xFF, static_cast<size_t>(size) * sizeof(float), stream));
        return;
    }
    CUDA_CHECK(cudaMemsetAsync(output, 0xFF, size * sizeof(float), stream));

    auto up = acquireDeviceBufferAsync<double>(size, stream);
    auto down = acquireDeviceBufferAsync<double>(size, stream);

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    intradayDiffKernel<<<grid, block, 0, stream>>>(open, close, up.get(), down.get(), size);
    CUDA_CHECK(cudaGetLastError());

    thrust::device_ptr<double> upPtr(up.get());
    thrust::device_ptr<double> downPtr(down.get());
    thrust::inclusive_scan(thrust::cuda::par.on(stream), upPtr, upPtr + size, upPtr);
    thrust::inclusive_scan(thrust::cuda::par.on(stream), downPtr, downPtr + size, downPtr);

    imiKernel<<<grid, block, 0, stream>>>(up.get(), down.get(), output, period, size);
    CUDA_CHECK(cudaGetLastError());
}

void tacuda::IMI::calculate(const float* input, float* output, int size,
                            cudaStream_t stream) noexcept(false) {
    const float* open = input;
    // Expects input to be a contiguous array of floats in column-major OHLC layout:
    // [O1, O2, ..., On, H1, ..., Hn, L1, ..., Ln, C1, ..., Cn], each section of length 'size'.
    // Only Open and Close are used: 'open' points to the start, 'close' points to input + 3 * size.
    const float* close = input + 3 * size;
    calculate(open, close, output, size, stream);
}
