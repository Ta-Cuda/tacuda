/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <stdexcept>
#include <indicators/APO.h>
#include <indicators/MA.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>

namespace {
__global__ void apoKernel(const float* __restrict__ fastMa,
                          const float* __restrict__ slowMa,
                          float* __restrict__ output,
                          int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    float f = fastMa[idx];
    float s = slowMa[idx];
    output[idx] = (isnan(f) || isnan(s)) ? nanf("") : (f - s);
}
} // namespace

tacuda::APO::APO(int fastPeriod, int slowPeriod, tacuda::MAType type)
    : fastPeriod(fastPeriod), slowPeriod(slowPeriod), type(type) {}

void tacuda::APO::calculate(const float* input, float* output, int size, cudaStream_t stream) noexcept(false) {
    if (fastPeriod <= 0 || slowPeriod <= 0) {
        throw std::invalid_argument("APO: invalid periods");
    }
    if (size <= 0) {
        return;
    }
    // TA-Lib TA_INT_PO normalizes reversed periods (slow = max) instead of
    // rejecting them.
    const int fastP = fastPeriod < slowPeriod ? fastPeriod : slowPeriod;
    const int slowP = fastPeriod < slowPeriod ? slowPeriod : fastPeriod;

    auto fastMa = acquireDeviceBufferAsync<float>(size, stream);
    auto slowMa = acquireDeviceBufferAsync<float>(size, stream);

    tacuda::MA fast(fastP, type);
    tacuda::MA slow(slowP, type);
    fast.calculate(input, fastMa.get(), size, stream);
    slow.calculate(input, slowMa.get(), size, stream);

    dim3 block = defaultBlock();
    dim3 grid = defaultGrid(size);
    apoKernel<<<grid, block, 0, stream>>>(fastMa.get(), slowMa.get(), output, size);
    CUDA_CHECK(cudaGetLastError());
}
