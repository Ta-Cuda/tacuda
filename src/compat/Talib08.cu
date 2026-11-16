/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <utils/Talib08.h>
#include <utils/DeviceBufferPool.h>
#include <utils/ErrorState.h>
#include <utils/UnsupportedOperation.h>
#include "Talib08Formulas.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tacuda { namespace talib08 {
#define TA08_HD __host__ __device__ inline
#include "talib08_dispatch.inc"
#undef TA08_HD

namespace {
constexpr size_t kScratchBudget = 256u << 20;
constexpr size_t kDoublesPerBar = 64;
static_assert(sizeof(int) == 4, "TA integer outputs require signed 32-bit storage");

struct Columns { const float* in[6] = {}; void* out[4] = {}; };

__global__ void calculate(Function function, Columns columns, Params params,
                           int length, int series, Info info,
                           unsigned char* workspace, size_t stride, int workers) {
    const int worker = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker >= workers) return;
    for (size_t row = worker; row < static_cast<size_t>(series); row += workers) {
        ref::Arena arena{workspace + static_cast<size_t>(worker) * stride, stride};
        const size_t offset = static_cast<size_t>(row) * length;
        const double* inputs[6] = {};
        void* outputs[4] = {};
        for (int j = 0; j < info.inputs; ++j) {
            double* target = arena.allocate(static_cast<size_t>(length) * sizeof(double));
            inputs[j] = target;
            for (int i = 0; i < length; ++i) target[i] = columns.in[j][offset + i];
        }
        for (int j = 0; j < info.outputs; ++j) {
            const bool integer = (info.integerMask & (1 << j)) != 0;
            outputs[j] = arena.allocate(static_cast<size_t>(length) * (integer ? sizeof(int) : sizeof(double)));
            for (int i = 0; i < length; ++i) {
                if (integer) static_cast<int*>(columns.out[j])[offset + i] = 0;
                else static_cast<float*>(columns.out[j])[offset + i] = nanf("");
            }
        }
        // The NumPy TA-Lib API skips leading NaNs common to the required
        // columns before calling the native batch function. Like that wrapper,
        // retain its final bar on all-NaN input (some cumulative functions
        // publish a seed there). Middle NaNs keep
        // the native function's propagation/recovery semantics.
        int prefix = 0;
        for (; prefix < length - 1; ++prefix) {
            bool anyNan = false;
            for (int j = 0; j < info.inputs; ++j) anyNan |= isnan(inputs[j][prefix]);
            if (!anyNan) break;
        }
        if (prefix == length) continue;
        for (int j = 0; j < info.inputs; ++j) inputs[j] += prefix;
        int begin = 0, count = 0;
        const auto status = evaluate(function, arena, inputs, outputs, length - prefix, &begin, &count, params);
        // The fixed arena covers inputs/outputs and the selected batch call
        // graph. Allocation never falls back to the CUDA device heap.
        assert(status == TA_SUCCESS && !arena.failed);
        if (status != TA_SUCCESS || arena.failed) continue;
        for (int j = 0; j < info.outputs; ++j) {
            const bool integer = (info.integerMask & (1 << j)) != 0;
            for (int i = 0; i < count; ++i) {
                if (integer) static_cast<int*>(columns.out[j])[offset + prefix + begin + i] =
                    static_cast<int*>(outputs[j])[i] + ((function == Function::ZIGZAG && j == 2) ? prefix : 0);
                else static_cast<float*>(columns.out[j])[offset + prefix + begin + i] = static_cast<float>(static_cast<double*>(outputs[j])[i]);
            }
        }
    }
}

ctStatus_t exception_status() {
    try { throw; }
    catch (const detail::UnsupportedOperation& e) { detail::set_last_error(e.what()); return CT_STATUS_UNSUPPORTED; }
    catch (const std::bad_alloc& e) { detail::set_last_error(e.what()); return CT_STATUS_ALLOC_FAILED; }
    catch (const std::exception& e) { detail::set_last_error(e.what()); return CT_STATUS_DEVICE_ERROR; }
    catch (...) { detail::set_last_error("TA-Lib 0.8.3 dispatch failed"); return CT_STATUS_DEVICE_ERROR; }
}

bool valid(Function f, const float* const* in, void* const* out,
           int length, int series, const Params& params) {
    if (static_cast<unsigned>(f) >= static_cast<unsigned>(Function::Count) ||
        length <= 0 || length > TA_INDEX_MAX || series <= 0 || !in || !out ||
        static_cast<size_t>(series) > std::numeric_limits<size_t>::max() / (sizeof(double) * length)) return false;
    const auto info = infos[static_cast<int>(f)];
    for (int i = 0; i < info.inputs; ++i) if (!in[i]) return false;
    for (int i = 0; i < info.outputs; ++i) if (!out[i]) return false;
    ref::Arena unused{nullptr, 0};
    return lookback(f, unused, params) >= 0;
}
}

ctStatus_t run_device(Function function, const float* const* inputs,
                      void* const* outputs, int length, int series,
                      const Params& params, cudaStream_t stream) {
    detail::clear_last_error();
    if (!valid(function, inputs, outputs, length, series, params)) {
        detail::set_last_error("TA-Lib 0.8.3: invalid pointer, shape or option");
        return CT_STATUS_INVALID_ARGUMENT;
    }
    try {
        const size_t rowBytes = (static_cast<size_t>(length) * kDoublesPerBar + 512) * sizeof(double);
        const int workers = static_cast<int>(std::min<size_t>(series, std::max<size_t>(1, kScratchBudget / rowBytes)));
        auto workspace = acquireDeviceBufferAsync<unsigned char>(rowBytes * workers, stream);
        const Info info = infos[static_cast<int>(function)];
        Columns columns;
        for (int i = 0; i < info.inputs; ++i) columns.in[i] = inputs[i];
        for (int i = 0; i < info.outputs; ++i) columns.out[i] = outputs[i];
        calculate<<<(workers + 31) / 32, 32, 0, stream>>>(function, columns, params, length, series,
                                                         info, workspace.get(), rowBytes, workers);
        cudaError_t error = cudaPeekAtLastError();
        if (error == cudaSuccess && stream == nullptr) error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess) {
            detail::set_last_error(cudaGetErrorString(error));
            return CT_STATUS_DEVICE_ERROR;
        }
        return CT_STATUS_SUCCESS;
    } catch (...) { return exception_status(); }
}

ctStatus_t run_host(Function function, const float* const* inputs,
                    void* const* outputs, int length, int series,
                    const Params& params, cudaStream_t stream) {
    detail::clear_last_error();
    if (!valid(function, inputs, outputs, length, series, params)) {
        detail::set_last_error("TA-Lib 0.8.3: invalid pointer, shape or option");
        return CT_STATUS_INVALID_ARGUMENT;
    }
    if (detail::stream_is_capturing(stream)) {
        detail::set_last_error("TA-Lib 0.8.3 host calls require completion; capture the device API over resident buffers");
        return CT_STATUS_UNSUPPORTED;
    }
    try {
        const Info info = infos[static_cast<int>(function)];
        const size_t bytes = static_cast<size_t>(length) * series * sizeof(float);
        std::vector<DeviceBufferPtrAsync<float>> deviceInputs, deviceOutputs;
        Columns columns;
        auto check = [](cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); };
        for (int i = 0; i < info.inputs; ++i) {
            deviceInputs.push_back(acquireDeviceBufferAsync<float>(bytes / sizeof(float), stream));
            columns.in[i] = deviceInputs.back().get();
            check(cudaMemcpyAsync(deviceInputs.back().get(), inputs[i], bytes, cudaMemcpyHostToDevice, stream));
        }
        for (int i = 0; i < info.outputs; ++i) {
            deviceOutputs.push_back(acquireDeviceBufferAsync<float>(bytes / sizeof(float), stream));
            columns.out[i] = deviceOutputs.back().get();
        }
        ctStatus_t status = run_device(function, columns.in, columns.out, length, series, params, stream);
        if (status != CT_STATUS_SUCCESS) { (void)cudaStreamSynchronize(stream); return status; }
        for (int i = 0; i < info.outputs; ++i)
            check(cudaMemcpyAsync(outputs[i], columns.out[i], bytes, cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        return CT_STATUS_SUCCESS;
    } catch (...) { (void)cudaStreamSynchronize(stream); return exception_status(); }
}
}}
