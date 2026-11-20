/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
/**
 * @file api_batch.cu
 * @brief Batched (multi-symbol / portfolio) entry points.
 *
 * Strategy (host variants):
 *   1. Upload every input matrix (numSeries x seriesLen, series-major) in one
 *      bulk H2D copy on the base stream. Host matrices are temporarily
 *      page-locked when large enough for the registration cost to pay off.
 *   2. Launch one fused kernel for supported workloads, or run a row-batched
 *      legacy family over the whole matrix with its own arithmetic (RowsPlan);
 *      otherwise dispatch series round-robin across an internal pool of
 *      non-blocking streams.
 *   3. Gather results with one bulk D2H copy per output matrix and
 *      synchronize before returning (host-API semantics).
 *
 * Device variants (_batch_d) skip steps 1 and 3: matrices are already
 * GPU-resident. With stream == 0, fused and row-batched work runs on the
 * legacy default stream and the fallback pool first waits for preceding work
 * on it.
 */

#include <algorithm>
#include <climits>
#include <cstddef>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "tacuda.h"
#include "tacuda_device.h"
#include <utils/BatchRows.h>
#include <utils/DeviceBufferPool.h>
#include <utils/ErrorState.h>
#include <utils/UnsupportedOperation.h>
#include <utils/Talib08.h>
#include "BatchFused.h"
#include "BatchShort.h"

namespace {

using tacuda::detail::FusedCall;
using tacuda::detail::FusedFamily;
using tacuda::detail::fused_call;

constexpr int kMaxIo = 5;              // max input/output matrices per call (WOBV: OHLCV)
constexpr int kAutoStreamCap = 4;      // auto concurrency heuristic cap
constexpr size_t kPinThresholdBytes = 1u << 20;  // pin host memory from 1 MB

/**
 * Row-batched execution of a legacy family that keeps its arithmetic, so
 * every row is bit-identical to the per-series dispatcher:
 *
 * - Flat: the family's single-series entry point runs over blocks of whole
 *   rows as if each block were one series (one block unless the matrix
 *   exceeds kFlatChunk elements). Its kernels only read the window ending at
 *   the element they write, so every element at or past the row's first
 *   valid index `warmup` sees only its own row. The elements before it were
 *   computed from the previous row's tail and are restored from the block's
 *   first row, whose warm-up values do not depend on the data.
 * - Rows: a dedicated launcher runs the family's sequential single-series
 *   state machine with one GPU thread per row.
 *
 * Parameters the plan does not accept (`valid`), rows without a valid
 * element, a single row and TACUDA_BATCH_FUSED=0 keep the per-series path,
 * which also owns error reporting.
 */
struct RowsPlan {
    enum class Kind { None, Flat, Rows };
    Kind kind = Kind::None;
    bool valid = false;
    long long warmup = 0;   // Flat: first valid index of every row
    long long longRow = 0;  // Flat: see kLongRowSeries

    // Flat calls cover whole rows, at most kFlatChunk elements per call (or
    // one row when a row is longer): the stacked single-series wrappers
    // allocate scratch of up to 3 * n floats, which must stay bounded where
    // the per-series path needed a few kilobytes. A row must also fit the
    // int indexing (and 3 * n sizes) of the single-series kernels.
    static constexpr long long kFlatChunk = 1LL << 22;
    static constexpr long long kFlatRowLimit = INT_MAX / 4;
    // Flattening only removes launches. With fewer than kLongRowSeries rows
    // of at least `longRow` elements the per-series kernels already fill the
    // GPU and hide their launches; the flat call then adds its warm-up copy
    // (and, for composite families, loses the per-series stream overlap).
    // RTX 3090 Ti: up to 7% slower at 8 x 250000 (ACCBANDS 13% at 32 x 62500).
    static constexpr int kLongRowSeries = 64;
    static constexpr long long kLongRow = 1LL << 16;

    bool enabled(int seriesLen, int numSeries) const {
        if (kind == Kind::None || !valid || numSeries < 2) return false;
        if (tacuda::detail::batch_mode() == tacuda::detail::BatchMode::Off) return false;
        if (kind == Kind::Flat) {
            if (warmup < 0 || warmup >= seriesLen) return false;
            if (seriesLen > kFlatRowLimit) return false;
            if (numSeries < kLongRowSeries && seriesLen >= longRow) return false;
        }
        return true;
    }
};

inline RowsPlan rows_flat(bool valid, long long warmup,
                          long long longRow = RowsPlan::kLongRow) {
    return {RowsPlan::Kind::Flat, valid, warmup, longRow};
}
inline RowsPlan rows_kernel(bool valid) {
    return {RowsPlan::Kind::Rows, valid, 0, 0};
}

/// Placeholder whole-matrix launcher of families without a RowsPlan.
struct NoRows {
    ctStatus_t operator()(const float* const*, float* const*, int, cudaStream_t) const {
        return CT_STATUS_UNSUPPORTED;
    }
};

ctStatus_t classify_batch_exception(const char* context) {
    try { throw; }
    catch (const tacuda::detail::UnsupportedOperation& e) {
        tacuda::detail::set_last_error(e.what());
        return CT_STATUS_UNSUPPORTED;
    }
    catch (const std::invalid_argument& e) {
        tacuda::detail::set_last_error(e.what());
        return CT_STATUS_INVALID_ARGUMENT;
    }
    catch (const std::exception& e) {
        tacuda::detail::set_last_error(e.what());
        return CT_STATUS_DEVICE_ERROR;
    }
    catch (...) {
        tacuda::detail::set_last_error(context ? context : "batch dispatch failed");
        return CT_STATUS_KERNEL_FAILED;
    }
}

/// Process-wide pool of non-blocking streams reused by every batch call. A
/// stream belongs to the device it was created on, so each device has its own.
class BatchStreamPool {
public:
    static BatchStreamPool& instance() {
        static BatchStreamPool pool;
        return pool;
    }

    /// Returns `count` streams, creating them lazily. Thread-safe.
    std::vector<cudaStream_t> get(int count) {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) {
            return {};  // callers fall back to the default stream
        }
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<cudaStream_t>& streams = byDevice_[device];
        while (static_cast<int>(streams.size()) < count) {
            cudaStream_t s = nullptr;
            if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) {
                break;  // degrade gracefully to fewer streams
            }
            streams.push_back(s);
        }
        int usable = std::min<int>(count, static_cast<int>(streams.size()));
        return {streams.begin(), streams.begin() + usable};
    }

private:
    BatchStreamPool() = default;
    ~BatchStreamPool() {
        for (auto& entry : byDevice_) {
            for (cudaStream_t s : entry.second) {
                cudaStreamDestroy(s);
            }
        }
    }
    std::mutex mutex_;
    std::map<int, std::vector<cudaStream_t>> byDevice_;
};

/// RAII: temporarily page-lock a host matrix for faster async copies.
class HostPin {
public:
    HostPin() = default;
    void pin(const void* ptr, size_t bytes) {
        if (!ptr_ && ptr && bytes >= kPinThresholdBytes &&
            cudaHostRegister(const_cast<void*>(ptr), bytes,
                             cudaHostRegisterDefault) == cudaSuccess) {
            ptr_ = ptr;
        }
        // Registration failure is not an error: copies stay pageable.
    }
    ~HostPin() {
        if (ptr_) {
            cudaHostUnregister(const_cast<void*>(ptr_));
        }
    }
    HostPin(const HostPin&) = delete;
    HostPin& operator=(const HostPin&) = delete;

private:
    const void* ptr_ = nullptr;
};

/// RAII: device matrices from the shared buffer pool.
class DeviceMatrices {
public:
    ctStatus_t acquire(int count, size_t bytesEach) {
        for (int i = 0; i < count; ++i) {
            try {
                ptrs_[i] = static_cast<float*>(
                    DeviceBufferPool::instance().acquire(bytesEach));
            } catch (...) {
                tacuda::detail::set_last_error(
                    "batch: failed to allocate device matrix");
                return CT_STATUS_ALLOC_FAILED;
            }
            // Track immediately so a partial acquire cannot leak the
            // matrices already taken from the pool.
            count_ = i + 1;
        }
        return CT_STATUS_SUCCESS;
    }
    float* at(int i) const { return ptrs_[i]; }
    ~DeviceMatrices() {
        for (int i = 0; i < count_; ++i) {
            if (ptrs_[i]) {
                DeviceBufferPool::instance().release(ptrs_[i]);
            }
        }
    }

private:
    float* ptrs_[kMaxIo * 2] = {};
    int count_ = 0;
};

struct BatchStreams {
    std::vector<cudaStream_t> streams;  // dispatch targets (>= 1)
    cudaStream_t base = nullptr;        // transfer / ordering stream
};

BatchStreams resolve_streams(int numSeries, int maxConcurrency,
                             cudaStream_t userStream) {
    BatchStreams r;
    if (userStream != nullptr) {
        r.streams = {userStream};
        r.base = userStream;
        return r;
    }
    int want = maxConcurrency > 0 ? maxConcurrency
                                  : std::min(numSeries, kAutoStreamCap);
    want = std::max(1, std::min(want, numSeries));
    r.streams = BatchStreamPool::instance().get(want);
    if (r.streams.empty()) {
        r.streams = {static_cast<cudaStream_t>(nullptr)};  // default stream
    }
    r.base = r.streams.front();
    return r;
}

/// Make every dispatch stream wait for work already enqueued on `base`.
ctStatus_t fan_out(const BatchStreams& bs) {
    if (bs.streams.size() <= 1) return CT_STATUS_SUCCESS;
    cudaEvent_t ev = nullptr;
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
        return CT_STATUS_DEVICE_ERROR;
    }
    ctStatus_t st = CT_STATUS_SUCCESS;
    if (cudaEventRecord(ev, bs.base) != cudaSuccess) st = CT_STATUS_DEVICE_ERROR;
    for (size_t i = 1; st == CT_STATUS_SUCCESS && i < bs.streams.size(); ++i) {
        if (cudaStreamWaitEvent(bs.streams[i], ev, 0) != cudaSuccess) {
            st = CT_STATUS_DEVICE_ERROR;
        }
    }
    cudaEventDestroy(ev);
    return st;
}

/// Make `base` wait for work enqueued on every other dispatch stream.
ctStatus_t fan_in(const BatchStreams& bs) {
    ctStatus_t st = CT_STATUS_SUCCESS;
    for (size_t i = 1; i < bs.streams.size(); ++i) {
        cudaEvent_t ev = nullptr;
        if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
            return CT_STATUS_DEVICE_ERROR;
        }
        if (cudaEventRecord(ev, bs.streams[i]) != cudaSuccess ||
            cudaStreamWaitEvent(bs.base, ev, 0) != cudaSuccess) {
            st = CT_STATUS_DEVICE_ERROR;
        }
        cudaEventDestroy(ev);
    }
    return st;
}

/// stream == 0 is the legacy default stream — the default of PyTorch
/// (`torch.cuda.current_stream().cuda_stream == 0`), CuPy and plain CUDA —
/// so callers expect the batch to run after the work they queued there. The
/// pool streams are non-blocking and never order against it implicitly:
/// without this join a kernel still writing the inputs (or reading the
/// outputs) on that stream races the batch. Kernels that NaN-fill their
/// output before computing it then leave a whole row of NaN.
ctStatus_t join_legacy_stream(const BatchStreams& bs) {
    cudaEvent_t ev = nullptr;
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
        return CT_STATUS_DEVICE_ERROR;
    }
    ctStatus_t st = CT_STATUS_SUCCESS;
    if (cudaEventRecord(ev, cudaStreamLegacy) != cudaSuccess) st = CT_STATUS_DEVICE_ERROR;
    for (size_t i = 0; st == CT_STATUS_SUCCESS && i < bs.streams.size(); ++i) {
        if (cudaStreamWaitEvent(bs.streams[i], ev, 0) != cudaSuccess) {
            st = CT_STATUS_DEVICE_ERROR;
        }
    }
    cudaEventDestroy(ev);
    return st;
}

/// Drains every dispatch stream and reports the first synchronization
/// failure: an async kernel fault must never turn into CT_STATUS_SUCCESS.
ctStatus_t sync_all(const BatchStreams& bs) {
    ctStatus_t st = CT_STATUS_SUCCESS;
    for (cudaStream_t s : bs.streams) {
        cudaError_t err = cudaStreamSynchronize(s);
        if (err != cudaSuccess && st == CT_STATUS_SUCCESS) {
            tacuda::detail::set_last_error(cudaGetErrorString(err));
            st = CT_STATUS_DEVICE_ERROR;
        }
    }
    return st;
}

/// Runs a row-batched plan on `stream`. Rows plans: one call of the
/// whole-matrix launcher. Flat plans: the single-series entry point over
/// consecutive blocks of whole rows, each block followed by the warm-up
/// restore of its rows from the block's first row.
template <typename RowsFn>
ctStatus_t launch_rows(const RowsPlan& plan, RowsFn& whole,
                       const float* const* in, int nIn, float* const* out,
                       int nOut, int seriesLen, int numSeries,
                       cudaStream_t stream) {
    try {
        if (plan.kind != RowsPlan::Kind::Flat) {
            return whole(in, out, 0, stream);
        }
        const int rowsPerCall = static_cast<int>(std::min<long long>(
            numSeries, std::max<long long>(1, RowsPlan::kFlatChunk / seriesLen)));
        ctStatus_t st = CT_STATUS_SUCCESS;
        for (int r0 = 0; st == CT_STATUS_SUCCESS && r0 < numSeries; r0 += rowsPerCall) {
            const int rows = std::min(rowsPerCall, numSeries - r0);
            const size_t off = static_cast<size_t>(r0) * seriesLen;
            const float* blockIn[kMaxIo];
            float* blockOut[kMaxIo];
            for (int j = 0; j < nIn; ++j) blockIn[j] = in[j] + off;
            for (int j = 0; j < nOut; ++j) blockOut[j] = out[j] + off;
            st = whole(blockIn, blockOut, rows * seriesLen, stream);
            if (st == CT_STATUS_SUCCESS && plan.warmup > 0) {
                tacuda::detail::copy_row_prefix(blockOut, nOut, seriesLen, rows,
                                                static_cast<int>(plan.warmup), stream);
            }
        }
        return st;
    } catch (...) {
        return classify_batch_exception("batch launch failed");
    }
}

/// The stream-0 completion contract of the batch calls that run directly on
/// the legacy default stream (fused and row-batched); on a caller's stream
/// they stay asynchronous.
ctStatus_t finish_direct(ctStatus_t st, cudaStream_t userStream) {
    if (userStream == nullptr) {
        const cudaError_t err = cudaStreamSynchronize(nullptr);
        if (err != cudaSuccess && st == CT_STATUS_SUCCESS) {
            tacuda::detail::set_last_error(cudaGetErrorString(err));
            st = CT_STATUS_DEVICE_ERROR;
        }
    }
    return st;
}

/**
 * Generic host-matrix batch runner.
 * `launchSeries(dIn, dOut, stream)` receives per-series device pointers and
 * must enqueue stream-ordered work only; `wholeRows(dIn, dOut, n, stream)`
 * is the whole-matrix launcher of a row-batched plan.
 */
template <typename LaunchFn, typename RowsFn = NoRows>
ctStatus_t run_host_batch(const float* const* hostIns, int nIn,
                          float* const* hostOuts, int nOut, int seriesLen,
                          int numSeries, int maxConcurrency,
                          cudaStream_t userStream, LaunchFn&& launchSeries,
                          FusedCall fused = {}, RowsPlan rows = {},
                          RowsFn&& wholeRows = RowsFn{}) {
    tacuda::detail::clear_last_error();
    if (seriesLen <= 0 || numSeries <= 0 || nIn > kMaxIo || nOut > kMaxIo) {
        return CT_STATUS_INVALID_ARGUMENT;
    }
    for (int j = 0; j < nIn; ++j) {
        if (!hostIns[j]) return CT_STATUS_INVALID_ARGUMENT;
    }
    for (int j = 0; j < nOut; ++j) {
        if (!hostOuts[j]) return CT_STATUS_INVALID_ARGUMENT;
    }

    const size_t total = static_cast<size_t>(seriesLen) * numSeries;
    const size_t bytes = total * sizeof(float);

    HostPin pins[kMaxIo * 2];
    for (int j = 0; j < nIn; ++j) pins[j].pin(hostIns[j], bytes);
    for (int j = 0; j < nOut; ++j) pins[kMaxIo + j].pin(hostOuts[j], bytes);

    DeviceMatrices dIn, dOut;
    ctStatus_t st = dIn.acquire(nIn, bytes);
    if (st == CT_STATUS_SUCCESS) st = dOut.acquire(nOut, bytes);
    if (st != CT_STATUS_SUCCESS) return st;

    const bool useFused = fused.enabled(seriesLen, numSeries);
    const bool useRows = !useFused && rows.enabled(seriesLen, numSeries);
    const bool direct = useFused || useRows;
    BatchStreams bs = direct ? BatchStreams{{userStream}, userStream}
                             : resolve_streams(numSeries, maxConcurrency, userStream);

    for (int j = 0; st == CT_STATUS_SUCCESS && j < nIn; ++j) {
        if (cudaMemcpyAsync(dIn.at(j), hostIns[j], bytes,
                            cudaMemcpyHostToDevice, bs.base) != cudaSuccess) {
            tacuda::detail::set_last_error(cudaGetErrorString(cudaGetLastError()));
            st = CT_STATUS_COPY_FAILED;
        }
    }
    if (st == CT_STATUS_SUCCESS) st = fan_out(bs);

    if (direct && st == CT_STATUS_SUCCESS) {
        const float* in[kMaxIo];
        float* out[kMaxIo];
        for (int j = 0; j < nIn; ++j) in[j] = dIn.at(j);
        for (int j = 0; j < nOut; ++j) out[j] = dOut.at(j);
        st = useFused ? fused.launch(in, out, seriesLen, numSeries, bs.base)
                      : launch_rows(rows, wholeRows, in, nIn, out, nOut, seriesLen,
                                    numSeries, bs.base);
    }
    for (int i = 0; !direct && st == CT_STATUS_SUCCESS && i < numSeries; ++i) {
        const float* in[kMaxIo];
        float* out[kMaxIo];
        const size_t off = static_cast<size_t>(i) * seriesLen;
        for (int j = 0; j < nIn; ++j) in[j] = dIn.at(j) + off;
        for (int j = 0; j < nOut; ++j) out[j] = dOut.at(j) + off;
        try {
            st = launchSeries(in, out, bs.streams[i % bs.streams.size()]);
        } catch (...) {
            st = classify_batch_exception("batch series launch failed");
        }
    }

    if (st == CT_STATUS_SUCCESS) st = fan_in(bs);
    for (int j = 0; st == CT_STATUS_SUCCESS && j < nOut; ++j) {
        if (cudaMemcpyAsync(hostOuts[j], dOut.at(j), bytes,
                            cudaMemcpyDeviceToHost, bs.base) != cudaSuccess) {
            tacuda::detail::set_last_error(cudaGetErrorString(cudaGetLastError()));
            st = CT_STATUS_COPY_FAILED;
        }
    }

    // Always drain the pool streams so buffer release below is safe even on
    // the failure path; a failed synchronize is a failed batch call.
    ctStatus_t syncSt = sync_all(bs);
    if (st == CT_STATUS_SUCCESS) st = syncSt;
    return st;
}

/// Generic device-matrix batch runner (no transfers).
template <typename LaunchFn, typename RowsFn = NoRows>
ctStatus_t run_device_batch(const float* const* dIns, int nIn,
                            float* const* dOuts, int nOut, int seriesLen,
                            int numSeries, int maxConcurrency,
                            cudaStream_t userStream, LaunchFn&& launchSeries,
                            FusedCall fused = {}, RowsPlan rows = {},
                            RowsFn&& wholeRows = RowsFn{}) {
    tacuda::detail::clear_last_error();
    if (seriesLen <= 0 || numSeries <= 0 || nIn > kMaxIo || nOut > kMaxIo) {
        return CT_STATUS_INVALID_ARGUMENT;
    }
    for (int j = 0; j < nIn; ++j) {
        if (!dIns[j]) return CT_STATUS_INVALID_ARGUMENT;
    }
    for (int j = 0; j < nOut; ++j) {
        if (!dOuts[j]) return CT_STATUS_INVALID_ARGUMENT;
    }

    if (fused.enabled(seriesLen, numSeries)) {
        // On a caller's stream this is just one asynchronous kernel,
        // including capture; stream 0 keeps its completion contract.
        return finish_direct(fused.launch(dIns, dOuts, seriesLen, numSeries, userStream),
                             userStream);
    }
    if (rows.enabled(seriesLen, numSeries)) {
        return finish_direct(launch_rows(rows, wholeRows, dIns, nIn, dOuts, nOut,
                                         seriesLen, numSeries, userStream),
                             userStream);
    }
    BatchStreams bs = resolve_streams(numSeries, maxConcurrency, userStream);
    ctStatus_t st = CT_STATUS_SUCCESS;
    if (userStream == nullptr) st = join_legacy_stream(bs);

    for (int i = 0; st == CT_STATUS_SUCCESS && i < numSeries; ++i) {
        const float* in[kMaxIo];
        float* out[kMaxIo];
        const size_t off = static_cast<size_t>(i) * seriesLen;
        for (int j = 0; j < nIn; ++j) in[j] = dIns[j] + off;
        for (int j = 0; j < nOut; ++j) out[j] = dOuts[j] + off;
        try {
            st = launchSeries(in, out, bs.streams[i % bs.streams.size()]);
        } catch (...) {
            st = classify_batch_exception("batch series launch failed");
        }
    }

    // With a user stream everything is ordered on it: stay non-blocking like
    // the other _d entry points. Multi-stream dispatch must block instead,
    // because no single stream observes completion of the whole batch — and
    // a failed synchronize is a failed batch call.
    if (userStream == nullptr) {
        ctStatus_t syncSt = sync_all(bs);
        if (st == CT_STATUS_SUCCESS) st = syncSt;
    }
    return st;
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// Host-matrix batch API
// ---------------------------------------------------------------------------

ctStatus_t ct_sma_batch(const float* host_inputs, float* host_outputs,
                        int seriesLen, int numSeries, int period,
                        int maxConcurrency, cudaStream_t stream) {
    const float* ins[1] = {host_inputs};
    float* outs[1] = {host_outputs};
    return run_host_batch(
        ins, 1, outs, 1, seriesLen, numSeries, maxConcurrency, stream,
        [&](const float* const* in, float* const* out, cudaStream_t s) {
            return ct_sma_d(in[0], out[0], seriesLen, period, s);
        }, fused_call(FusedFamily::Sma, period));
}

ctStatus_t ct_ema_batch(const float* host_inputs, float* host_outputs,
                        int seriesLen, int numSeries, int period,
                        int maxConcurrency, cudaStream_t stream) {
    const float* ins[1] = {host_inputs};
    float* outs[1] = {host_outputs};
    return run_host_batch(
        ins, 1, outs, 1, seriesLen, numSeries, maxConcurrency, stream,
        [&](const float* const* in, float* const* out, cudaStream_t s) {
            return ct_ema_d(in[0], out[0], seriesLen, period, s);
        }, fused_call(FusedFamily::Ema, period));
}

ctStatus_t ct_rsi_batch(const float* host_inputs, float* host_outputs,
                        int seriesLen, int numSeries, int period,
                        int maxConcurrency, cudaStream_t stream) {
    const float* ins[1] = {host_inputs};
    float* outs[1] = {host_outputs};
    return run_host_batch(
        ins, 1, outs, 1, seriesLen, numSeries, maxConcurrency, stream,
        [&](const float* const* in, float* const* out, cudaStream_t s) {
            return ct_rsi_d(in[0], out[0], seriesLen, period, s);
        }, fused_call(FusedFamily::Rsi, period));
}

ctStatus_t ct_atr_batch(const float* host_highs, const float* host_lows,
                        const float* host_closes, float* host_outputs,
                        int seriesLen, int numSeries, int period, float initial,
                        int maxConcurrency, cudaStream_t stream) {
    const float* ins[3] = {host_highs, host_lows, host_closes};
    float* outs[1] = {host_outputs};
    return run_host_batch(
        ins, 3, outs, 1, seriesLen, numSeries, maxConcurrency, stream,
        [&](const float* const* in, float* const* out, cudaStream_t s) {
            return ct_atr_d(in[0], in[1], in[2], out[0], seriesLen, period,
                            initial, s);
        }, fused_call(FusedFamily::Atr, period));
}

ctStatus_t ct_macd_batch(const float* host_inputs, float* host_macds,
                         float* host_signals, float* host_hists, int seriesLen,
                         int numSeries, int fastPeriod, int slowPeriod,
                         int signalPeriod, int maType, int maxConcurrency,
                         cudaStream_t stream) {
    if (maType == CT_MA_DEFAULT) maType = CT_MA_EMA;
    const float* ins[1] = {host_inputs};
    float* outs[3] = {host_macds, host_signals, host_hists};
    return run_host_batch(
        ins, 1, outs, 3, seriesLen, numSeries, maxConcurrency, stream,
        [&](const float* const* in, float* const* out, cudaStream_t s) {
            return ct_macd_d(in[0], out[0], out[1], out[2], seriesLen,
                             fastPeriod, slowPeriod, signalPeriod, maType, s);
        }, fused_call(FusedFamily::Macd, fastPeriod, slowPeriod, signalPeriod, maType));
}

ctStatus_t ct_bbands_batch(const float* host_inputs, float* host_uppers,
                           float* host_middles, float* host_lowers,
                           int seriesLen, int numSeries, int period,
                           float upperMul, float lowerMul, int maxConcurrency,
                           cudaStream_t stream) {
    const float* ins[1] = {host_inputs};
    float* outs[3] = {host_uppers, host_middles, host_lowers};
    return run_host_batch(
        ins, 1, outs, 3, seriesLen, numSeries, maxConcurrency, stream,
        [&](const float* const* in, float* const* out, cudaStream_t s) {
            return ct_bbands_d(in[0], out[0], out[1], out[2], seriesLen,
                               period, upperMul, lowerMul, s);
        }, fused_call(FusedFamily::Bbands, period, 0, 0, 1, upperMul, lowerMul));
}

ctStatus_t ct_stochastic_batch(const float* host_highs, const float* host_lows,
                               const float* host_closes, float* host_ks,
                               float* host_ds, int seriesLen, int numSeries,
                               int kPeriod, int dPeriod, int maxConcurrency,
                               cudaStream_t stream) {
    const float* ins[3] = {host_highs, host_lows, host_closes};
    float* outs[2] = {host_ks, host_ds};
    return run_host_batch(
        ins, 3, outs, 2, seriesLen, numSeries, maxConcurrency, stream,
        [&](const float* const* in, float* const* out, cudaStream_t s) {
            return ct_stochastic_d(in[0], in[1], in[2], out[0], out[1],
                                   seriesLen, kPeriod, dPeriod, s);
        }, fused_call(FusedFamily::Stochastic, kPeriod, dPeriod));
}

// ---------------------------------------------------------------------------
// Device-matrix batch API
// ---------------------------------------------------------------------------

ctStatus_t ct_sma_batch_d(const float* d_inputs, float* d_outputs,
                          int seriesLen, int numSeries, int period,
                          int maxConcurrency, cudaStream_t stream) {
    const float* ins[1] = {d_inputs};
    float* outs[1] = {d_outputs};
    return run_device_batch(
        ins, 1, outs, 1, seriesLen, numSeries, maxConcurrency, stream,
        [&](const float* const* in, float* const* out, cudaStream_t s) {
            return ct_sma_d(in[0], out[0], seriesLen, period, s);
        }, fused_call(FusedFamily::Sma, period));
}

ctStatus_t ct_ema_batch_d(const float* d_inputs, float* d_outputs,
                          int seriesLen, int numSeries, int period,
                          int maxConcurrency, cudaStream_t stream) {
    const float* ins[1] = {d_inputs};
    float* outs[1] = {d_outputs};
    return run_device_batch(
        ins, 1, outs, 1, seriesLen, numSeries, maxConcurrency, stream,
        [&](const float* const* in, float* const* out, cudaStream_t s) {
            return ct_ema_d(in[0], out[0], seriesLen, period, s);
        }, fused_call(FusedFamily::Ema, period));
}

ctStatus_t ct_rsi_batch_d(const float* d_inputs, float* d_outputs,
                          int seriesLen, int numSeries, int period,
                          int maxConcurrency, cudaStream_t stream) {
    const float* ins[1] = {d_inputs};
    float* outs[1] = {d_outputs};
    return run_device_batch(
        ins, 1, outs, 1, seriesLen, numSeries, maxConcurrency, stream,
        [&](const float* const* in, float* const* out, cudaStream_t s) {
            return ct_rsi_d(in[0], out[0], seriesLen, period, s);
        }, fused_call(FusedFamily::Rsi, period));
}

ctStatus_t ct_atr_batch_d(const float* d_highs, const float* d_lows,
                          const float* d_closes, float* d_outputs,
                          int seriesLen, int numSeries, int period,
                          float initial, int maxConcurrency,
                          cudaStream_t stream) {
    const float* ins[3] = {d_highs, d_lows, d_closes};
    float* outs[1] = {d_outputs};
    return run_device_batch(
        ins, 3, outs, 1, seriesLen, numSeries, maxConcurrency, stream,
        [&](const float* const* in, float* const* out, cudaStream_t s) {
            return ct_atr_d(in[0], in[1], in[2], out[0], seriesLen, period,
                            initial, s);
        }, fused_call(FusedFamily::Atr, period));
}

ctStatus_t ct_macd_batch_d(const float* d_inputs, float* d_macds,
                           float* d_signals, float* d_hists, int seriesLen,
                           int numSeries, int fastPeriod, int slowPeriod,
                           int signalPeriod, int maType, int maxConcurrency,
                           cudaStream_t stream) {
    if (maType == CT_MA_DEFAULT) maType = CT_MA_EMA;
    const float* ins[1] = {d_inputs};
    float* outs[3] = {d_macds, d_signals, d_hists};
    return run_device_batch(
        ins, 1, outs, 3, seriesLen, numSeries, maxConcurrency, stream,
        [&](const float* const* in, float* const* out, cudaStream_t s) {
            return ct_macd_d(in[0], out[0], out[1], out[2], seriesLen,
                             fastPeriod, slowPeriod, signalPeriod, maType, s);
        }, fused_call(FusedFamily::Macd, fastPeriod, slowPeriod, signalPeriod, maType));
}

ctStatus_t ct_bbands_batch_d(const float* d_inputs, float* d_uppers,
                             float* d_middles, float* d_lowers, int seriesLen,
                             int numSeries, int period, float upperMul,
                             float lowerMul, int maxConcurrency,
                             cudaStream_t stream) {
    const float* ins[1] = {d_inputs};
    float* outs[3] = {d_uppers, d_middles, d_lowers};
    return run_device_batch(
        ins, 1, outs, 3, seriesLen, numSeries, maxConcurrency, stream,
        [&](const float* const* in, float* const* out, cudaStream_t s) {
            return ct_bbands_d(in[0], out[0], out[1], out[2], seriesLen,
                               period, upperMul, lowerMul, s);
        }, fused_call(FusedFamily::Bbands, period, 0, 0, 1, upperMul, lowerMul));
}

ctStatus_t ct_stochastic_batch_d(const float* d_highs, const float* d_lows,
                                 const float* d_closes, float* d_ks,
                                 float* d_ds, int seriesLen, int numSeries,
                                 int kPeriod, int dPeriod, int maxConcurrency,
                                 cudaStream_t stream) {
    const float* ins[3] = {d_highs, d_lows, d_closes};
    float* outs[2] = {d_ks, d_ds};
    return run_device_batch(
        ins, 3, outs, 2, seriesLen, numSeries, maxConcurrency, stream,
        [&](const float* const* in, float* const* out, cudaStream_t s) {
            return ct_stochastic_d(in[0], in[1], in[2], out[0], out[1],
                                   seriesLen, kPeriod, dPeriod, s);
        }, fused_call(FusedFamily::Stochastic, kPeriod, dPeriod));
}

// Generated batch families (scripts/generate_batch_api.py).
#include "api_batch_generated.inc"

}  // extern "C"
