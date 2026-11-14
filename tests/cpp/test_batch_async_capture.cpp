/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
// *_batch_d on a caller's stream must (1) never block the host and (2) be
// capturable into a CUDA graph whose replay reproduces the eager result.
//
// The family list covers every internal path that used to synchronize or
// allocate outside stream order: Thrust scans (were par: cudaMalloc +
// cudaStreamSynchronize + cudaFree per scan), host-read seeds
// (transform_reduce in ADX/DX/±DI/MACDEXT, the KAMA y0 copy), raw
// cudaMalloc/cudaFree in wrappers (MA/MAMA, STOCHF, ACCBANDS, AROON,
// *_ex_d), pooled scratch (DeviceBufferPool queried events on the capturing
// stream) and the kernels that now write their own warm-up NaNs.

#include <tacuda.h>
#include <tacuda_device.h>

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace {

constexpr int kLen = 300;     // > 256: KAMA takes its multi-block scan path
constexpr int kSeries = 6;
constexpr int kOutputs = 3;   // widest family output count used below

struct DeviceBuffer {
    float* ptr = nullptr;
    explicit DeviceBuffer(size_t count) {
        EXPECT_EQ(cudaMalloc(&ptr, count * sizeof(float)), cudaSuccess);
    }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

/// OHLCV portfolio of kSeries x kLen bars plus kOutputs output matrices.
struct Portfolio {
    static constexpr size_t kCount = static_cast<size_t>(kLen) * kSeries;
    DeviceBuffer open{kCount}, high{kCount}, low{kCount}, close{kCount}, volume{kCount};
    DeviceBuffer out[kOutputs] = {DeviceBuffer{kCount}, DeviceBuffer{kCount},
                                  DeviceBuffer{kCount}};

    Portfolio() {
        std::vector<float> o(kCount), h(kCount), l(kCount), c(kCount), v(kCount);
        for (int s = 0; s < kSeries; ++s) {
            float price = 50.0f + 10.0f * s;
            for (int t = 0; t < kLen; ++t) {
                const size_t i = static_cast<size_t>(s) * kLen + t;
                const float step = 0.8f * std::sin(0.31f * t + s) + 0.3f * std::cos(1.7f * t * (s + 1));
                o[i] = price;
                price += step;
                c[i] = price;
                h[i] = std::max(o[i], c[i]) + 0.4f + 0.2f * std::sin(0.9f * t);
                l[i] = std::min(o[i], c[i]) - 0.4f - 0.2f * std::cos(1.1f * t);
                v[i] = 1000.0f + 500.0f * (1.0f + std::sin(0.13f * t + 2.0f * s));
            }
        }
        upload(open, o); upload(high, h); upload(low, l); upload(close, c); upload(volume, v);
    }

    static void upload(DeviceBuffer& d, const std::vector<float>& h) {
        ASSERT_EQ(cudaMemcpy(d.ptr, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice),
                  cudaSuccess);
    }

    /// Fills every output with a sentinel so untouched elements are visible.
    void poison_outputs(cudaStream_t s) {
        for (auto& o : out) {
            ASSERT_EQ(cudaMemsetAsync(o.ptr, 0x5A, kCount * sizeof(float), s), cudaSuccess);
        }
    }

    std::vector<uint32_t> download_outputs() {
        std::vector<uint32_t> bits(kCount * kOutputs);
        for (int k = 0; k < kOutputs; ++k) {
            EXPECT_EQ(cudaMemcpy(bits.data() + k * kCount, out[k].ptr, kCount * sizeof(float),
                                 cudaMemcpyDeviceToHost),
                      cudaSuccess);
        }
        return bits;
    }
};

struct Family {
    const char* name;
    std::function<ctStatus_t(Portfolio&, cudaStream_t)> call;
};

std::vector<Family> families() {
    const int T = kLen, S = kSeries;
    return {
        {"mfi", [=](Portfolio& p, cudaStream_t s) { return ct_mfi_batch_d(p.high.ptr, p.low.ptr, p.close.ptr, p.volume.ptr, p.out[0].ptr, T, S, 14, 0, s); }},
        {"rsi", [=](Portfolio& p, cudaStream_t s) { return ct_rsi_batch_d(p.close.ptr, p.out[0].ptr, T, S, 14, 0, s); }},
        {"atr", [=](Portfolio& p, cudaStream_t s) { return ct_atr_batch_d(p.high.ptr, p.low.ptr, p.close.ptr, p.out[0].ptr, T, S, 14, 0.0f, 0, s); }},
        {"natr", [=](Portfolio& p, cudaStream_t s) { return ct_natr_batch_d(p.high.ptr, p.low.ptr, p.close.ptr, p.out[0].ptr, T, S, 14, 0, s); }},
        {"adosc", [=](Portfolio& p, cudaStream_t s) { return ct_adosc_batch_d(p.high.ptr, p.low.ptr, p.close.ptr, p.volume.ptr, p.out[0].ptr, T, S, 3, 10, 0, s); }},
        {"ad", [=](Portfolio& p, cudaStream_t s) { return ct_ad_batch_d(p.high.ptr, p.low.ptr, p.close.ptr, p.volume.ptr, p.out[0].ptr, T, S, 0, s); }},
        {"obv", [=](Portfolio& p, cudaStream_t s) { return ct_obv_batch_d(p.close.ptr, p.volume.ptr, p.out[0].ptr, T, S, 0, s); }},
        {"stddev", [=](Portfolio& p, cudaStream_t s) { return ct_stddev_batch_d(p.close.ptr, p.out[0].ptr, T, S, 20, 0, s); }},
        {"sum", [=](Portfolio& p, cudaStream_t s) { return ct_sum_batch_d(p.close.ptr, p.out[0].ptr, T, S, 20, 0, s); }},
        {"min", [=](Portfolio& p, cudaStream_t s) { return ct_min_batch_d(p.low.ptr, p.out[0].ptr, T, S, 10, 0, s); }},
        {"max", [=](Portfolio& p, cudaStream_t s) { return ct_max_batch_d(p.high.ptr, p.out[0].ptr, T, S, 10, 0, s); }},
        {"sma", [=](Portfolio& p, cudaStream_t s) { return ct_sma_batch_d(p.close.ptr, p.out[0].ptr, T, S, 20, 0, s); }},
        {"ema", [=](Portfolio& p, cudaStream_t s) { return ct_ema_batch_d(p.close.ptr, p.out[0].ptr, T, S, 20, 0, s); }},
        {"wma", [=](Portfolio& p, cudaStream_t s) { return ct_wma_batch_d(p.close.ptr, p.out[0].ptr, T, S, 20, 0, s); }},
        {"bbands", [=](Portfolio& p, cudaStream_t s) { return ct_bbands_batch_d(p.close.ptr, p.out[0].ptr, p.out[1].ptr, p.out[2].ptr, T, S, 20, 2.0f, 2.0f, 0, s); }},
        {"stoch", [=](Portfolio& p, cudaStream_t s) { return ct_stochastic_batch_d(p.high.ptr, p.low.ptr, p.close.ptr, p.out[0].ptr, p.out[1].ptr, T, S, 14, 3, 0, s); }},
        {"stochf", [=](Portfolio& p, cudaStream_t s) { return ct_stochf_batch_d(p.high.ptr, p.low.ptr, p.close.ptr, p.out[0].ptr, p.out[1].ptr, T, S, 14, 3, 0, 0, s); }},
        {"adx", [=](Portfolio& p, cudaStream_t s) { return ct_adx_batch_d(p.high.ptr, p.low.ptr, p.close.ptr, p.out[0].ptr, T, S, 14, 0, s); }},
        {"dx", [=](Portfolio& p, cudaStream_t s) { return ct_dx_batch_d(p.high.ptr, p.low.ptr, p.close.ptr, p.out[0].ptr, T, S, 14, 0, s); }},
        {"plus_di", [=](Portfolio& p, cudaStream_t s) { return ct_plus_di_batch_d(p.high.ptr, p.low.ptr, p.close.ptr, p.out[0].ptr, T, S, 14, 0, s); }},
        {"minus_di", [=](Portfolio& p, cudaStream_t s) { return ct_minus_di_batch_d(p.high.ptr, p.low.ptr, p.close.ptr, p.out[0].ptr, T, S, 14, 0, s); }},
        {"macdext_ema", [=](Portfolio& p, cudaStream_t s) { return ct_macdext_batch_d(p.close.ptr, p.out[0].ptr, p.out[1].ptr, p.out[2].ptr, T, S, 12, 1, 26, 1, 9, 1, 0, s); }},
        {"macdext_sma", [=](Portfolio& p, cudaStream_t s) { return ct_macdext_batch_d(p.close.ptr, p.out[0].ptr, p.out[1].ptr, p.out[2].ptr, T, S, 12, 0, 26, 0, 9, 0, 0, s); }},
        {"kama_long_tail", [=](Portfolio& p, cudaStream_t s) { return ct_kama_batch_d(p.close.ptr, p.out[0].ptr, T, S, 10, 2, 30, 0, s); }},
        {"kama_short_tail", [=](Portfolio& p, cudaStream_t s) { return ct_kama_batch_d(p.close.ptr, p.out[0].ptr, T, S, 100, 2, 30, 0, s); }},
        {"ma_mama", [=](Portfolio& p, cudaStream_t s) { return ct_ma_batch_d(p.close.ptr, p.out[0].ptr, T, S, 30, 7, 0, s); }},
        {"accbands", [=](Portfolio& p, cudaStream_t s) { return ct_accbands_batch_d(p.high.ptr, p.low.ptr, p.close.ptr, p.out[0].ptr, p.out[1].ptr, p.out[2].ptr, T, S, 20, 0, s); }},
        {"aroon", [=](Portfolio& p, cudaStream_t s) { return ct_aroon_batch_d(p.high.ptr, p.low.ptr, p.out[0].ptr, p.out[1].ptr, T, S, 14, 0, s); }},
        {"cmo", [=](Portfolio& p, cudaStream_t s) { return ct_cmo_batch_d(p.close.ptr, p.out[0].ptr, T, S, 14, 0, s); }},
        {"correl", [=](Portfolio& p, cudaStream_t s) { return ct_correl_batch_d(p.high.ptr, p.low.ptr, p.out[0].ptr, T, S, 30, 0, s); }},
        {"beta", [=](Portfolio& p, cudaStream_t s) { return ct_beta_batch_d(p.high.ptr, p.low.ptr, p.out[0].ptr, T, S, 5, 0, s); }},
        {"nvi", [=](Portfolio& p, cudaStream_t s) { return ct_nvi_batch_d(p.close.ptr, p.volume.ptr, p.out[0].ptr, T, S, 0, s); }},
        {"pvi", [=](Portfolio& p, cudaStream_t s) { return ct_pvi_batch_d(p.close.ptr, p.volume.ptr, p.out[0].ptr, T, S, 0, s); }},
        {"imi", [=](Portfolio& p, cudaStream_t s) { return ct_imi_batch_d(p.open.ptr, p.close.ptr, p.out[0].ptr, T, S, 14, 0, s); }},
        {"obtr", [=](Portfolio& p, cudaStream_t s) { return ct_obtr_batch_d(p.high.ptr, p.low.ptr, p.close.ptr, p.out[0].ptr, T, S, 0, s); }},
        {"asi", [=](Portfolio& p, cudaStream_t s) { return ct_asi_batch_d(p.open.ptr, p.high.ptr, p.low.ptr, p.close.ptr, p.out[0].ptr, T, S, 0, s); }},
        // Mixed-precision device path (scratch used to be cudaMalloc/cudaFree).
        {"rsi_ex_f64", [=](Portfolio& p, cudaStream_t s) { return ct_rsi_ex_d(p.close.ptr, p.out[0].ptr, T * S, 14, CT_DTYPE_F32, CT_DTYPE_F64, s); }},
        {"sma_ex_f64", [=](Portfolio& p, cudaStream_t s) { return ct_sma_ex_d(p.close.ptr, p.out[0].ptr, T * S, 20, CT_DTYPE_F32, CT_DTYPE_F64, s); }},
        {"ema_ex_f64", [=](Portfolio& p, cudaStream_t s) { return ct_ema_ex_d(p.close.ptr, p.out[0].ptr, T * S, 20, CT_DTYPE_F32, CT_DTYPE_F64, s); }},
        {"atr_ex_f64", [=](Portfolio& p, cudaStream_t s) { return ct_atr_ex_d(p.high.ptr, p.low.ptr, p.close.ptr, p.out[0].ptr, T * S, 14, 0.0f, CT_DTYPE_F32, CT_DTYPE_F64, s); }},
        {"bbands_ex_f64", [=](Portfolio& p, cudaStream_t s) { return ct_bbands_ex_d(p.close.ptr, p.out[0].ptr, p.out[1].ptr, p.out[2].ptr, T * S, 20, 2.0f, 2.0f, CT_DTYPE_F32, CT_DTYPE_F64, s); }},
        {"macd_ex_f64", [=](Portfolio& p, cudaStream_t s) { return ct_macd_ex_d(p.close.ptr, p.out[0].ptr, p.out[1].ptr, p.out[2].ptr, T * S, 12, 26, 9, 1, CT_DTYPE_F32, CT_DTYPE_F64, s); }},
    };
}

std::string last_error() {
    const char* e = ct_get_last_error();
    return e ? e : "";
}

/// Host function that keeps everything queued behind it pending until the
/// test thread opens the gate (or a timeout expires).
struct Gate {
    std::mutex m;
    std::condition_variable cv;
    bool open = false;
    bool timedOut = false;

    static void CUDART_CB wait(void* self) {
        auto* g = static_cast<Gate*>(self);
        std::unique_lock<std::mutex> lock(g->m);
        if (!g->cv.wait_for(lock, std::chrono::seconds(5), [g] { return g->open; })) {
            g->timedOut = true;
        }
    }
    void release() {
        { std::lock_guard<std::mutex> lock(m); open = true; }
        cv.notify_all();
    }
};

class StreamGuard {
public:
    StreamGuard() { EXPECT_EQ(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking), cudaSuccess); }
    ~StreamGuard() { cudaStreamDestroy(s_); }
    cudaStream_t get() const { return s_; }
private:
    cudaStream_t s_ = nullptr;
};

}  // namespace

// A user-stream *_batch_d call only enqueues work. The stream is held by a
// host function until the call has returned: any synchronization inside the
// call would wait for the gate, which then times out.
TEST(BatchAsyncCapture, UserStreamCallsDoNotBlockTheHost) {
    Portfolio p;
    StreamGuard stream;
    for (const Family& f : families()) {
        SCOPED_TRACE(f.name);
        // Warm-up: first-use costs (module loading, pool growth) are not the subject.
        ASSERT_EQ(f.call(p, stream.get()), CT_STATUS_SUCCESS) << last_error();
        ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);

        Gate gate;
        ASSERT_EQ(cudaLaunchHostFunc(stream.get(), &Gate::wait, &gate), cudaSuccess);
        const ctStatus_t st = f.call(p, stream.get());
        const bool returnedWhileHeld = cudaStreamQuery(stream.get()) == cudaErrorNotReady;
        (void)cudaGetLastError();  // NotReady is a status, not a launch error
        gate.release();
        ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
        EXPECT_EQ(st, CT_STATUS_SUCCESS) << last_error();
        EXPECT_FALSE(gate.timedOut) << f.name << " synchronized the caller's stream";
        EXPECT_TRUE(returnedWhileHeld) << f.name << " returned only after its stream drained";
    }
}

// Strict (global-mode) capture: also rejects cudaMalloc/cudaFree and other
// unsafe calls, not just synchronization. The graph is replayed twice into
// poisoned outputs and must reproduce the eager result bit for bit.
TEST(BatchAsyncCapture, GraphCaptureReplaysEagerResult) {
    Portfolio p;
    StreamGuard stream;
    for (const Family& f : families()) {
        SCOPED_TRACE(f.name);
        p.poison_outputs(stream.get());
        ASSERT_EQ(f.call(p, stream.get()), CT_STATUS_SUCCESS) << last_error();
        ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
        const std::vector<uint32_t> eager = p.download_outputs();

        cudaGraph_t graph = nullptr;
        ASSERT_EQ(cudaStreamBeginCapture(stream.get(), cudaStreamCaptureModeGlobal), cudaSuccess);
        const ctStatus_t st = f.call(p, stream.get());
        const std::string err = last_error();
        const cudaError_t endErr = cudaStreamEndCapture(stream.get(), &graph);
        ASSERT_EQ(st, CT_STATUS_SUCCESS) << err;
        ASSERT_EQ(endErr, cudaSuccess) << cudaGetErrorString(endErr);
        ASSERT_NE(graph, nullptr);

        cudaGraphExec_t exec = nullptr;
        ASSERT_EQ(cudaGraphInstantiate(&exec, graph, 0), cudaSuccess);
        for (int replay = 0; replay < 2; ++replay) {
            p.poison_outputs(stream.get());
            ASSERT_EQ(cudaGraphLaunch(exec, stream.get()), cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
            const std::vector<uint32_t> replayed = p.download_outputs();
            size_t mismatches = 0;
            for (size_t i = 0; i < eager.size(); ++i) mismatches += eager[i] != replayed[i];
            EXPECT_EQ(mismatches, 0u) << "replay " << replay;
        }
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
    }
}

// The stream-0 contract is unchanged: the pool joins the legacy stream and
// the call returns with the batch complete, now with the pool streams
// actually overlapping (no per-series host synchronization).
TEST(BatchAsyncCapture, LegacyStreamPathMatchesUserStreamPath) {
    Portfolio p;
    StreamGuard stream;
    for (const Family& f : families()) {
        if (std::strstr(f.name, "_ex_") != nullptr) continue;  // not a batch call
        SCOPED_TRACE(f.name);
        p.poison_outputs(stream.get());
        ASSERT_EQ(f.call(p, stream.get()), CT_STATUS_SUCCESS) << last_error();
        ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
        const std::vector<uint32_t> onUserStream = p.download_outputs();

        p.poison_outputs(nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(f.call(p, nullptr), CT_STATUS_SUCCESS) << last_error();
        const std::vector<uint32_t> onLegacy = p.download_outputs();
        size_t mismatches = 0;
        for (size_t i = 0; i < onUserStream.size(); ++i) mismatches += onUserStream[i] != onLegacy[i];
        EXPECT_EQ(mismatches, 0u);
    }
}
