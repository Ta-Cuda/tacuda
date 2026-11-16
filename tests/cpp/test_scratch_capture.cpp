/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda.h>
#include <tacuda_device.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace {
constexpr int kSeries = 6, kLength = 300, kCount = kSeries * kLength;

class DispatchMode {
public:
    explicit DispatchMode(const char* mode) {
        if (const char* old = std::getenv("TACUDA_BATCH_FUSED")) { old_ = old; had_ = true; }
        set(mode);
    }
    ~DispatchMode() { set(had_ ? old_.c_str() : nullptr); }
private:
    static void set(const char* value) {
#ifdef _WIN32
        _putenv_s("TACUDA_BATCH_FUSED", value ? value : "");
#else
        if (value) setenv("TACUDA_BATCH_FUSED", value, 1);
        else unsetenv("TACUDA_BATCH_FUSED");
#endif
    }
    std::string old_;
    bool had_ = false;
};

// Each test owns its stream and recovers only its own expected capture errors.
// Nothing in the production API clears an arbitrary caller's runtime error.
struct Capture {
    cudaStream_t stream;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool active = false;
    explicit Capture(cudaStream_t s) : stream(s) {}
    cudaError_t begin() {
        const auto status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        active = status == cudaSuccess;
        return status;
    }
    cudaError_t end() { active = false; return cudaStreamEndCapture(stream, &graph); }
    ~Capture() {
        if (active) (void)cudaStreamEndCapture(stream, &graph);
        if (exec) (void)cudaGraphExecDestroy(exec);
        if (graph) (void)cudaGraphDestroy(graph);
        (void)cudaGetLastError();
    }
};

struct Inputs {
    float* p[5] = {};
    cudaStream_t stream = nullptr;
    Inputs() {
        EXPECT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        for (int k = 0; k < 5; ++k) {
            EXPECT_EQ(cudaMalloc(&p[k], kCount * sizeof(float)), cudaSuccess);
            if (k < 4) {
                std::vector<float> host(kCount);
                for (int j = 0; j < kCount; ++j)
                    host[j] = k == 3 ? 1000.0f : 100.0f + 0.01f * (j % kLength) + (k == 0 ? 1.0f : k == 1 ? -1.0f : 0.0f);
                EXPECT_EQ(cudaMemcpy(p[k], host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
            }
        }
        // Pageable H2D may return before DMA completes. Establish readiness
        // before the nonblocking stream uses these inputs.
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    ~Inputs() {
        (void)cudaStreamSynchronize(stream);
        for (auto ptr : p) (void)cudaFree(ptr);
        (void)cudaStreamDestroy(stream);
    }
    ctStatus_t mfi() { return ct_mfi_batch_d(p[0], p[1], p[2], p[3], p[4], kLength, kSeries, 14, 0, stream); }
    std::vector<unsigned char> output() {
        std::vector<unsigned char> result(kCount * sizeof(float));
        EXPECT_EQ(cudaMemcpy(result.data(), p[4], result.size(), cudaMemcpyDeviceToHost), cudaSuccess);
        return result;
    }
};

bool graph_memory_supported() {
    int driver = 0, runtime = 0, device = 0, pools = 0;
    if (cudaDriverGetVersion(&driver) != cudaSuccess || cudaRuntimeGetVersion(&runtime) != cudaSuccess ||
        driver < 11040 || runtime < 11040 || cudaGetDevice(&device) != cudaSuccess) return false;
    return cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, device) == cudaSuccess && pools != 0;
}
}

TEST(ScratchCapture, EagerFallbackPreservesOutput) {
    DispatchMode mode("0");
    Inputs in;
    ASSERT_EQ(in.mfi(), CT_STATUS_SUCCESS) << ct_get_last_error();
    ASSERT_EQ(cudaStreamSynchronize(in.stream), cudaSuccess);
    const auto baseline = in.output();
    ASSERT_EQ(in.mfi(), CT_STATUS_SUCCESS) << ct_get_last_error();
    ASSERT_EQ(cudaStreamSynchronize(in.stream), cudaSuccess);
    EXPECT_EQ(in.output(), baseline);
}

TEST(ScratchCapture, MissingPoolsReturnUnsupportedWithoutRuntimeError) {
    if (graph_memory_supported()) GTEST_SKIP() << "Requires a legacy device or the no-memory-pools shim";
    DispatchMode mode("0");
    Inputs in;
    const std::vector<std::function<ctStatus_t()>> calls = {
        [&] { return in.mfi(); },
        [&] { return ct_mfi_d(in.p[0], in.p[1], in.p[2], in.p[3], in.p[4], kCount, 14, in.stream); },
        [&] { return ct_rsi_ex_d(in.p[2], in.p[4], kCount, 14, CT_DTYPE_F32, CT_DTYPE_F64, in.stream); },
    };
    for (size_t index = 0; index < calls.size(); ++index) {
        SCOPED_TRACE(index);
        ASSERT_EQ(calls[index](), CT_STATUS_SUCCESS) << ct_get_last_error();
        ASSERT_EQ(cudaStreamSynchronize(in.stream), cudaSuccess);
        const auto baseline = in.output();
        {
            Capture capture(in.stream);
            ASSERT_EQ(capture.begin(), cudaSuccess);
            const auto status = calls[index]();
            const std::string error = ct_get_last_error() ? ct_get_last_error() : "";
            const auto end = capture.end();
            EXPECT_EQ(status, CT_STATUS_UNSUPPORTED) << error;
            EXPECT_NE(error.find("cudaDevAttrMemoryPoolsSupported"), std::string::npos) << error;
            EXPECT_EQ(end, cudaSuccess);
            EXPECT_EQ(cudaPeekAtLastError(), cudaSuccess);
        }
        ASSERT_EQ(calls[index](), CT_STATUS_SUCCESS) << ct_get_last_error();
        ASSERT_EQ(cudaStreamSynchronize(in.stream), cudaSuccess);
        EXPECT_EQ(in.output(), baseline);
    }
}

TEST(ScratchCapture, FusedGraphDoesNotNeedMemoryPools) {
    DispatchMode mode("1");
    Inputs in;
    ASSERT_EQ(in.mfi(), CT_STATUS_SUCCESS) << ct_get_last_error();
    ASSERT_EQ(cudaStreamSynchronize(in.stream), cudaSuccess);
    const auto baseline = in.output();
    Capture capture(in.stream);
    ASSERT_EQ(capture.begin(), cudaSuccess);
    ASSERT_EQ(in.mfi(), CT_STATUS_SUCCESS) << ct_get_last_error();
    ASSERT_EQ(capture.end(), cudaSuccess);
    size_t nodes = 0;
    ASSERT_EQ(cudaGraphGetNodes(capture.graph, nullptr, &nodes), cudaSuccess);
    EXPECT_EQ(nodes, 1u);
    ASSERT_EQ(cudaGraphInstantiate(&capture.exec, capture.graph, 0), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(capture.exec, in.stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(in.stream), cudaSuccess);
    EXPECT_EQ(in.output(), baseline);
}

TEST(ScratchCapture, FailedCaptureDoesNotContaminateTheNextTest) {
    DispatchMode mode("0");
    Inputs in;
    {
        Capture capture(in.stream);
        ASSERT_EQ(capture.begin(), cudaSuccess);
        EXPECT_EQ(cudaDeviceSynchronize(), cudaErrorStreamCaptureUnsupported);
        EXPECT_EQ(capture.end(), cudaErrorStreamCaptureInvalidated);
    }
    ASSERT_EQ(cudaPeekAtLastError(), cudaSuccess);
    ASSERT_EQ(in.mfi(), CT_STATUS_SUCCESS) << ct_get_last_error();
    ASSERT_EQ(cudaStreamSynchronize(in.stream), cudaSuccess);
}
