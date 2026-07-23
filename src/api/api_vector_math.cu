/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
/**
 * @file api_vector_math.cu
 * @brief Element-wise vector math (TA-Lib Math Transform / Math Operators).
 *
 * ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT,
 * TAN, TANH plus ADD, SUB, MULT, DIV. These operations have no lookback and
 * are embarrassingly parallel, so the GPU advantage grows with input size.
 * Grid-stride kernels keep occupancy high on multi-million element arrays.
 */

#include <cuda_runtime.h>
#include <cmath>
#include <exception>

#include "tacuda.h"
#include "tacuda_device.h"
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/ErrorState.h>

namespace {

struct OpAcos  { __device__ float operator()(float x) const { return acosf(x); } };
struct OpAsin  { __device__ float operator()(float x) const { return asinf(x); } };
struct OpAtan  { __device__ float operator()(float x) const { return atanf(x); } };
struct OpCeil  { __device__ float operator()(float x) const { return ceilf(x); } };
struct OpCos   { __device__ float operator()(float x) const { return cosf(x); } };
struct OpCosh  { __device__ float operator()(float x) const { return coshf(x); } };
struct OpExp   { __device__ float operator()(float x) const { return expf(x); } };
struct OpFloor { __device__ float operator()(float x) const { return floorf(x); } };
struct OpLn    { __device__ float operator()(float x) const { return logf(x); } };
struct OpLog10 { __device__ float operator()(float x) const { return log10f(x); } };
struct OpSin   { __device__ float operator()(float x) const { return sinf(x); } };
struct OpSinh  { __device__ float operator()(float x) const { return sinhf(x); } };
struct OpSqrt  { __device__ float operator()(float x) const { return sqrtf(x); } };
struct OpTan   { __device__ float operator()(float x) const { return tanf(x); } };
struct OpTanh  { __device__ float operator()(float x) const { return tanhf(x); } };

struct OpAdd  { __device__ float operator()(float a, float b) const { return a + b; } };
struct OpSub  { __device__ float operator()(float a, float b) const { return a - b; } };
struct OpMult { __device__ float operator()(float a, float b) const { return a * b; } };
struct OpDiv  { __device__ float operator()(float a, float b) const { return a / b; } };

template <typename Op>
__global__ void unaryKernel(const float* __restrict__ input,
                            float* __restrict__ output, int size, Op op) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += stride) {
        output[i] = op(input[i]);
    }
}

template <typename Op>
__global__ void binaryKernel(const float* __restrict__ input1,
                             const float* __restrict__ input2,
                             float* __restrict__ output, int size, Op op) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += stride) {
        output[i] = op(input1[i], input2[i]);
    }
}

inline dim3 strideGrid(int n) {
    int blocks = (n + 255) / 256;
    if (blocks > 4096) blocks = 4096; // grid-stride loop covers the rest
    return dim3(blocks, 1, 1);
}

template <typename Op>
ctStatus_t launch_unary_device(const float* d_input, float* d_output, int size,
                               cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) {
        tacuda::detail::set_last_error("vector math: invalid arguments");
        return CT_STATUS_INVALID_ARGUMENT;
    }
    unaryKernel<Op><<<strideGrid(size), dim3(256, 1, 1), 0, stream>>>(
        d_input, d_output, size, Op{});
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        tacuda::detail::set_last_error(cudaGetErrorString(err));
        return CT_STATUS_KERNEL_FAILED;
    }
    return CT_STATUS_SUCCESS;
}

template <typename Op>
ctStatus_t launch_binary_device(const float* d_input1, const float* d_input2,
                                float* d_output, int size, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input1 || !d_input2 || !d_output || size <= 0) {
        tacuda::detail::set_last_error("vector math: invalid arguments");
        return CT_STATUS_INVALID_ARGUMENT;
    }
    binaryKernel<Op><<<strideGrid(size), dim3(256, 1, 1), 0, stream>>>(
        d_input1, d_input2, d_output, size, Op{});
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        tacuda::detail::set_last_error(cudaGetErrorString(err));
        return CT_STATUS_KERNEL_FAILED;
    }
    return CT_STATUS_SUCCESS;
}

template <typename Op>
ctStatus_t run_unary_host(const float* h_in, float* h_out, int size,
                          cudaStream_t stream) try {
    tacuda::detail::clear_last_error();
    if (!h_in || !h_out || size <= 0) {
        tacuda::detail::set_last_error("vector math: invalid arguments");
        return CT_STATUS_INVALID_ARGUMENT;
    }
    auto d_in = acquireDeviceBufferAsync<float>(static_cast<size_t>(size), stream);
    auto d_out = acquireDeviceBufferAsync<float>(static_cast<size_t>(size), stream);
    if (!d_in || !d_out) {
        tacuda::detail::set_last_error("vector math: device allocation failed");
        return CT_STATUS_ALLOC_FAILED;
    }

    cudaError_t err = cudaMemcpyAsync(d_in.get(), h_in, size * sizeof(float),
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        tacuda::detail::set_last_error(cudaGetErrorString(err));
        return CT_STATUS_COPY_FAILED;
    }

    ctStatus_t st = launch_unary_device<Op>(d_in.get(), d_out.get(), size, stream);
    if (st != CT_STATUS_SUCCESS) return st;

    err = cudaMemcpyAsync(h_out, d_out.get(), size * sizeof(float),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        tacuda::detail::set_last_error(cudaGetErrorString(err));
        return CT_STATUS_COPY_FAILED;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        tacuda::detail::set_last_error(cudaGetErrorString(err));
        return CT_STATUS_DEVICE_ERROR;
    }
    return CT_STATUS_SUCCESS;
} catch (const std::exception& e) {
    tacuda::detail::set_last_error(e.what());
    return CT_STATUS_ALLOC_FAILED;
} catch (...) {
    tacuda::detail::set_last_error("vector math: unknown error");
    return CT_STATUS_KERNEL_FAILED;
}

template <typename Op>
ctStatus_t run_binary_host(const float* h_in1, const float* h_in2, float* h_out,
                           int size, cudaStream_t stream) try {
    tacuda::detail::clear_last_error();
    if (!h_in1 || !h_in2 || !h_out || size <= 0) {
        tacuda::detail::set_last_error("vector math: invalid arguments");
        return CT_STATUS_INVALID_ARGUMENT;
    }
    auto d_in1 = acquireDeviceBufferAsync<float>(static_cast<size_t>(size), stream);
    auto d_in2 = acquireDeviceBufferAsync<float>(static_cast<size_t>(size), stream);
    auto d_out = acquireDeviceBufferAsync<float>(static_cast<size_t>(size), stream);
    if (!d_in1 || !d_in2 || !d_out) {
        tacuda::detail::set_last_error("vector math: device allocation failed");
        return CT_STATUS_ALLOC_FAILED;
    }

    cudaError_t err = cudaMemcpyAsync(d_in1.get(), h_in1, size * sizeof(float),
                                      cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(d_in2.get(), h_in2, size * sizeof(float),
                              cudaMemcpyHostToDevice, stream);
    }
    if (err != cudaSuccess) {
        tacuda::detail::set_last_error(cudaGetErrorString(err));
        return CT_STATUS_COPY_FAILED;
    }

    ctStatus_t st = launch_binary_device<Op>(d_in1.get(), d_in2.get(),
                                             d_out.get(), size, stream);
    if (st != CT_STATUS_SUCCESS) return st;

    err = cudaMemcpyAsync(h_out, d_out.get(), size * sizeof(float),
                          cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        tacuda::detail::set_last_error(cudaGetErrorString(err));
        return CT_STATUS_COPY_FAILED;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        tacuda::detail::set_last_error(cudaGetErrorString(err));
        return CT_STATUS_DEVICE_ERROR;
    }
    return CT_STATUS_SUCCESS;
} catch (const std::exception& e) {
    tacuda::detail::set_last_error(e.what());
    return CT_STATUS_ALLOC_FAILED;
} catch (...) {
    tacuda::detail::set_last_error("vector math: unknown error");
    return CT_STATUS_KERNEL_FAILED;
}

} // namespace

extern "C" {

#define TACUDA_DEFINE_UNARY(name, Op)                                          \
  ctStatus_t ct_##name(const float* host_input, float* host_output, int size,  \
                       cudaStream_t stream) {                                  \
    return run_unary_host<Op>(host_input, host_output, size, stream);          \
  }                                                                            \
  ctStatus_t ct_##name##_d(const float* d_input, float* d_output, int size,    \
                           cudaStream_t stream) {                              \
    return launch_unary_device<Op>(d_input, d_output, size, stream);           \
  }

#define TACUDA_DEFINE_BINARY(name, Op)                                         \
  ctStatus_t ct_##name(const float* host_input1, const float* host_input2,     \
                       float* host_output, int size, cudaStream_t stream) {    \
    return run_binary_host<Op>(host_input1, host_input2, host_output, size,    \
                               stream);                                        \
  }                                                                            \
  ctStatus_t ct_##name##_d(const float* d_input1, const float* d_input2,       \
                           float* d_output, int size, cudaStream_t stream) {   \
    return launch_binary_device<Op>(d_input1, d_input2, d_output, size,        \
                                    stream);                                   \
  }

TACUDA_DEFINE_UNARY(acos, OpAcos)
TACUDA_DEFINE_UNARY(asin, OpAsin)
TACUDA_DEFINE_UNARY(atan, OpAtan)
TACUDA_DEFINE_UNARY(ceil, OpCeil)
TACUDA_DEFINE_UNARY(cos, OpCos)
TACUDA_DEFINE_UNARY(cosh, OpCosh)
TACUDA_DEFINE_UNARY(exp, OpExp)
TACUDA_DEFINE_UNARY(floor, OpFloor)
TACUDA_DEFINE_UNARY(ln, OpLn)
TACUDA_DEFINE_UNARY(log10, OpLog10)
TACUDA_DEFINE_UNARY(sin, OpSin)
TACUDA_DEFINE_UNARY(sinh, OpSinh)
TACUDA_DEFINE_UNARY(sqrt, OpSqrt)
TACUDA_DEFINE_UNARY(tan, OpTan)
TACUDA_DEFINE_UNARY(tanh, OpTanh)

TACUDA_DEFINE_BINARY(add, OpAdd)
TACUDA_DEFINE_BINARY(sub, OpSub)
TACUDA_DEFINE_BINARY(mult, OpMult)
TACUDA_DEFINE_BINARY(div, OpDiv)

#undef TACUDA_DEFINE_UNARY
#undef TACUDA_DEFINE_BINARY

} // extern "C"
