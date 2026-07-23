/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
/**
 * @file api_device.cu
 * @brief Device-pointer API implementation for tacuda indicators
 * 
 * These functions work directly with GPU memory, eliminating transfer overhead.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <limits>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include "tacuda.h"
#include "tacuda_device.h"
#include <utils/ErrorState.h>
#include <utils/Precision.h>

// Include all indicator headers
#include <indicators/SMA.h>
#include <indicators/EMA.h>
#include <indicators/WMA.h>
#include <indicators/DEMA.h>
#include <indicators/TEMA.h>
#include <indicators/TRIMA.h>
#include <indicators/KAMA.h>
#include <indicators/T3.h>
#include <indicators/RSI.h>
#include <indicators/Momentum.h>
#include <indicators/ROC.h>
#include <indicators/ROCP.h>
#include <indicators/ROCR.h>
#include <indicators/ROCR100.h>
#include <indicators/CMO.h>
#include <indicators/ATR.h>
#include <indicators/NATR.h>
#include <indicators/TRANGE.h>
#include <indicators/ADX.h>
#include <indicators/ADXR.h>
#include <indicators/DX.h>
#include <indicators/PlusDI.h>
#include <indicators/MinusDI.h>
#include <indicators/MACD.h>
#include <indicators/MACDEXT.h>
#include <indicators/Stochastic.h>
#include <indicators/StochasticFast.h>
#include <indicators/WILLR.h>
#include <indicators/CCI.h>
#include <indicators/ULTOSC.h>
#include <indicators/BBANDS.h>
#include <indicators/ACCBANDS.h>
#include <indicators/AD.h>
#include <indicators/ADOSC.h>
#include <indicators/OBV.h>
#include <indicators/MFI.h>
#include <indicators/StdDev.h>
#include <indicators/VAR.h>
#include <indicators/SUM.h>
#include <indicators/MAX.h>
#include <indicators/MIN.h>
#include <indicators/AvgPrice.h>
#include <indicators/MedPrice.h>
#include <indicators/TypPrice.h>
#include <indicators/WclPrice.h>
#include <indicators/LINEARREG.h>
#include <indicators/LINEARREG_SLOPE.h>
#include <indicators/LINEARREG_INTERCEPT.h>
#include <indicators/LINEARREG_ANGLE.h>
#include <indicators/TSF.h>
#include <indicators/Aroon.h>
#include <indicators/AroonOscillator.h>

// Headers for the generated ct_*_d wrappers (0.4.0 families + club pack).
#include "api_device_generated_includes.inc"

namespace {
// Re-throw current exception, classify it, and store the message.
static ctStatus_t classify_exception(const char* context = nullptr) {
    try { throw; }
    catch (const std::invalid_argument& e) {
        tacuda::detail::set_last_error(e.what());
        return CT_STATUS_INVALID_ARGUMENT;
    }
    catch (const std::runtime_error& e) {
        tacuda::detail::set_last_error(e.what());
        return CT_STATUS_DEVICE_ERROR;
    }
    catch (const std::exception& e) {
        tacuda::detail::set_last_error(e.what());
        return CT_STATUS_KERNEL_FAILED;
    }
    catch (...) {
        tacuda::detail::set_last_error(context ? context : "unknown error");
        return CT_STATUS_KERNEL_FAILED;
    }
}

template <typename DstT, typename SrcT>
__global__ void convertKernel(const SrcT* src, DstT* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = static_cast<DstT>(src[i]);
    }
}

template <>
__global__ void convertKernel<__half, float>(const float* src, __half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

template <>
__global__ void convertKernel<float, __half>(const __half* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

template <>
__global__ void convertKernel<double, __half>(const __half* src, double* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = static_cast<double>(__half2float(src[i]));
    }
}

template <>
__global__ void convertKernel<__half, double>(const double* src, __half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(static_cast<float>(src[i]));
    }
}

static ctStatus_t launch_convert(const void* src, ctDataType_t srcType,
                                 void* dst, ctDataType_t dstType,
                                 int count, cudaStream_t stream) {
    if (srcType == dstType) {
        const size_t elem = tacuda::precision::element_size(srcType);
        if (elem == 0) {
            tacuda::detail::set_last_error("launch_convert: unknown dtype");
            return CT_STATUS_INVALID_ARGUMENT;
        }
        const size_t bytes = static_cast<size_t>(count) * elem;
        cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
        return (err == cudaSuccess) ? CT_STATUS_SUCCESS : CT_STATUS_COPY_FAILED;
    }
    const int block = 256;
    const int grid = (count + block - 1) / block;
    if (srcType == CT_DTYPE_F16 && dstType == CT_DTYPE_F32) {
        convertKernel<float, __half><<<grid, block, 0, stream>>>(
            static_cast<const __half*>(src), static_cast<float*>(dst), count);
    } else if (srcType == CT_DTYPE_F32 && dstType == CT_DTYPE_F16) {
        convertKernel<__half, float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(src), static_cast<__half*>(dst), count);
    } else if (srcType == CT_DTYPE_F64 && dstType == CT_DTYPE_F32) {
        convertKernel<float, double><<<grid, block, 0, stream>>>(
            static_cast<const double*>(src), static_cast<float*>(dst), count);
    } else if (srcType == CT_DTYPE_F32 && dstType == CT_DTYPE_F64) {
        convertKernel<double, float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(src), static_cast<double*>(dst), count);
    } else if (srcType == CT_DTYPE_F16 && dstType == CT_DTYPE_F64) {
        convertKernel<double, __half><<<grid, block, 0, stream>>>(
            static_cast<const __half*>(src), static_cast<double*>(dst), count);
    } else if (srcType == CT_DTYPE_F64 && dstType == CT_DTYPE_F16) {
        convertKernel<__half, double><<<grid, block, 0, stream>>>(
            static_cast<const double*>(src), static_cast<__half*>(dst), count);
    } else {
        return CT_STATUS_UNSUPPORTED;
    }
    cudaError_t kerr = cudaGetLastError();
    return (kerr == cudaSuccess) ? CT_STATUS_SUCCESS : CT_STATUS_KERNEL_FAILED;
}

template <typename T>
__global__ void smaKernelCompute(const T* input, T* output, int size, int period) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const T qnan = std::numeric_limits<T>::quiet_NaN();
    if (idx < period - 1) {
        output[idx] = qnan;
        return;
    }
    T sum = static_cast<T>(0);
    for (int j = idx - period + 1; j <= idx; ++j) sum += input[j];
    output[idx] = sum / static_cast<T>(period);
}

template <typename T>
__global__ void emaKernelCompute(const T* input, T* output, int size, int period) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const T qnan = std::numeric_limits<T>::quiet_NaN();
    if (size <= 0 || period <= 0) return;
    for (int i = 0; i < size; ++i) output[i] = qnan;
    if (size < period) return;
    T seed = static_cast<T>(0);
    for (int i = 0; i < period; ++i) seed += input[i];
    seed /= static_cast<T>(period);
    output[period - 1] = seed;
    const T alpha = static_cast<T>(2.0) / static_cast<T>(period + 1);
    T prev = seed;
    for (int i = period; i < size; ++i) {
        prev = prev + alpha * (input[i] - prev);
        output[i] = prev;
    }
}

template <typename T>
__global__ void rsiKernelCompute(const T* input, T* output, int size, int period) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const T qnan = std::numeric_limits<T>::quiet_NaN();
    if (size <= 0 || period <= 0) return;
    for (int i = 0; i < size; ++i) output[i] = qnan;
    if (size <= period) return;

    T gain = static_cast<T>(0);
    T loss = static_cast<T>(0);
    for (int i = 1; i <= period; ++i) {
        T d = input[i] - input[i - 1];
        if (d > static_cast<T>(0)) gain += d;
        else loss -= d;
    }
    T avgGain = gain / static_cast<T>(period);
    T avgLoss = loss / static_cast<T>(period);
    T rs = (avgLoss == static_cast<T>(0)) ? static_cast<T>(0) : (avgGain / avgLoss);
    output[period] = (avgLoss == static_cast<T>(0))
                     ? static_cast<T>(100)
                     : static_cast<T>(100) - static_cast<T>(100) / (static_cast<T>(1) + rs);

    for (int i = period + 1; i < size; ++i) {
        T d = input[i] - input[i - 1];
        T g = d > static_cast<T>(0) ? d : static_cast<T>(0);
        T l = d < static_cast<T>(0) ? -d : static_cast<T>(0);
        avgGain = (avgGain * static_cast<T>(period - 1) + g) / static_cast<T>(period);
        avgLoss = (avgLoss * static_cast<T>(period - 1) + l) / static_cast<T>(period);
        if (avgLoss == static_cast<T>(0)) output[i] = static_cast<T>(100);
        else {
            rs = avgGain / avgLoss;
            output[i] = static_cast<T>(100) - static_cast<T>(100) / (static_cast<T>(1) + rs);
        }
    }
}

template <typename T>
__global__ void atrKernelCompute(const T* high, const T* low, const T* close,
                                 T* output, int size, int period) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const T qnan = std::numeric_limits<T>::quiet_NaN();
    for (int i = 0; i < size; ++i) output[i] = qnan;
    if (size <= period || period <= 0) return;

    T trSum = static_cast<T>(0);
    for (int i = 1; i <= period; ++i) {
        T hl = high[i] - low[i];
        T hc = fabs(high[i] - close[i - 1]);
        T lc = fabs(low[i] - close[i - 1]);
        T tr = hl;
        if (hc > tr) tr = hc;
        if (lc > tr) tr = lc;
        trSum += tr;
    }
    T atr = trSum / static_cast<T>(period);
    output[period] = atr;

    for (int i = period + 1; i < size; ++i) {
        T hl = high[i] - low[i];
        T hc = fabs(high[i] - close[i - 1]);
        T lc = fabs(low[i] - close[i - 1]);
        T tr = hl;
        if (hc > tr) tr = hc;
        if (lc > tr) tr = lc;
        atr = (atr * static_cast<T>(period - 1) + tr) / static_cast<T>(period);
        output[i] = atr;
    }
}

template <typename T>
__global__ void bbandsKernelCompute(const T* input, T* upper, T* middle, T* lower,
                                    int size, int period, T nbDevUp, T nbDevDn) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    const T qnan = std::numeric_limits<T>::quiet_NaN();
    if (idx < period - 1) {
        upper[idx] = middle[idx] = lower[idx] = qnan;
        return;
    }
    T sum = static_cast<T>(0);
    for (int j = idx - period + 1; j <= idx; ++j) sum += input[j];
    T mean = sum / static_cast<T>(period);
    T var = static_cast<T>(0);
    for (int j = idx - period + 1; j <= idx; ++j) {
        T d = input[j] - mean;
        var += d * d;
    }
    T stdev = sqrt(var / static_cast<T>(period));
    middle[idx] = mean;
    upper[idx] = mean + nbDevUp * stdev;
    lower[idx] = mean - nbDevDn * stdev;
}

template <typename T>
__global__ void macdKernelCompute(const T* input, T* macd, T* signal, T* hist,
                                  T* fastEma, T* slowEma, int size,
                                  int fastPeriod, int slowPeriod, int signalPeriod) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const T qnan = std::numeric_limits<T>::quiet_NaN();
    for (int i = 0; i < size; ++i) {
        macd[i] = signal[i] = hist[i] = fastEma[i] = slowEma[i] = qnan;
    }
    if (size <= slowPeriod || fastPeriod <= 0 || slowPeriod <= 0 || signalPeriod <= 0) return;

    // fast ema
    T fastSeed = static_cast<T>(0);
    for (int i = 0; i < fastPeriod; ++i) fastSeed += input[i];
    fastSeed /= static_cast<T>(fastPeriod);
    fastEma[fastPeriod - 1] = fastSeed;
    T fastAlpha = static_cast<T>(2.0) / static_cast<T>(fastPeriod + 1);
    T prevFast = fastSeed;
    for (int i = fastPeriod; i < size; ++i) {
        prevFast = prevFast + fastAlpha * (input[i] - prevFast);
        fastEma[i] = prevFast;
    }

    // slow ema
    T slowSeed = static_cast<T>(0);
    for (int i = 0; i < slowPeriod; ++i) slowSeed += input[i];
    slowSeed /= static_cast<T>(slowPeriod);
    slowEma[slowPeriod - 1] = slowSeed;
    T slowAlpha = static_cast<T>(2.0) / static_cast<T>(slowPeriod + 1);
    T prevSlow = slowSeed;
    for (int i = slowPeriod; i < size; ++i) {
        prevSlow = prevSlow + slowAlpha * (input[i] - prevSlow);
        slowEma[i] = prevSlow;
        if (!isnan(static_cast<double>(fastEma[i]))) {
            macd[i] = fastEma[i] - slowEma[i];
        }
    }

    int signalStart = slowPeriod - 1 + signalPeriod - 1;
    if (signalStart >= size) return;
    T sigSeed = static_cast<T>(0);
    for (int i = slowPeriod - 1; i <= signalStart; ++i) sigSeed += macd[i];
    sigSeed /= static_cast<T>(signalPeriod);
    signal[signalStart] = sigSeed;
    hist[signalStart] = macd[signalStart] - sigSeed;
    T sigAlpha = static_cast<T>(2.0) / static_cast<T>(signalPeriod + 1);
    T prevSig = sigSeed;
    for (int i = signalStart + 1; i < size; ++i) {
        prevSig = prevSig + sigAlpha * (macd[i] - prevSig);
        signal[i] = prevSig;
        hist[i] = macd[i] - prevSig;
    }
}

template <typename ComputeT, typename KernelLauncher>
static ctStatus_t run_single_input_compute_path(const void* d_input, void* d_output, int size,
                                                ctDataType_t storageType, cudaStream_t stream,
                                                KernelLauncher&& launcher) {
    ComputeT* in = nullptr;
    ComputeT* out = nullptr;
    if (cudaMalloc(&in, static_cast<size_t>(size) * sizeof(ComputeT)) != cudaSuccess ||
        cudaMalloc(&out, static_cast<size_t>(size) * sizeof(ComputeT)) != cudaSuccess) {
        if (in) cudaFree(in);
        if (out) cudaFree(out);
        return CT_STATUS_ALLOC_FAILED;
    }
    auto cleanup = [&]() { cudaFree(in); cudaFree(out); };
    ctStatus_t st = launch_convert(d_input, storageType, in,
                                   std::is_same<ComputeT, double>::value ? CT_DTYPE_F64 : CT_DTYPE_F32,
                                   size, stream);
    if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
    st = launcher(in, out);
    if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
    st = launch_convert(out,
                        std::is_same<ComputeT, double>::value ? CT_DTYPE_F64 : CT_DTYPE_F32,
                        d_output, storageType, size, stream);
    cleanup();
    return st;
}
} // namespace

extern "C" {

// =============================================================================
// Memory Management
// =============================================================================

float* ct_device_alloc(int count) {
    return static_cast<float*>(ct_device_alloc_ex(count, CT_DTYPE_F32));
}

void* ct_device_alloc_ex(int count, ctDataType_t storageType) {
    tacuda::detail::clear_last_error();
    if (count <= 0) {
        tacuda::detail::set_last_error("ct_device_alloc_ex: count must be > 0");
        return nullptr;
    }
    const size_t elem = tacuda::precision::element_size(storageType);
    if (elem == 0) {
        tacuda::detail::set_last_error("ct_device_alloc_ex: unknown dtype");
        return nullptr;
    }
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, static_cast<size_t>(count) * elem);
    if (err != cudaSuccess) {
        tacuda::detail::set_last_error(cudaGetErrorString(err));
        return nullptr;
    }
    return ptr;
}

void ct_device_free(float* d_ptr) {
    ct_device_free_ex(d_ptr);
}

void ct_device_free_ex(void* d_ptr) {
    if (d_ptr) cudaFree(d_ptr);
}

ctStatus_t ct_device_upload(float* d_dst, const float* h_src, int count, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    return ct_device_upload_ex(d_dst, h_src, count, CT_DTYPE_F32, stream);
}

ctStatus_t ct_device_upload_ex(void* d_dst, const void* h_src, int count,
                               ctDataType_t storageType, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_dst || !h_src || count <= 0) {
        tacuda::detail::set_last_error("ct_device_upload_ex: null pointer or invalid count");
        return CT_STATUS_INVALID_ARGUMENT;
    }
    const size_t elem = tacuda::precision::element_size(storageType);
    if (elem == 0) {
        tacuda::detail::set_last_error("ct_device_upload_ex: unknown dtype");
        return CT_STATUS_INVALID_ARGUMENT;
    }
    cudaError_t err = cudaMemcpyAsync(d_dst, h_src,
                                      static_cast<size_t>(count) * elem,
                                      cudaMemcpyHostToDevice, stream);
    return (err == cudaSuccess) ? CT_STATUS_SUCCESS : CT_STATUS_COPY_FAILED;
}

ctStatus_t ct_device_download(float* h_dst, const float* d_src, int count, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    return ct_device_download_ex(h_dst, d_src, count, CT_DTYPE_F32, stream);
}

ctStatus_t ct_device_download_ex(void* h_dst, const void* d_src, int count,
                                 ctDataType_t storageType, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!h_dst || !d_src || count <= 0) {
        tacuda::detail::set_last_error("ct_device_download_ex: null pointer or invalid count");
        return CT_STATUS_INVALID_ARGUMENT;
    }
    const size_t elem = tacuda::precision::element_size(storageType);
    if (elem == 0) {
        tacuda::detail::set_last_error("ct_device_download_ex: unknown dtype");
        return CT_STATUS_INVALID_ARGUMENT;
    }
    cudaError_t err = cudaMemcpyAsync(h_dst, d_src,
                                      static_cast<size_t>(count) * elem,
                                      cudaMemcpyDeviceToHost, stream);
    return (err == cudaSuccess) ? CT_STATUS_SUCCESS : CT_STATUS_COPY_FAILED;
}

ctStatus_t ct_device_sync(cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        tacuda::detail::set_last_error(cudaGetErrorString(err));
        return CT_STATUS_DEVICE_ERROR;
    }
    return CT_STATUS_SUCCESS;
}

// =============================================================================
// Helper macro for simple single-input indicators
// =============================================================================

#define IMPL_DEVICE_INDICATOR_1IN(name, IndicatorClass, ...) \
ctStatus_t name(const float* d_input, float* d_output, int size, \
                int period, cudaStream_t stream) { \
    if (!d_input || !d_output || size <= 0) { \
        tacuda::detail::set_last_error("null pointer or invalid size"); \
        return CT_STATUS_INVALID_ARGUMENT; \
    } \
    try { \
        tacuda::IndicatorClass ind(__VA_ARGS__); \
        ind.calculate(d_input, d_output, size, stream); \
        return CT_STATUS_SUCCESS; \
    } catch (...) { \
        return classify_exception(); \
    } \
}

// =============================================================================
// Moving Averages
// =============================================================================

ctStatus_t ct_sma_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::SMA ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_sma_ex_d(const void* d_input, void* d_output, int size, int period,
                       ctDataType_t storageType, ctDataType_t computeType, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) {
        tacuda::detail::set_last_error("invalid argument");
        return CT_STATUS_INVALID_ARGUMENT;
    }
    if (!tacuda::precision::is_supported_compute_pair(storageType, computeType)) {
        tacuda::detail::set_last_error("unsupported storage/compute dtype pair");
        return CT_STATUS_UNSUPPORTED;
    }
    if (storageType == CT_DTYPE_F32 && computeType == CT_DTYPE_F32) {
        return ct_sma_d(static_cast<const float*>(d_input), static_cast<float*>(d_output), size, period, stream);
    }
    if (computeType == CT_DTYPE_F32) {
        return run_single_input_compute_path<float>(
            d_input, d_output, size, storageType, stream,
            [=](const float* in, float* out) {
                const int block = 256, grid = (size + block - 1) / block;
                smaKernelCompute<float><<<grid, block, 0, stream>>>(in, out, size, period);
                return (cudaGetLastError() == cudaSuccess) ? CT_STATUS_SUCCESS : CT_STATUS_KERNEL_FAILED;
            });
    }
    return run_single_input_compute_path<double>(
        d_input, d_output, size, storageType, stream,
        [=](const double* in, double* out) {
            const int block = 256, grid = (size + block - 1) / block;
            smaKernelCompute<double><<<grid, block, 0, stream>>>(in, out, size, period);
            return (cudaGetLastError() == cudaSuccess) ? CT_STATUS_SUCCESS : CT_STATUS_KERNEL_FAILED;
        });
}

ctStatus_t ct_ema_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::EMA ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_ema_ex_d(const void* d_input, void* d_output, int size, int period,
                       ctDataType_t storageType, ctDataType_t computeType, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) {
        tacuda::detail::set_last_error("invalid argument");
        return CT_STATUS_INVALID_ARGUMENT;
    }
    if (!tacuda::precision::is_supported_compute_pair(storageType, computeType)) {
        tacuda::detail::set_last_error("unsupported storage/compute dtype pair");
        return CT_STATUS_UNSUPPORTED;
    }
    if (storageType == CT_DTYPE_F32 && computeType == CT_DTYPE_F32) {
        return ct_ema_d(static_cast<const float*>(d_input), static_cast<float*>(d_output), size, period, stream);
    }
    if (computeType == CT_DTYPE_F32) {
        return run_single_input_compute_path<float>(
            d_input, d_output, size, storageType, stream,
            [=](const float* in, float* out) {
                emaKernelCompute<float><<<1, 1, 0, stream>>>(in, out, size, period);
                return (cudaGetLastError() == cudaSuccess) ? CT_STATUS_SUCCESS : CT_STATUS_KERNEL_FAILED;
            });
    }
    return run_single_input_compute_path<double>(
        d_input, d_output, size, storageType, stream,
        [=](const double* in, double* out) {
            emaKernelCompute<double><<<1, 1, 0, stream>>>(in, out, size, period);
            return (cudaGetLastError() == cudaSuccess) ? CT_STATUS_SUCCESS : CT_STATUS_KERNEL_FAILED;
        });
}

ctStatus_t ct_wma_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::WMA ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_dema_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::DEMA ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_tema_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::TEMA ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_trima_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::TRIMA ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_kama_d(const float* d_input, float* d_output, int size, 
                     int period, int fastPeriod, int slowPeriod, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::KAMA ind(period, fastPeriod, slowPeriod);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_t3_d(const float* d_input, float* d_output, int size,
                   int period, float vFactor, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::T3 ind(period, vFactor);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

// =============================================================================
// Momentum Indicators
// =============================================================================

ctStatus_t ct_rsi_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::RSI ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_rsi_ex_d(const void* d_input, void* d_output, int size, int period,
                       ctDataType_t storageType, ctDataType_t computeType, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) {
        tacuda::detail::set_last_error("invalid argument");
        return CT_STATUS_INVALID_ARGUMENT;
    }
    if (!tacuda::precision::is_supported_compute_pair(storageType, computeType)) {
        tacuda::detail::set_last_error("unsupported storage/compute dtype pair");
        return CT_STATUS_UNSUPPORTED;
    }
    if (storageType == CT_DTYPE_F32 && computeType == CT_DTYPE_F32) {
        return ct_rsi_d(static_cast<const float*>(d_input), static_cast<float*>(d_output), size, period, stream);
    }
    if (computeType == CT_DTYPE_F32) {
        return run_single_input_compute_path<float>(
            d_input, d_output, size, storageType, stream,
            [=](const float* in, float* out) {
                rsiKernelCompute<float><<<1, 1, 0, stream>>>(in, out, size, period);
                return (cudaGetLastError() == cudaSuccess) ? CT_STATUS_SUCCESS : CT_STATUS_KERNEL_FAILED;
            });
    }
    return run_single_input_compute_path<double>(
        d_input, d_output, size, storageType, stream,
        [=](const double* in, double* out) {
            rsiKernelCompute<double><<<1, 1, 0, stream>>>(in, out, size, period);
            return (cudaGetLastError() == cudaSuccess) ? CT_STATUS_SUCCESS : CT_STATUS_KERNEL_FAILED;
        });
}

ctStatus_t ct_momentum_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::Momentum ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_roc_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::ROC ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_rocp_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::ROCP ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_rocr_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::ROCR ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_rocr100_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::ROCR100 ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_cmo_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::CMO ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

// =============================================================================
// Volatility Indicators
// =============================================================================

ctStatus_t ct_atr_d(const float* d_high, const float* d_low, const float* d_close,
                    float* d_output, int size, int period, float percentile, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::ATR ind(period, percentile);
        ind.calculate(d_high, d_low, d_close, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_atr_ex_d(const void* d_high, const void* d_low, const void* d_close,
                       void* d_output, int size, int period, float percentile,
                       ctDataType_t storageType, ctDataType_t computeType,
                       cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_output || size <= 0) {
        tacuda::detail::set_last_error("invalid argument");
        return CT_STATUS_INVALID_ARGUMENT;
    }
    if (!tacuda::precision::is_supported_compute_pair(storageType, computeType)) {
        return CT_STATUS_UNSUPPORTED;
    }
    if (storageType == CT_DTYPE_F32 && computeType == CT_DTYPE_F32) {
        return ct_atr_d(static_cast<const float*>(d_high), static_cast<const float*>(d_low),
                        static_cast<const float*>(d_close), static_cast<float*>(d_output),
                        size, period, percentile, stream);
    }
    if (computeType == CT_DTYPE_F32) {
        float *h = nullptr, *l = nullptr, *c = nullptr, *o = nullptr;
        if (cudaMalloc(&h, static_cast<size_t>(size) * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&l, static_cast<size_t>(size) * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&c, static_cast<size_t>(size) * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&o, static_cast<size_t>(size) * sizeof(float)) != cudaSuccess) {
            if (h) cudaFree(h); if (l) cudaFree(l); if (c) cudaFree(c); if (o) cudaFree(o);
            return CT_STATUS_ALLOC_FAILED;
        }
        auto cleanup = [&]() { cudaFree(h); cudaFree(l); cudaFree(c); cudaFree(o); };
        ctStatus_t st = launch_convert(d_high, storageType, h, CT_DTYPE_F32, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        st = launch_convert(d_low, storageType, l, CT_DTYPE_F32, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        st = launch_convert(d_close, storageType, c, CT_DTYPE_F32, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        atrKernelCompute<float><<<1, 1, 0, stream>>>(h, l, c, o, size, period);
        if (cudaGetLastError() != cudaSuccess) { cleanup(); return CT_STATUS_KERNEL_FAILED; }
        st = launch_convert(o, CT_DTYPE_F32, d_output, storageType, size, stream);
        cleanup();
        return st;
    } else {
        double *h = nullptr, *l = nullptr, *c = nullptr, *o = nullptr;
        if (cudaMalloc(&h, static_cast<size_t>(size) * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&l, static_cast<size_t>(size) * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&c, static_cast<size_t>(size) * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&o, static_cast<size_t>(size) * sizeof(double)) != cudaSuccess) {
            if (h) cudaFree(h); if (l) cudaFree(l); if (c) cudaFree(c); if (o) cudaFree(o);
            return CT_STATUS_ALLOC_FAILED;
        }
        auto cleanup = [&]() { cudaFree(h); cudaFree(l); cudaFree(c); cudaFree(o); };
        ctStatus_t st = launch_convert(d_high, storageType, h, CT_DTYPE_F64, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        st = launch_convert(d_low, storageType, l, CT_DTYPE_F64, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        st = launch_convert(d_close, storageType, c, CT_DTYPE_F64, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        atrKernelCompute<double><<<1, 1, 0, stream>>>(h, l, c, o, size, period);
        if (cudaGetLastError() != cudaSuccess) { cleanup(); return CT_STATUS_KERNEL_FAILED; }
        st = launch_convert(o, CT_DTYPE_F64, d_output, storageType, size, stream);
        cleanup();
        return st;
    }
}

ctStatus_t ct_natr_d(const float* d_high, const float* d_low, const float* d_close,
                     float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::NATR ind(period);
        ind.calculate(d_high, d_low, d_close, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_trange_d(const float* d_high, const float* d_low, const float* d_close,
                       float* d_output, int size, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::TRANGE ind;
        ind.calculate(d_high, d_low, d_close, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

// =============================================================================
// Trend Indicators
// =============================================================================

ctStatus_t ct_adx_d(const float* d_high, const float* d_low, const float* d_close,
                    float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::ADX ind(period);
        ind.calculate(d_high, d_low, d_close, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_adxr_d(const float* d_high, const float* d_low, const float* d_close,
                     float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::ADXR ind(period);
        ind.calculate(d_high, d_low, d_close, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_dx_d(const float* d_high, const float* d_low, const float* d_close,
                   float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::DX ind(period);
        ind.calculate(d_high, d_low, d_close, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_plus_di_d(const float* d_high, const float* d_low, const float* d_close,
                        float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::PlusDI ind(period);
        ind.calculate(d_high, d_low, d_close, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_minus_di_d(const float* d_high, const float* d_low, const float* d_close,
                         float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::MinusDI ind(period);
        ind.calculate(d_high, d_low, d_close, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

// =============================================================================
// Oscillators
// =============================================================================

ctStatus_t ct_macd_d(const float* d_input, float* d_macd, float* d_signal, float* d_hist,
                     int size, int fastPeriod, int slowPeriod, int signalPeriod,
                     int maType, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_macd || !d_signal || !d_hist || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::MACDEXT ind(fastPeriod, slowPeriod, signalPeriod, static_cast<tacuda::MAType>(maType));
        ind.calculate(d_input, d_macd, d_signal, d_hist, size, stream);
        return CT_STATUS_SUCCESS;
    } catch (...) {
        return classify_exception();
    }
}

ctStatus_t ct_macdext_d(const float* d_input, float* d_macd, float* d_signal, float* d_hist,
                        int size, int fastPeriod, int fastType, int slowPeriod,
                        int slowType, int signalPeriod, int signalType,
                        cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_macd || !d_signal || !d_hist || size <= 0) {
        tacuda::detail::set_last_error("invalid argument");
        return CT_STATUS_INVALID_ARGUMENT;
    }
    try {
        tacuda::MACDEXT ind(fastPeriod, static_cast<tacuda::MAType>(fastType),
                            slowPeriod, static_cast<tacuda::MAType>(slowType),
                            signalPeriod, static_cast<tacuda::MAType>(signalType));
        ind.calculate(d_input, d_macd, d_signal, d_hist, size, stream);
        return CT_STATUS_SUCCESS;
    } catch (...) {
        return classify_exception();
    }
}

ctStatus_t ct_macd_ex_d(const void* d_input, void* d_macd, void* d_signal, void* d_hist,
                        int size, int fastPeriod, int slowPeriod, int signalPeriod,
                        int maType, ctDataType_t storageType, ctDataType_t computeType,
                        cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_macd || !d_signal || !d_hist || size <= 0) {
        tacuda::detail::set_last_error("invalid argument");
        return CT_STATUS_INVALID_ARGUMENT;
    }
    if (!tacuda::precision::is_supported_compute_pair(storageType, computeType)) {
        return CT_STATUS_UNSUPPORTED;
    }
    if (storageType == CT_DTYPE_F32 && computeType == CT_DTYPE_F32) {
        return ct_macd_d(static_cast<const float*>(d_input), static_cast<float*>(d_macd),
                         static_cast<float*>(d_signal), static_cast<float*>(d_hist),
                         size, fastPeriod, slowPeriod, signalPeriod, maType, stream);
    }
    if (computeType == CT_DTYPE_F32) {
        float *in = nullptr, *m = nullptr, *s = nullptr, *h = nullptr, *f = nullptr, *sl = nullptr;
        if (cudaMalloc(&in, static_cast<size_t>(size) * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&m, static_cast<size_t>(size) * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&s, static_cast<size_t>(size) * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&h, static_cast<size_t>(size) * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&f, static_cast<size_t>(size) * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&sl, static_cast<size_t>(size) * sizeof(float)) != cudaSuccess) {
            if (in) cudaFree(in); if (m) cudaFree(m); if (s) cudaFree(s); if (h) cudaFree(h); if (f) cudaFree(f); if (sl) cudaFree(sl);
            return CT_STATUS_ALLOC_FAILED;
        }
        auto cleanup = [&]() { cudaFree(in); cudaFree(m); cudaFree(s); cudaFree(h); cudaFree(f); cudaFree(sl); };
        ctStatus_t st = launch_convert(d_input, storageType, in, CT_DTYPE_F32, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        macdKernelCompute<float><<<1, 1, 0, stream>>>(in, m, s, h, f, sl, size, fastPeriod, slowPeriod, signalPeriod);
        if (cudaGetLastError() != cudaSuccess) { cleanup(); return CT_STATUS_KERNEL_FAILED; }
        st = launch_convert(m, CT_DTYPE_F32, d_macd, storageType, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        st = launch_convert(s, CT_DTYPE_F32, d_signal, storageType, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        st = launch_convert(h, CT_DTYPE_F32, d_hist, storageType, size, stream);
        cleanup();
        return st;
    } else {
        double *in = nullptr, *m = nullptr, *s = nullptr, *h = nullptr, *f = nullptr, *sl = nullptr;
        if (cudaMalloc(&in, static_cast<size_t>(size) * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&m, static_cast<size_t>(size) * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&s, static_cast<size_t>(size) * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&h, static_cast<size_t>(size) * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&f, static_cast<size_t>(size) * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&sl, static_cast<size_t>(size) * sizeof(double)) != cudaSuccess) {
            if (in) cudaFree(in); if (m) cudaFree(m); if (s) cudaFree(s); if (h) cudaFree(h); if (f) cudaFree(f); if (sl) cudaFree(sl);
            return CT_STATUS_ALLOC_FAILED;
        }
        auto cleanup = [&]() { cudaFree(in); cudaFree(m); cudaFree(s); cudaFree(h); cudaFree(f); cudaFree(sl); };
        ctStatus_t st = launch_convert(d_input, storageType, in, CT_DTYPE_F64, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        macdKernelCompute<double><<<1, 1, 0, stream>>>(in, m, s, h, f, sl, size, fastPeriod, slowPeriod, signalPeriod);
        if (cudaGetLastError() != cudaSuccess) { cleanup(); return CT_STATUS_KERNEL_FAILED; }
        st = launch_convert(m, CT_DTYPE_F64, d_macd, storageType, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        st = launch_convert(s, CT_DTYPE_F64, d_signal, storageType, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        st = launch_convert(h, CT_DTYPE_F64, d_hist, storageType, size, stream);
        cleanup();
        return st;
    }
}

ctStatus_t ct_stochastic_d(const float* d_high, const float* d_low, const float* d_close,
                           float* d_slowK, float* d_slowD, int size, int kPeriod,
                           int dPeriod, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_slowK || !d_slowD || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::Stochastic ind(kPeriod, dPeriod);
        ind.calculate(d_high, d_low, d_close, d_slowK, d_slowD, size, stream);
        return CT_STATUS_SUCCESS;
    } catch (...) {
        return classify_exception();
    }
}

ctStatus_t ct_stochastic_fast_d(const float* d_high, const float* d_low, const float* d_close,
                                float* d_fastK, float* d_fastD, int size, int kPeriod,
                                int dPeriod, int /*dMaType*/, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_fastK || !d_fastD || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    float* tmp = nullptr;
    const size_t bytes = static_cast<size_t>(size) * 2 * sizeof(float);
    if (cudaMalloc(&tmp, bytes) != cudaSuccess) return CT_STATUS_ALLOC_FAILED;
    auto cleanup = [&]() { cudaFree(tmp); };
    try {
        tacuda::StochasticFast ind(kPeriod, dPeriod);
        ind.calculate(d_high, d_low, d_close, tmp, size, stream);
        cudaError_t e1 = cudaMemcpyAsync(d_fastK, tmp,          size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaError_t e2 = cudaMemcpyAsync(d_fastD, tmp + size,   size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cleanup();
        if (e1 != cudaSuccess || e2 != cudaSuccess) return CT_STATUS_COPY_FAILED;
        return CT_STATUS_SUCCESS;
    } catch (...) {
        cleanup();
        return CT_STATUS_KERNEL_FAILED;
    }
}

ctStatus_t ct_willr_d(const float* d_high, const float* d_low, const float* d_close,
                      float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::WILLR ind(period);
        ind.calculate(d_high, d_low, d_close, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_cci_d(const float* d_high, const float* d_low, const float* d_close,
                    float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::CCI ind(period);
        ind.calculate(d_high, d_low, d_close, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_ultosc_d(const float* d_high, const float* d_low, const float* d_close,
                       float* d_output, int size, int period1, int period2, int period3,
                       cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::ULTOSC ind(period1, period2, period3);
        ind.calculate(d_high, d_low, d_close, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

// =============================================================================
// Bands
// =============================================================================

ctStatus_t ct_bbands_d(const float* d_input, float* d_upper, float* d_middle, float* d_lower,
                       int size, int period, float nbDevUp, float nbDevDn, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_upper || !d_middle || !d_lower || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::BBANDS ind(period, nbDevUp, nbDevDn);
        ind.calculate(d_input, d_upper, d_middle, d_lower, size, stream);
        return CT_STATUS_SUCCESS;
    } catch (...) {
        return classify_exception();
    }
}

ctStatus_t ct_bbands_ex_d(const void* d_input, void* d_upper, void* d_middle, void* d_lower,
                          int size, int period, float nbDevUp, float nbDevDn,
                          ctDataType_t storageType, ctDataType_t computeType,
                          cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_upper || !d_middle || !d_lower || size <= 0) {
        tacuda::detail::set_last_error("invalid argument");
        return CT_STATUS_INVALID_ARGUMENT;
    }
    if (!tacuda::precision::is_supported_compute_pair(storageType, computeType)) {
        return CT_STATUS_UNSUPPORTED;
    }
    if (storageType == CT_DTYPE_F32 && computeType == CT_DTYPE_F32) {
        return ct_bbands_d(static_cast<const float*>(d_input), static_cast<float*>(d_upper),
                           static_cast<float*>(d_middle), static_cast<float*>(d_lower),
                           size, period, nbDevUp, nbDevDn, stream);
    }
    if (computeType == CT_DTYPE_F32) {
        float *in = nullptr, *u = nullptr, *m = nullptr, *l = nullptr;
        if (cudaMalloc(&in, static_cast<size_t>(size) * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&u, static_cast<size_t>(size) * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&m, static_cast<size_t>(size) * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&l, static_cast<size_t>(size) * sizeof(float)) != cudaSuccess) {
            if (in) cudaFree(in); if (u) cudaFree(u); if (m) cudaFree(m); if (l) cudaFree(l);
            return CT_STATUS_ALLOC_FAILED;
        }
        auto cleanup = [&]() { cudaFree(in); cudaFree(u); cudaFree(m); cudaFree(l); };
        ctStatus_t st = launch_convert(d_input, storageType, in, CT_DTYPE_F32, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        const int block = 256, grid = (size + block - 1) / block;
        bbandsKernelCompute<float><<<grid, block, 0, stream>>>(in, u, m, l, size, period, nbDevUp, nbDevDn);
        if (cudaGetLastError() != cudaSuccess) { cleanup(); return CT_STATUS_KERNEL_FAILED; }
        st = launch_convert(u, CT_DTYPE_F32, d_upper, storageType, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        st = launch_convert(m, CT_DTYPE_F32, d_middle, storageType, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        st = launch_convert(l, CT_DTYPE_F32, d_lower, storageType, size, stream);
        cleanup();
        return st;
    } else {
        double *in = nullptr, *u = nullptr, *m = nullptr, *l = nullptr;
        if (cudaMalloc(&in, static_cast<size_t>(size) * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&u, static_cast<size_t>(size) * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&m, static_cast<size_t>(size) * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&l, static_cast<size_t>(size) * sizeof(double)) != cudaSuccess) {
            if (in) cudaFree(in); if (u) cudaFree(u); if (m) cudaFree(m); if (l) cudaFree(l);
            return CT_STATUS_ALLOC_FAILED;
        }
        auto cleanup = [&]() { cudaFree(in); cudaFree(u); cudaFree(m); cudaFree(l); };
        ctStatus_t st = launch_convert(d_input, storageType, in, CT_DTYPE_F64, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        const int block = 256, grid = (size + block - 1) / block;
        bbandsKernelCompute<double><<<grid, block, 0, stream>>>(in, u, m, l, size, period, nbDevUp, nbDevDn);
        if (cudaGetLastError() != cudaSuccess) { cleanup(); return CT_STATUS_KERNEL_FAILED; }
        st = launch_convert(u, CT_DTYPE_F64, d_upper, storageType, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        st = launch_convert(m, CT_DTYPE_F64, d_middle, storageType, size, stream);
        if (st != CT_STATUS_SUCCESS) { cleanup(); return st; }
        st = launch_convert(l, CT_DTYPE_F64, d_lower, storageType, size, stream);
        cleanup();
        return st;
    }
}

ctStatus_t ct_accbands_d(const float* d_high, const float* d_low, const float* d_close,
                         float* d_upper, float* d_middle, float* d_lower,
                         int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_upper || !d_middle || !d_lower || size <= 0) 
        { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    float* tmp = nullptr;
    const size_t bytes = static_cast<size_t>(size) * 3 * sizeof(float);
    if (cudaMalloc(&tmp, bytes) != cudaSuccess) return CT_STATUS_ALLOC_FAILED;
    auto cleanup = [&]() { cudaFree(tmp); };
    try {
        tacuda::ACCBANDS ind(period);
        ind.calculate(d_high, d_low, d_close, tmp, size, stream);
        cudaError_t e1 = cudaMemcpyAsync(d_upper,  tmp,            size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaError_t e2 = cudaMemcpyAsync(d_middle, tmp + size,     size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaError_t e3 = cudaMemcpyAsync(d_lower,  tmp + 2 * size, size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cleanup();
        if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) return CT_STATUS_COPY_FAILED;
        return CT_STATUS_SUCCESS;
    } catch (...) {
        cleanup();
        return CT_STATUS_KERNEL_FAILED;
    }
}

// =============================================================================
// Volume Indicators
// =============================================================================

ctStatus_t ct_ad_d(const float* d_high, const float* d_low, const float* d_close,
                   const float* d_volume, float* d_output, int size, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_volume || !d_output || size <= 0) 
        { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::AD ind;
        ind.calculate(d_high, d_low, d_close, d_volume, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_adosc_d(const float* d_high, const float* d_low, const float* d_close,
                      const float* d_volume, float* d_output, int size,
                      int fastPeriod, int slowPeriod, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_volume || !d_output || size <= 0) 
        { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::ADOSC ind(fastPeriod, slowPeriod);
        ind.calculate(d_high, d_low, d_close, d_volume, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_obv_d(const float* d_close, const float* d_volume, float* d_output,
                    int size, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_close || !d_volume || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::OBV ind;
        ind.calculate(d_close, d_volume, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_mfi_d(const float* d_high, const float* d_low, const float* d_close,
                    const float* d_volume, float* d_output, int size, int period,
                    cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_volume || !d_output || size <= 0) 
        { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::MFI ind(period);
        ind.calculate(d_high, d_low, d_close, d_volume, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

// =============================================================================
// Statistical
// =============================================================================

ctStatus_t ct_stddev_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::StdDev ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_var_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::VAR ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_sum_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::SUM ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_max_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::MAX ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_min_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::MIN ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

// =============================================================================
// Price Transform
// =============================================================================

ctStatus_t ct_avgprice_d(const float* d_open, const float* d_high, const float* d_low,
                         const float* d_close, float* d_output, int size, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_open || !d_high || !d_low || !d_close || !d_output || size <= 0) 
        { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::AvgPrice ind;
        ind.calculate(d_open, d_high, d_low, d_close, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_medprice_d(const float* d_high, const float* d_low, float* d_output,
                         int size, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::MedPrice ind;
        ind.calculate(d_high, d_low, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_typprice_d(const float* d_high, const float* d_low, const float* d_close,
                         float* d_output, int size, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::TypPrice ind;
        ind.calculate(d_high, d_low, d_close, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_wclprice_d(const float* d_high, const float* d_low, const float* d_close,
                         float* d_output, int size, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_close || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::WclPrice ind;
        ind.calculate(d_high, d_low, d_close, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

// =============================================================================
// Linear Regression
// =============================================================================

ctStatus_t ct_linearreg_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::LINEARREG ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_linearreg_slope_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::LINEARREG_SLOPE ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_linearreg_intercept_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::LINEARREG_INTERCEPT ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_linearreg_angle_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::LINEARREG_ANGLE ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

ctStatus_t ct_tsf_d(const float* d_input, float* d_output, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_input || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::TSF ind(period);
        ind.calculate(d_input, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

// =============================================================================
// Aroon
// =============================================================================

ctStatus_t ct_aroon_d(const float* d_high, const float* d_low, float* d_aroonDown,
                      float* d_aroonUp, int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_aroonDown || !d_aroonUp || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    float* tmp = nullptr;
    const size_t bytes = static_cast<size_t>(size) * 3 * sizeof(float);
    if (cudaMalloc(&tmp, bytes) != cudaSuccess) return CT_STATUS_ALLOC_FAILED;
    auto cleanup = [&]() { cudaFree(tmp); };
    try {
        tacuda::Aroon ind(period, period);
        ind.calculate(d_high, d_low, tmp, size, stream);
        cudaError_t e1 = cudaMemcpyAsync(d_aroonUp,   tmp,            size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaError_t e2 = cudaMemcpyAsync(d_aroonDown, tmp + size,     size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cleanup();
        if (e1 != cudaSuccess || e2 != cudaSuccess) return CT_STATUS_COPY_FAILED;
        return CT_STATUS_SUCCESS;
    } catch (...) {
        cleanup();
        return CT_STATUS_KERNEL_FAILED;
    }
}

ctStatus_t ct_aroonosc_d(const float* d_high, const float* d_low, float* d_output,
                         int size, int period, cudaStream_t stream) {
    tacuda::detail::clear_last_error();
    if (!d_high || !d_low || !d_output || size <= 0) { tacuda::detail::set_last_error("null pointer or invalid size"); return CT_STATUS_INVALID_ARGUMENT; }
    try {
        tacuda::AroonOscillator ind(period);
        ind.calculate(d_high, d_low, d_output, size, stream);
        return CT_STATUS_SUCCESS;
  } catch (...) {
    return classify_exception();
  }
}

// Generated ct_*_d wrappers (scripts/generate_batch_api.py).
#include "api_device_generated.inc"

} // extern "C"

