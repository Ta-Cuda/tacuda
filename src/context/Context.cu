/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <tacuda/Context.h>
#include <utils/CudaUtils.h>
#include <utils/DeviceBufferPool.h>
#include <utils/Precision.h>

// Include indicators
#include <indicators/SMA.h>
#include <indicators/EMA.h>
#include <indicators/WMA.h>
#include <indicators/DEMA.h>
#include <indicators/TEMA.h>
#include <indicators/KAMA.h>
#include <indicators/T3.h>
#include <indicators/TRIMA.h>
#include <indicators/RSI.h>
#include <indicators/Momentum.h>
#include <indicators/ROC.h>
#include <indicators/ROCP.h>
#include <indicators/ROCR.h>
#include <indicators/CMO.h>
#include <indicators/WILLR.h>
#include <indicators/CCI.h>
#include <indicators/ATR.h>
#include <indicators/NATR.h>
#include <indicators/TRANGE.h>
#include <indicators/StdDev.h>
#include <indicators/BBANDS.h>
#include <indicators/ADX.h>
#include <indicators/ADXR.h>
#include <indicators/DX.h>
#include <indicators/PlusDI.h>
#include <indicators/MinusDI.h>
#include <indicators/Aroon.h>
#include <indicators/AroonOscillator.h>
#include <indicators/MACDEXT.h>
#include <indicators/Stochastic.h>
#include <indicators/StochasticFast.h>
#include <indicators/StochRSI.h>
#include <indicators/OBV.h>
#include <indicators/AD.h>
#include <indicators/ADOSC.h>
#include <indicators/MFI.h>
#include <indicators/TypPrice.h>
#include <indicators/MedPrice.h>
#include <indicators/WclPrice.h>
#include <indicators/AvgPrice.h>
#include <indicators/MIN.h>
#include <indicators/MAX.h>
#include <indicators/SUM.h>

#include <stdexcept>

namespace tacuda {

namespace {
size_t dtype_bytes(ctDataType_t dtype) {
    const size_t elem = precision::element_size(dtype);
    if (elem == 0) {
        throw std::invalid_argument("unknown ctDataType_t");
    }
    return elem;
}
}  // namespace

// ============================================================================
// DeviceBuffer Implementation
// ============================================================================

DeviceBuffer::DeviceBuffer(size_t size, cudaStream_t stream,
                           std::shared_ptr<void> streamGuard)
    : owner_(nullptr), streamGuard_(streamGuard), data_(nullptr), size_(size) {
    if (size_ > 0) {
        float* base = static_cast<float*>(
            DeviceBufferPool::instance().acquire(size_ * sizeof(float)));
        // Stream-aware release: the buffer re-enters the pool only after all
        // work already enqueued on the owning stream has finished. The guard
        // keeps that stream alive until this deleter has run, so a buffer
        // may safely outlive the Context that created it.
        owner_ = std::shared_ptr<float>(
            base, [stream, guard = std::move(streamGuard)](float* p) {
                if (p) DeviceBufferPool::instance().releaseAsync(p, stream);
            });
        data_ = base;
    }
}

DeviceBuffer::DeviceBuffer(const float* host_data, size_t size, cudaStream_t stream,
                           std::shared_ptr<void> streamGuard)
    : owner_(nullptr), streamGuard_(streamGuard), data_(nullptr), size_(size) {
    if (size_ > 0) {
        float* base = static_cast<float*>(
            DeviceBufferPool::instance().acquire(size_ * sizeof(float)));
        owner_ = std::shared_ptr<float>(
            base, [stream, guard = std::move(streamGuard)](float* p) {
                if (p) DeviceBufferPool::instance().releaseAsync(p, stream);
            });
        data_ = base;

        CUDA_CHECK(cudaMemcpyAsync(data_, host_data, size_ * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
    }
}

DeviceBuffer::~DeviceBuffer() = default;

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept = default;

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept = default;

void DeviceBuffer::upload(const float* host_data, cudaStream_t stream) {
    if (size_ > 0 && data_) {
        CUDA_CHECK(cudaMemcpyAsync(data_, host_data, size_ * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
    }
}

void DeviceBuffer::download(float* host_data, cudaStream_t stream) const {
    if (size_ > 0 && data_) {
        CUDA_CHECK(cudaMemcpyAsync(host_data, data_, size_ * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    }
}

DeviceBuffer DeviceBuffer::view(size_t offset, size_t count) const {
    if (offset + count > size_) {
        throw std::out_of_range("DeviceBuffer::view out of range");
    }

    DeviceBuffer result;
    result.owner_ = owner_;
    result.streamGuard_ = streamGuard_;
    result.data_ = data_ + offset;
    result.size_ = count;
    return result;
}

// ============================================================================
// DeviceBufferBase Implementation
// ============================================================================

DeviceBufferBase::DeviceBufferBase(size_t size, ctDataType_t dtype, cudaStream_t stream,
                                   std::shared_ptr<void> streamGuard)
    : owner_(nullptr), streamGuard_(streamGuard), data_(nullptr), size_(size),
      dtype_(dtype), owned_(true) {
    if (size_ == 0) return;
    const size_t bytes = size_ * dtype_bytes(dtype_);
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    // Defer the free until work enqueued on the owning stream has finished.
    // The guard keeps that stream alive until this deleter runs, so the
    // buffer may safely outlive the Context that created it (same contract
    // as DeviceBuffer).
    owner_ = std::shared_ptr<void>(
        ptr, [stream, guard = std::move(streamGuard)](void* p) {
            if (p) {
                cudaEvent_t ev = nullptr;
                if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) == cudaSuccess &&
                    cudaEventRecord(ev, stream) == cudaSuccess) {
                    cudaEventSynchronize(ev);
                    cudaEventDestroy(ev);
                } else {
                    if (ev) cudaEventDestroy(ev);
                    cudaStreamSynchronize(stream);
                }
                cudaFree(p);
            }
        });
    data_ = ptr;
}

DeviceBufferBase::DeviceBufferBase(const void* host_data, size_t size, ctDataType_t dtype,
                                   cudaStream_t stream, std::shared_ptr<void> streamGuard)
    : DeviceBufferBase(size, dtype, stream, std::move(streamGuard)) {
    upload(host_data, stream);
}

DeviceBufferBase::~DeviceBufferBase() = default;
DeviceBufferBase::DeviceBufferBase(DeviceBufferBase&& other) noexcept = default;
DeviceBufferBase& DeviceBufferBase::operator=(DeviceBufferBase&& other) noexcept = default;

void DeviceBufferBase::upload(const void* host_data, cudaStream_t stream) {
    if (size_ == 0 || !data_ || !host_data) return;
    const size_t bytes = size_ * dtype_bytes(dtype_);
    CUDA_CHECK(cudaMemcpyAsync(data_, host_data, bytes, cudaMemcpyHostToDevice, stream));
}

void DeviceBufferBase::download(void* host_data, cudaStream_t stream) const {
    if (size_ == 0 || !data_ || !host_data) return;
    const size_t bytes = size_ * dtype_bytes(dtype_);
    CUDA_CHECK(cudaMemcpyAsync(host_data, data_, bytes, cudaMemcpyDeviceToHost, stream));
}

DeviceBufferBase DeviceBufferBase::view(size_t offset, size_t count) const {
    if (offset + count > size_) {
        throw std::out_of_range("DeviceBufferBase::view out of range");
    }
    DeviceBufferBase out;
    out.owner_ = owner_;
    out.streamGuard_ = streamGuard_;
    out.size_ = count;
    out.dtype_ = dtype_;
    out.owned_ = false;
    out.data_ = static_cast<void*>(static_cast<unsigned char*>(data_) + offset * dtype_bytes(dtype_));
    return out;
}

// ============================================================================
// Context Implementation
// ============================================================================

Context::Context(size_t default_size)
    : stream_(0), default_size_(default_size) {
    CUDA_CHECK(cudaStreamCreate(&stream_));
    // Control block: the stream is synchronized and destroyed only after the
    // Context AND every DeviceBuffer created from it dropped their guard.
    streamOwner_ = std::shared_ptr<void>(
        static_cast<void*>(stream_), [](void* s) {
            cudaStream_t cs = static_cast<cudaStream_t>(s);
            if (cs) {
                (void)cudaStreamSynchronize(cs);
                (void)cudaStreamDestroy(cs);
            }
        });
}

Context::~Context() = default;

Context::Context(Context&& other) noexcept
    : stream_(other.stream_),
      streamOwner_(std::move(other.streamOwner_)),
      default_size_(other.default_size_),
      storageDType_(other.storageDType_),
      computeDType_(other.computeDType_) {
    other.stream_ = nullptr;
}

Context& Context::operator=(Context&& other) noexcept {
    if (this != &other) {
        stream_ = other.stream_;
        streamOwner_ = std::move(other.streamOwner_);
        default_size_ = other.default_size_;
        storageDType_ = other.storageDType_;
        computeDType_ = other.computeDType_;
        other.stream_ = nullptr;
    }
    return *this;
}

namespace {
// Multi-input indicators read every input at the same indices; unequal
// buffers would make kernels read out of bounds on the GPU.
void require_same_size(std::initializer_list<size_t> sizes, const char* what) {
    const size_t* it = sizes.begin();
    for (const size_t* p = it; p != sizes.end(); ++p) {
        if (*p != *it) {
            throw std::invalid_argument(std::string(what) +
                                        ": input buffers must have equal sizes");
        }
    }
}
} // namespace

void Context::require_own_buffer(const DeviceBuffer& buf, const char* what) const {
    if (buf.stream_guard() && buf.stream_guard() != streamOwner_) {
        throw std::invalid_argument(
            std::string(what) +
            ": buffer belongs to a different Context (its release is ordered "
            "on that Context's stream); use the owning Context or re-upload");
    }
}

DeviceBuffer Context::upload(const float* host_data, size_t size) {
    return DeviceBuffer(host_data, size, stream_, streamOwner_);
}

DeviceBufferBase Context::upload(const void* host_data, size_t size, ctDataType_t dtype) {
    return DeviceBufferBase(host_data, size, dtype, stream_, streamOwner_);
}

DeviceOHLCV Context::uploadOHLCV(const float* open, const float* high,
                                  const float* low, const float* close,
                                  const float* volume, size_t size) {
    DeviceOHLCV ohlcv;
    ohlcv.size = size;
    ohlcv.open = DeviceBuffer(open, size, stream_, streamOwner_);
    ohlcv.high = DeviceBuffer(high, size, stream_, streamOwner_);
    ohlcv.low = DeviceBuffer(low, size, stream_, streamOwner_);
    ohlcv.close = DeviceBuffer(close, size, stream_, streamOwner_);
    if (volume) {
        ohlcv.volume = DeviceBuffer(volume, size, stream_, streamOwner_);
    }
    return ohlcv;
}

void Context::download(const DeviceBuffer& buffer, float* host_data) {
    require_own_buffer(buffer, "Context::download");
    buffer.download(host_data, stream_);
    synchronize();  // Ensure data is ready
}

void Context::download(const DeviceBufferBase& buffer, void* host_data) {
    buffer.download(host_data, stream_);
    synchronize();
}

DeviceBuffer Context::allocate(size_t size) {
    return DeviceBuffer(size, stream_, streamOwner_);
}

DeviceBufferBase Context::allocate(size_t size, ctDataType_t dtype) {
    return DeviceBufferBase(size, dtype, stream_, streamOwner_);
}

void Context::setPrecision(ctDataType_t storageType, ctDataType_t computeType) {
    if (!precision::is_supported_compute_pair(storageType, computeType)) {
        throw std::invalid_argument("Unsupported precision pair");
    }
    storageDType_ = storageType;
    computeDType_ = computeType;
}

void Context::synchronize() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// ============================================================================
// Moving Averages
// ============================================================================

DeviceBuffer Context::sma(const DeviceBuffer& input, int period) {
    require_own_buffer(input, "Context::sma");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    SMA ind(period);
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

DeviceBuffer Context::ema(const DeviceBuffer& input, int period) {
    require_own_buffer(input, "Context::ema");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    EMA ind(period);
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

DeviceBuffer Context::wma(const DeviceBuffer& input, int period) {
    require_own_buffer(input, "Context::wma");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    WMA ind(period);
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

DeviceBuffer Context::dema(const DeviceBuffer& input, int period) {
    require_own_buffer(input, "Context::dema");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    DEMA ind(period);
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

DeviceBuffer Context::tema(const DeviceBuffer& input, int period) {
    require_own_buffer(input, "Context::tema");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    TEMA ind(period);
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

DeviceBuffer Context::kama(const DeviceBuffer& input, int period) {
    require_own_buffer(input, "Context::kama");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    KAMA ind(period, 2, 30);  // Default fast=2, slow=30 like TA-Lib
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

DeviceBuffer Context::t3(const DeviceBuffer& input, int period, double vfactor) {
    require_own_buffer(input, "Context::t3");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    T3 ind(period, vfactor);
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

DeviceBuffer Context::trima(const DeviceBuffer& input, int period) {
    require_own_buffer(input, "Context::trima");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    TRIMA ind(period);
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

// ============================================================================
// Momentum Indicators
// ============================================================================

DeviceBuffer Context::rsi(const DeviceBuffer& input, int period) {
    require_own_buffer(input, "Context::rsi");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    RSI ind(period);
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

DeviceBuffer Context::momentum(const DeviceBuffer& input, int period) {
    require_own_buffer(input, "Context::momentum");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    Momentum ind(period);
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

DeviceBuffer Context::roc(const DeviceBuffer& input, int period) {
    require_own_buffer(input, "Context::roc");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    ROC ind(period);
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

DeviceBuffer Context::rocp(const DeviceBuffer& input, int period) {
    require_own_buffer(input, "Context::rocp");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    ROCP ind(period);
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

DeviceBuffer Context::rocr(const DeviceBuffer& input, int period) {
    require_own_buffer(input, "Context::rocr");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    ROCR ind(period);
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

DeviceBuffer Context::cmo(const DeviceBuffer& input, int period) {
    require_own_buffer(input, "Context::cmo");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    CMO ind(period);
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

DeviceBuffer Context::willr(const DeviceBuffer& high, const DeviceBuffer& low,
                             const DeviceBuffer& close, int period) {
    require_own_buffer(high, "Context::willr");
    require_own_buffer(low, "Context::willr");
    require_own_buffer(close, "Context::willr");
    require_same_size({high.size(), low.size(), close.size()}, "Context::willr");
    DeviceBuffer output(close.size(), stream_, streamOwner_);
    WILLR ind(period);
    ind.calculate(high.data(), low.data(), close.data(), output.data(), 
                  close.size(), stream_);
    return output;
}

DeviceBuffer Context::cci(const DeviceBuffer& high, const DeviceBuffer& low,
                           const DeviceBuffer& close, int period) {
    require_own_buffer(high, "Context::cci");
    require_own_buffer(low, "Context::cci");
    require_own_buffer(close, "Context::cci");
    require_same_size({high.size(), low.size(), close.size()}, "Context::cci");
    DeviceBuffer output(close.size(), stream_, streamOwner_);
    CCI ind(period);
    ind.calculate(high.data(), low.data(), close.data(), output.data(),
                  close.size(), stream_);
    return output;
}

// ============================================================================
// Volatility Indicators
// ============================================================================

DeviceBuffer Context::atr(const DeviceBuffer& high, const DeviceBuffer& low,
                           const DeviceBuffer& close, int period) {
    require_own_buffer(high, "Context::atr");
    require_own_buffer(low, "Context::atr");
    require_own_buffer(close, "Context::atr");
    require_same_size({high.size(), low.size(), close.size()}, "Context::atr");
    DeviceBuffer output(close.size(), stream_, streamOwner_);
    ATR ind(period);
    ind.calculate(high.data(), low.data(), close.data(), output.data(),
                  close.size(), stream_);
    return output;
}

DeviceBuffer Context::natr(const DeviceBuffer& high, const DeviceBuffer& low,
                            const DeviceBuffer& close, int period) {
    require_own_buffer(high, "Context::natr");
    require_own_buffer(low, "Context::natr");
    require_own_buffer(close, "Context::natr");
    require_same_size({high.size(), low.size(), close.size()}, "Context::natr");
    DeviceBuffer output(close.size(), stream_, streamOwner_);
    NATR ind(period);
    ind.calculate(high.data(), low.data(), close.data(), output.data(),
                  close.size(), stream_);
    return output;
}

DeviceBuffer Context::trange(const DeviceBuffer& high, const DeviceBuffer& low,
                              const DeviceBuffer& close) {
    require_own_buffer(high, "Context::trange");
    require_own_buffer(low, "Context::trange");
    require_own_buffer(close, "Context::trange");
    require_same_size({high.size(), low.size(), close.size()}, "Context::trange");
    DeviceBuffer output(close.size(), stream_, streamOwner_);
    TRANGE ind;
    ind.calculate(high.data(), low.data(), close.data(), output.data(),
                  close.size(), stream_);
    return output;
}

DeviceBuffer Context::stddev(const DeviceBuffer& input, int period, double /*nbdev*/) {
    require_own_buffer(input, "Context::stddev");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    StdDev ind(period);  // nbdev is applied separately if needed
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

Context::BollingerResult Context::bbands(const DeviceBuffer& input, int period,
                                          double nbdevup, double nbdevdn, int /*matype*/) {
    require_own_buffer(input, "Context::bbands");
    BollingerResult result;
    size_t size = input.size();
    
    // Allocate output buffers
    DeviceBuffer combined(size * 3, stream_, streamOwner_);
    
    BBANDS ind(period, static_cast<float>(nbdevup), static_cast<float>(nbdevdn));
    ind.calculate(input.data(), combined.data(), size, stream_);
    
    // Split into separate buffers
    result.upper = combined.view(0, size);
    result.middle = combined.view(size, size);
    result.lower = combined.view(size * 2, size);
    
    return result;
}

// ============================================================================
// Trend Indicators
// ============================================================================

DeviceBuffer Context::adx(const DeviceBuffer& high, const DeviceBuffer& low,
                           const DeviceBuffer& close, int period) {
    require_own_buffer(high, "Context::adx");
    require_own_buffer(low, "Context::adx");
    require_own_buffer(close, "Context::adx");
    require_same_size({high.size(), low.size(), close.size()}, "Context::adx");
    DeviceBuffer output(close.size(), stream_, streamOwner_);
    ADX ind(period);
    ind.calculate(high.data(), low.data(), close.data(), output.data(),
                  close.size(), stream_);
    return output;
}

DeviceBuffer Context::adxr(const DeviceBuffer& high, const DeviceBuffer& low,
                            const DeviceBuffer& close, int period) {
    require_own_buffer(high, "Context::adxr");
    require_own_buffer(low, "Context::adxr");
    require_own_buffer(close, "Context::adxr");
    require_same_size({high.size(), low.size(), close.size()}, "Context::adxr");
    DeviceBuffer output(close.size(), stream_, streamOwner_);
    ADXR ind(period);
    ind.calculate(high.data(), low.data(), close.data(), output.data(),
                  close.size(), stream_);
    return output;
}

DeviceBuffer Context::dx(const DeviceBuffer& high, const DeviceBuffer& low,
                          const DeviceBuffer& close, int period) {
    require_own_buffer(high, "Context::dx");
    require_own_buffer(low, "Context::dx");
    require_own_buffer(close, "Context::dx");
    require_same_size({high.size(), low.size(), close.size()}, "Context::dx");
    DeviceBuffer output(close.size(), stream_, streamOwner_);
    DX ind(period);
    ind.calculate(high.data(), low.data(), close.data(), output.data(),
                  close.size(), stream_);
    return output;
}

DeviceBuffer Context::plusDI(const DeviceBuffer& high, const DeviceBuffer& low,
                              const DeviceBuffer& close, int period) {
    require_own_buffer(high, "Context::plusDI");
    require_own_buffer(low, "Context::plusDI");
    require_own_buffer(close, "Context::plusDI");
    require_same_size({high.size(), low.size(), close.size()}, "Context::plusDI");
    DeviceBuffer output(close.size(), stream_, streamOwner_);
    PlusDI ind(period);
    ind.calculate(high.data(), low.data(), close.data(), output.data(),
                  close.size(), stream_);
    return output;
}

DeviceBuffer Context::minusDI(const DeviceBuffer& high, const DeviceBuffer& low,
                               const DeviceBuffer& close, int period) {
    require_own_buffer(high, "Context::minusDI");
    require_own_buffer(low, "Context::minusDI");
    require_own_buffer(close, "Context::minusDI");
    require_same_size({high.size(), low.size(), close.size()}, "Context::minusDI");
    DeviceBuffer output(close.size(), stream_, streamOwner_);
    MinusDI ind(period);
    ind.calculate(high.data(), low.data(), close.data(), output.data(),
                  close.size(), stream_);
    return output;
}

Context::AroonResult Context::aroon(const DeviceBuffer& high, const DeviceBuffer& low, int period) {
    require_own_buffer(high, "Context::aroon");
    require_own_buffer(low, "Context::aroon");
    require_same_size({high.size(), low.size()}, "Context::aroon");
    AroonResult result;
    size_t size = high.size();
    
    DeviceBuffer combined(size * 2, stream_, streamOwner_);
    Aroon ind(period, period);  // Same period for up and down
    ind.calculate(high.data(), low.data(), combined.data(), size, stream_);
    
    result.up = combined.view(0, size);
    result.down = combined.view(size, size);
    
    return result;
}

DeviceBuffer Context::aroonOsc(const DeviceBuffer& high, const DeviceBuffer& low, int period) {
    require_own_buffer(high, "Context::aroonOsc");
    require_own_buffer(low, "Context::aroonOsc");
    require_same_size({high.size(), low.size()}, "Context::aroonOsc");
    DeviceBuffer output(high.size(), stream_, streamOwner_);
    AroonOscillator ind(period);
    ind.calculate(high.data(), low.data(), output.data(), high.size(), stream_);
    return output;
}

// ============================================================================
// MACD
// ============================================================================

Context::MACDResult Context::macd(const DeviceBuffer& input, int fastperiod,
                                   int slowperiod, int signalperiod) {
    require_own_buffer(input, "Context::macd");
    MACDResult result;
    size_t size = input.size();
    
    DeviceBuffer combined(size * 3, stream_, streamOwner_);
    MACDEXT ind(fastperiod, slowperiod, signalperiod, MAType::EMA);
    ind.calculate(input.data(), combined.data(), static_cast<int>(size), stream_);
    
    result.macd = combined.view(0, size);
    result.signal = combined.view(size, size);
    result.hist = combined.view(size * 2, size);
    
    return result;
}

// ============================================================================
// Stochastic
// ============================================================================

Context::StochResult Context::stoch(const DeviceBuffer& high, const DeviceBuffer& low,
                                     const DeviceBuffer& close, int fastK, int slowK, int slowD) {
    require_own_buffer(high, "Context::stoch");
    require_own_buffer(low, "Context::stoch");
    require_own_buffer(close, "Context::stoch");
    require_same_size({high.size(), low.size(), close.size()}, "Context::stoch");
    StochResult result;
    size_t size = close.size();
    
    DeviceBuffer combined(size * 2, stream_, streamOwner_);
    if (slowK != slowD) {
        throw std::invalid_argument("Context::stoch: slowK must equal slowD (only one slow period is supported)");
    }
    Stochastic ind(fastK, slowK);
    ind.calculate(high.data(), low.data(), close.data(), combined.data(), static_cast<int>(size), stream_);
    
    result.slowK = combined.view(0, size);
    result.slowD = combined.view(size, size);
    
    return result;
}

Context::StochFastResult Context::stochFast(const DeviceBuffer& high, const DeviceBuffer& low,
                                             const DeviceBuffer& close, int fastK, int fastD) {
    require_own_buffer(high, "Context::stochFast");
    require_own_buffer(low, "Context::stochFast");
    require_own_buffer(close, "Context::stochFast");
    require_same_size({high.size(), low.size(), close.size()}, "Context::stochFast");
    StochFastResult result;
    size_t size = close.size();
    
    DeviceBuffer combined(size * 2, stream_, streamOwner_);
    StochasticFast ind(fastK, fastD);
    ind.calculate(high.data(), low.data(), close.data(), combined.data(), size, stream_);
    
    result.fastK = combined.view(0, size);
    result.fastD = combined.view(size, size);
    
    return result;
}

DeviceBuffer Context::stochRSI(const DeviceBuffer& input, int period, int fastK, int fastD) {
    require_own_buffer(input, "Context::stochRSI");
    DeviceBuffer output(input.size() * 2, stream_, streamOwner_);
    StochRSI ind(period, fastK, fastD);
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

// ============================================================================
// Volume Indicators
// ============================================================================

DeviceBuffer Context::obv(const DeviceBuffer& close, const DeviceBuffer& volume) {
    require_own_buffer(close, "Context::obv");
    require_own_buffer(volume, "Context::obv");
    require_same_size({close.size(), volume.size()}, "Context::obv");
    DeviceBuffer output(close.size(), stream_, streamOwner_);
    OBV ind;
    ind.calculate(close.data(), volume.data(), output.data(), close.size(), stream_);
    return output;
}

DeviceBuffer Context::ad(const DeviceBuffer& high, const DeviceBuffer& low,
                          const DeviceBuffer& close, const DeviceBuffer& volume) {
    require_own_buffer(high, "Context::ad");
    require_own_buffer(low, "Context::ad");
    require_own_buffer(close, "Context::ad");
    require_own_buffer(volume, "Context::ad");
    require_same_size({high.size(), low.size(), close.size(), volume.size()}, "Context::ad");
    DeviceBuffer output(close.size(), stream_, streamOwner_);
    AD ind;
    ind.calculate(high.data(), low.data(), close.data(), volume.data(),
                  output.data(), close.size(), stream_);
    return output;
}

DeviceBuffer Context::adosc(const DeviceBuffer& high, const DeviceBuffer& low,
                             const DeviceBuffer& close, const DeviceBuffer& volume,
                             int fastperiod, int slowperiod) {
    require_own_buffer(high, "Context::adosc");
    require_own_buffer(low, "Context::adosc");
    require_own_buffer(close, "Context::adosc");
    require_own_buffer(volume, "Context::adosc");
    require_same_size({high.size(), low.size(), close.size(), volume.size()}, "Context::adosc");
    DeviceBuffer output(close.size(), stream_, streamOwner_);
    ADOSC ind(fastperiod, slowperiod);
    ind.calculate(high.data(), low.data(), close.data(), volume.data(),
                  output.data(), close.size(), stream_);
    return output;
}

DeviceBuffer Context::mfi(const DeviceBuffer& high, const DeviceBuffer& low,
                           const DeviceBuffer& close, const DeviceBuffer& volume, int period) {
    require_own_buffer(high, "Context::mfi");
    require_own_buffer(low, "Context::mfi");
    require_own_buffer(close, "Context::mfi");
    require_own_buffer(volume, "Context::mfi");
    require_same_size({high.size(), low.size(), close.size(), volume.size()}, "Context::mfi");
    DeviceBuffer output(close.size(), stream_, streamOwner_);
    MFI ind(period);
    ind.calculate(high.data(), low.data(), close.data(), volume.data(),
                  output.data(), close.size(), stream_);
    return output;
}

// ============================================================================
// Price Transforms
// ============================================================================

DeviceBuffer Context::typPrice(const DeviceBuffer& high, const DeviceBuffer& low,
                                const DeviceBuffer& close) {
    require_own_buffer(high, "Context::typPrice");
    require_own_buffer(low, "Context::typPrice");
    require_own_buffer(close, "Context::typPrice");
    require_same_size({high.size(), low.size(), close.size()}, "Context::typPrice");
    DeviceBuffer output(close.size(), stream_, streamOwner_);
    TypPrice ind;
    ind.calculate(high.data(), low.data(), close.data(), output.data(),
                  close.size(), stream_);
    return output;
}

DeviceBuffer Context::medPrice(const DeviceBuffer& high, const DeviceBuffer& low) {
    require_own_buffer(high, "Context::medPrice");
    require_own_buffer(low, "Context::medPrice");
    require_same_size({high.size(), low.size()}, "Context::medPrice");
    DeviceBuffer output(high.size(), stream_, streamOwner_);
    MedPrice ind;
    ind.calculate(high.data(), low.data(), output.data(), high.size(), stream_);
    return output;
}

DeviceBuffer Context::wclPrice(const DeviceBuffer& high, const DeviceBuffer& low,
                                const DeviceBuffer& close) {
    require_own_buffer(high, "Context::wclPrice");
    require_own_buffer(low, "Context::wclPrice");
    require_own_buffer(close, "Context::wclPrice");
    require_same_size({high.size(), low.size(), close.size()}, "Context::wclPrice");
    DeviceBuffer output(close.size(), stream_, streamOwner_);
    WclPrice ind;
    ind.calculate(high.data(), low.data(), close.data(), output.data(),
                  close.size(), stream_);
    return output;
}

DeviceBuffer Context::avgPrice(const DeviceBuffer& open, const DeviceBuffer& high,
                                const DeviceBuffer& low, const DeviceBuffer& close) {
    require_own_buffer(open, "Context::avgPrice");
    require_own_buffer(high, "Context::avgPrice");
    require_own_buffer(low, "Context::avgPrice");
    require_own_buffer(close, "Context::avgPrice");
    require_same_size({open.size(), high.size(), low.size(), close.size()}, "Context::avgPrice");
    DeviceBuffer output(close.size(), stream_, streamOwner_);
    AvgPrice ind;
    ind.calculate(open.data(), high.data(), low.data(), close.data(),
                  output.data(), close.size(), stream_);
    return output;
}

// ============================================================================
// Math Operations - CUDA Kernels
// ============================================================================

__global__ void addKernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void subtractKernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

__global__ void multiplyKernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

__global__ void divideKernel(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (b[i] != 0.0f) ? a[i] / b[i] : 0.0f;
}

__global__ void scaleKernel(const float* in, float factor, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * factor;
}

__global__ void offsetKernel(const float* in, float value, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] + value;
}

DeviceBuffer Context::add(const DeviceBuffer& a, const DeviceBuffer& b) {
    require_own_buffer(a, "Context::add");
    require_own_buffer(b, "Context::add");
    require_same_size({a.size(), b.size()}, "Context::add");
    size_t size = std::min(a.size(), b.size());
    DeviceBuffer output(size, stream_, streamOwner_);
    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    addKernel<<<numBlocks, blockSize, 0, stream_>>>(a.data(), b.data(), output.data(), size);
    
    return output;
}

DeviceBuffer Context::subtract(const DeviceBuffer& a, const DeviceBuffer& b) {
    require_own_buffer(a, "Context::subtract");
    require_own_buffer(b, "Context::subtract");
    require_same_size({a.size(), b.size()}, "Context::subtract");
    size_t size = std::min(a.size(), b.size());
    DeviceBuffer output(size, stream_, streamOwner_);
    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    subtractKernel<<<numBlocks, blockSize, 0, stream_>>>(a.data(), b.data(), output.data(), size);
    
    return output;
}

DeviceBuffer Context::multiply(const DeviceBuffer& a, const DeviceBuffer& b) {
    require_own_buffer(a, "Context::multiply");
    require_own_buffer(b, "Context::multiply");
    require_same_size({a.size(), b.size()}, "Context::multiply");
    size_t size = std::min(a.size(), b.size());
    DeviceBuffer output(size, stream_, streamOwner_);
    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    multiplyKernel<<<numBlocks, blockSize, 0, stream_>>>(a.data(), b.data(), output.data(), size);
    
    return output;
}

DeviceBuffer Context::divide(const DeviceBuffer& a, const DeviceBuffer& b) {
    require_own_buffer(a, "Context::divide");
    require_own_buffer(b, "Context::divide");
    require_same_size({a.size(), b.size()}, "Context::divide");
    size_t size = std::min(a.size(), b.size());
    DeviceBuffer output(size, stream_, streamOwner_);
    
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    divideKernel<<<numBlocks, blockSize, 0, stream_>>>(a.data(), b.data(), output.data(), size);
    
    return output;
}

DeviceBuffer Context::scale(const DeviceBuffer& input, float factor) {
    require_own_buffer(input, "Context::scale");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    
    int blockSize = 256;
    int numBlocks = (input.size() + blockSize - 1) / blockSize;
    scaleKernel<<<numBlocks, blockSize, 0, stream_>>>(input.data(), factor, output.data(), input.size());
    
    return output;
}

DeviceBuffer Context::offset(const DeviceBuffer& input, float value) {
    require_own_buffer(input, "Context::offset");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    
    int blockSize = 256;
    int numBlocks = (input.size() + blockSize - 1) / blockSize;
    offsetKernel<<<numBlocks, blockSize, 0, stream_>>>(input.data(), value, output.data(), input.size());
    
    return output;
}

// ============================================================================
// Utility
// ============================================================================

DeviceBuffer Context::min(const DeviceBuffer& input, int period) {
    require_own_buffer(input, "Context::min");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    MIN ind(period);
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

DeviceBuffer Context::max(const DeviceBuffer& input, int period) {
    require_own_buffer(input, "Context::max");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    MAX ind(period);
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

DeviceBuffer Context::sum(const DeviceBuffer& input, int period) {
    require_own_buffer(input, "Context::sum");
    DeviceBuffer output(input.size(), stream_, streamOwner_);
    SUM ind(period);
    ind.calculate(input.data(), output.data(), input.size(), stream_);
    return output;
}

} // namespace tacuda

