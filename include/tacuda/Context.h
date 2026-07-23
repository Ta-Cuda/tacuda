/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_CONTEXT_H
#define TACUDA_CONTEXT_H

#include <cuda_runtime.h>
#include <tacuda.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <stdexcept>

// Export/import macro for shared library (Windows DLL)
#ifdef _WIN32
  #ifdef tacuda_EXPORTS
    #define TACUDA_API __declspec(dllexport)
  #else
    #define TACUDA_API __declspec(dllimport)
  #endif
#else
  #define TACUDA_API __attribute__((visibility("default")))
#endif

namespace tacuda {

// Forward declarations
class DeviceBuffer;
class DeviceBufferBase;
class Context;

// ============================================================================
// DeviceBuffer - RAII wrapper for GPU memory
// ============================================================================
class TACUDA_API DeviceBuffer {
public:
    DeviceBuffer() : owner_(nullptr), data_(nullptr), size_(0) {}
    
    // `streamGuard` (optional) keeps the owning stream alive until the last
    // buffer using it is released — Context passes its stream control block
    // so a DeviceBuffer may safely outlive the Context that created it.
    // When passing a raw user-owned stream without a guard, the caller must
    // keep that stream alive for the buffer's lifetime.
    DeviceBuffer(size_t size, cudaStream_t stream = 0,
                 std::shared_ptr<void> streamGuard = nullptr);
    DeviceBuffer(const float* host_data, size_t size, cudaStream_t stream = 0,
                 std::shared_ptr<void> streamGuard = nullptr);
    
    // View constructor (non-owning)
    DeviceBuffer(float* device_ptr, size_t size) 
        : owner_(nullptr), data_(device_ptr), size_(size) {}
    
    ~DeviceBuffer();
    
    // Move semantics
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
    
    // No copy
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // Accessors
    float* data() { return data_; }
    const float* data() const { return data_; }
    size_t size() const { return size_; }
    bool empty() const { return data_ == nullptr || size_ == 0; }
    
    // Upload/Download
    void upload(const float* host_data, cudaStream_t stream = 0);
    void download(float* host_data, cudaStream_t stream = 0) const;
    
    // Create a view (slice) of this buffer
    DeviceBuffer view(size_t offset, size_t count) const;
    
    // Identity of the owning stream control block (nullptr for buffers on
    // raw user-managed streams). Context uses it to reject buffers created
    // by a different Context — mixing streams would release memory into the
    // pool while another Context's kernels may still be writing it.
    const std::shared_ptr<void>& stream_guard() const { return streamGuard_; }

private:
    // Shared owner for the underlying CUDA allocation. Views keep this alive.
    // The owner's deleter captures the stream guard, so the stream outlives
    // every allocation that was released against it.
    std::shared_ptr<float> owner_;
    std::shared_ptr<void> streamGuard_;
    float* data_;
    size_t size_;
};

// ============================================================================
// DeviceBufferBase - dtype-aware GPU memory wrapper
// ============================================================================
class TACUDA_API DeviceBufferBase {
public:
    DeviceBufferBase()
        : owner_(nullptr), data_(nullptr), size_(0), dtype_(CT_DTYPE_F32), owned_(false) {}

    // `streamGuard` keeps the owning stream alive until the buffer is freed
    // (see DeviceBuffer); Context passes its stream control block.
    DeviceBufferBase(size_t size, ctDataType_t dtype, cudaStream_t stream = 0,
                     std::shared_ptr<void> streamGuard = nullptr);
    DeviceBufferBase(const void* host_data, size_t size, ctDataType_t dtype,
                     cudaStream_t stream = 0,
                     std::shared_ptr<void> streamGuard = nullptr);
    DeviceBufferBase(void* device_ptr, size_t size, ctDataType_t dtype)
        : owner_(nullptr), data_(device_ptr), size_(size), dtype_(dtype), owned_(false) {}

    ~DeviceBufferBase();
    DeviceBufferBase(DeviceBufferBase&& other) noexcept;
    DeviceBufferBase& operator=(DeviceBufferBase&& other) noexcept;

    DeviceBufferBase(const DeviceBufferBase&) = delete;
    DeviceBufferBase& operator=(const DeviceBufferBase&) = delete;

    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }
    ctDataType_t dtype() const { return dtype_; }
    bool empty() const { return data_ == nullptr || size_ == 0; }

    void upload(const void* host_data, cudaStream_t stream = 0);
    void download(void* host_data, cudaStream_t stream = 0) const;
    DeviceBufferBase view(size_t offset, size_t count) const;

    // See DeviceBuffer::stream_guard().
    const std::shared_ptr<void>& stream_guard() const { return streamGuard_; }

private:
    std::shared_ptr<void> owner_;
    std::shared_ptr<void> streamGuard_;
    void* data_;
    size_t size_;
    ctDataType_t dtype_;
    bool owned_;
};

// ============================================================================
// OHLCV - Structure for market data on GPU
// ============================================================================
struct TACUDA_API DeviceOHLCV {
    DeviceBuffer open;
    DeviceBuffer high;
    DeviceBuffer low;
    DeviceBuffer close;
    DeviceBuffer volume;
    size_t size;
    
    DeviceOHLCV() : size(0) {}
    DeviceOHLCV(DeviceOHLCV&&) = default;
    DeviceOHLCV& operator=(DeviceOHLCV&&) = default;
};

// ============================================================================
// Context - Main class for GPU indicator pipeline
// ============================================================================
class TACUDA_API Context {
public:
    explicit Context(size_t default_size = 0);
    ~Context();
    
    // No copy
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    
    // Move transfers stream ownership (the source is left stream-less, so
    // the CUDA stream is destroyed exactly once).
    Context(Context&& other) noexcept;
    Context& operator=(Context&& other) noexcept;
    
    // ========================================================================
    // Data Management
    // ========================================================================
    
    // Upload single array
    DeviceBuffer upload(const float* host_data, size_t size);
    
    // Upload OHLCV data
    DeviceOHLCV uploadOHLCV(const float* open, const float* high,
                            const float* low, const float* close,
                            const float* volume, size_t size);
    
    // Download to host
    void download(const DeviceBuffer& buffer, float* host_data);
    
    // Allocate empty buffer
    DeviceBuffer allocate(size_t size);
    DeviceBufferBase allocate(size_t size, ctDataType_t dtype);

    // Upload/download typed buffers
    DeviceBufferBase upload(const void* host_data, size_t size, ctDataType_t dtype);
    void download(const DeviceBufferBase& buffer, void* host_data);

    // Runtime precision policy
    void setPrecision(ctDataType_t storageType, ctDataType_t computeType);
    ctDataType_t storageDType() const { return storageDType_; }
    ctDataType_t computeDType() const { return computeDType_; }
    
    // ========================================================================
    // Moving Averages
    // ========================================================================
    DeviceBuffer sma(const DeviceBuffer& input, int period);
    DeviceBuffer ema(const DeviceBuffer& input, int period);
    DeviceBuffer wma(const DeviceBuffer& input, int period);
    DeviceBuffer dema(const DeviceBuffer& input, int period);
    DeviceBuffer tema(const DeviceBuffer& input, int period);
    DeviceBuffer kama(const DeviceBuffer& input, int period);
    DeviceBuffer t3(const DeviceBuffer& input, int period, double vfactor = 0.7);
    DeviceBuffer trima(const DeviceBuffer& input, int period);
    
    // ========================================================================
    // Momentum Indicators
    // ========================================================================
    DeviceBuffer rsi(const DeviceBuffer& input, int period);
    DeviceBuffer momentum(const DeviceBuffer& input, int period);
    DeviceBuffer roc(const DeviceBuffer& input, int period);
    DeviceBuffer rocp(const DeviceBuffer& input, int period);
    DeviceBuffer rocr(const DeviceBuffer& input, int period);
    DeviceBuffer cmo(const DeviceBuffer& input, int period);
    DeviceBuffer willr(const DeviceBuffer& high, const DeviceBuffer& low, 
                       const DeviceBuffer& close, int period);
    DeviceBuffer cci(const DeviceBuffer& high, const DeviceBuffer& low,
                     const DeviceBuffer& close, int period);
    
    // ========================================================================
    // Volatility Indicators
    // ========================================================================
    DeviceBuffer atr(const DeviceBuffer& high, const DeviceBuffer& low,
                     const DeviceBuffer& close, int period);
    DeviceBuffer natr(const DeviceBuffer& high, const DeviceBuffer& low,
                      const DeviceBuffer& close, int period);
    DeviceBuffer trange(const DeviceBuffer& high, const DeviceBuffer& low,
                        const DeviceBuffer& close);
    DeviceBuffer stddev(const DeviceBuffer& input, int period, double nbdev = 1.0);
    
    // Bollinger Bands - returns upper, middle, lower
    struct BollingerResult {
        DeviceBuffer upper;
        DeviceBuffer middle;
        DeviceBuffer lower;
    };
    BollingerResult bbands(const DeviceBuffer& input, int period, 
                           double nbdevup = 2.0, double nbdevdn = 2.0, int matype = 0);
    
    // ========================================================================
    // Trend Indicators
    // ========================================================================
    DeviceBuffer adx(const DeviceBuffer& high, const DeviceBuffer& low,
                     const DeviceBuffer& close, int period);
    DeviceBuffer adxr(const DeviceBuffer& high, const DeviceBuffer& low,
                      const DeviceBuffer& close, int period);
    DeviceBuffer dx(const DeviceBuffer& high, const DeviceBuffer& low,
                    const DeviceBuffer& close, int period);
    DeviceBuffer plusDI(const DeviceBuffer& high, const DeviceBuffer& low,
                        const DeviceBuffer& close, int period);
    DeviceBuffer minusDI(const DeviceBuffer& high, const DeviceBuffer& low,
                         const DeviceBuffer& close, int period);
    
    // Aroon - returns up and down
    struct AroonResult {
        DeviceBuffer up;
        DeviceBuffer down;
    };
    AroonResult aroon(const DeviceBuffer& high, const DeviceBuffer& low, int period);
    DeviceBuffer aroonOsc(const DeviceBuffer& high, const DeviceBuffer& low, int period);
    
    // ========================================================================
    // MACD Family
    // ========================================================================
    struct MACDResult {
        DeviceBuffer macd;
        DeviceBuffer signal;
        DeviceBuffer hist;
    };
    MACDResult macd(const DeviceBuffer& input, int fastperiod = 12, 
                    int slowperiod = 26, int signalperiod = 9);
    
    // ========================================================================
    // Stochastic
    // ========================================================================
    struct StochResult {
        DeviceBuffer slowK;
        DeviceBuffer slowD;
    };
    StochResult stoch(const DeviceBuffer& high, const DeviceBuffer& low,
                      const DeviceBuffer& close, int fastK = 14, 
                      int slowK = 3, int slowD = 3);
    
    struct StochFastResult {
        DeviceBuffer fastK;
        DeviceBuffer fastD;
    };
    StochFastResult stochFast(const DeviceBuffer& high, const DeviceBuffer& low,
                              const DeviceBuffer& close, int fastK = 14, int fastD = 3);
    
    DeviceBuffer stochRSI(const DeviceBuffer& input, int period = 14,
                          int fastK = 5, int fastD = 3);
    
    // ========================================================================
    // Volume Indicators
    // ========================================================================
    DeviceBuffer obv(const DeviceBuffer& close, const DeviceBuffer& volume);
    DeviceBuffer ad(const DeviceBuffer& high, const DeviceBuffer& low,
                    const DeviceBuffer& close, const DeviceBuffer& volume);
    DeviceBuffer adosc(const DeviceBuffer& high, const DeviceBuffer& low,
                       const DeviceBuffer& close, const DeviceBuffer& volume,
                       int fastperiod = 3, int slowperiod = 10);
    DeviceBuffer mfi(const DeviceBuffer& high, const DeviceBuffer& low,
                     const DeviceBuffer& close, const DeviceBuffer& volume, int period);
    
    // ========================================================================
    // Price Transforms
    // ========================================================================
    DeviceBuffer typPrice(const DeviceBuffer& high, const DeviceBuffer& low,
                          const DeviceBuffer& close);
    DeviceBuffer medPrice(const DeviceBuffer& high, const DeviceBuffer& low);
    DeviceBuffer wclPrice(const DeviceBuffer& high, const DeviceBuffer& low,
                          const DeviceBuffer& close);
    DeviceBuffer avgPrice(const DeviceBuffer& open, const DeviceBuffer& high,
                          const DeviceBuffer& low, const DeviceBuffer& close);
    
    // ========================================================================
    // Math Operations (element-wise)
    // ========================================================================
    DeviceBuffer add(const DeviceBuffer& a, const DeviceBuffer& b);
    DeviceBuffer subtract(const DeviceBuffer& a, const DeviceBuffer& b);
    DeviceBuffer multiply(const DeviceBuffer& a, const DeviceBuffer& b);
    DeviceBuffer divide(const DeviceBuffer& a, const DeviceBuffer& b);
    DeviceBuffer scale(const DeviceBuffer& input, float factor);
    DeviceBuffer offset(const DeviceBuffer& input, float value);
    
    // ========================================================================
    // Utility
    // ========================================================================
    DeviceBuffer min(const DeviceBuffer& input, int period);
    DeviceBuffer max(const DeviceBuffer& input, int period);
    DeviceBuffer sum(const DeviceBuffer& input, int period);
    
    // Stream management
    cudaStream_t stream() const { return stream_; }
    void synchronize();
    
private:
    cudaStream_t stream_;
    // Control block owning the CUDA stream: destroyed (sync + destroy) only
    // when the Context AND every DeviceBuffer created from it are gone.
    std::shared_ptr<void> streamOwner_;
    size_t default_size_;
    ctDataType_t storageDType_ = CT_DTYPE_F32;
    ctDataType_t computeDType_ = CT_DTYPE_F32;
    
    // Helper to run indicator and return result buffer
    template<typename IndicatorT, typename... Args>
    DeviceBuffer runIndicator(const DeviceBuffer& input, Args&&... args);
    
    template<typename IndicatorT, typename... Args>
    DeviceBuffer runIndicatorMultiInput(size_t size, Args&&... args);

    // Rejects buffers created by a different Context: running kernels on
    // this stream over a foreign buffer would let its release fire before
    // this stream's work completes. Buffers on raw user-managed streams
    // (no guard) are the caller's responsibility and pass through.
    void require_own_buffer(const DeviceBuffer& buf, const char* what) const;
};

} // namespace tacuda

#endif // TACUDA_CONTEXT_H

