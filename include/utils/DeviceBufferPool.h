/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef DEVICE_BUFFER_POOL_H
#define DEVICE_BUFFER_POOL_H

#include <cstddef>
#include <cuda_runtime.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

// Process-wide CUDA memory pool.
//
// Lifetime states of an allocation:
//   in use   — handed out by acquire(); tracked in `allocs` only.
//   pending  — released with releaseAsync(); reusable once its CUDA event
//              fires (all work enqueued on the releasing stream finished).
//   cached   — in the per-(device,size) free list, ready for reuse.
//
// Buffers are cached per device: a pointer allocated on device X is only
// ever handed out again on device X. cleanup() frees cached/pending memory
// but never touches buffers currently in use; the destructor frees
// everything (process teardown).
class DeviceBufferPool {
public:
    static DeviceBufferPool& instance();

    void* acquire(size_t bytes);
    void release(void* ptr);
    // Stream-aware release: marks the buffer reusable only after all work
    // already enqueued on the provided stream has completed. Use this
    // whenever kernels or async copies touching the buffer may still be
    // in flight.
    void releaseAsync(void* ptr, cudaStream_t stream);

    // Frees all cached and pending device buffers. Buffers still in use
    // stay valid and return to the pool through release()/releaseAsync().
    void cleanup();

    // Sets the maximum amount of cached device memory in bytes. Returns the previous limit.
    size_t setMaxCacheBytes(size_t bytes);

    // Number of cudaMalloc calls performed by the pool.
    size_t allocationCount() const;

    ~DeviceBufferPool();

private:
    DeviceBufferPool() = default;
    DeviceBufferPool(const DeviceBufferPool&) = delete;
    DeviceBufferPool& operator=(const DeviceBufferPool&) = delete;

    struct AllocInfo {
        size_t size = 0;
        int device = 0;
    };

    struct PendingBuffer {
        void* ptr = nullptr;
        size_t size = 0;
        int device = 0;
        cudaEvent_t readyEvent = nullptr;
        bool cacheWhenReady = true;
    };

    using FreeKey = std::pair<int, size_t>; // (device, bytes)

    cudaEvent_t acquireEventLocked();
    void recycleEventLocked(cudaEvent_t ev);
    void drainPendingLocked(size_t sizeFilter = 0);
    void freeLocked(void* ptr); // cudaFree + erase from allocs

    std::map<FreeKey, std::vector<void*>> freeBuffers;
    std::unordered_map<void*, AllocInfo> allocs; // every live allocation
    std::vector<PendingBuffer> pendingBuffers;
    std::vector<cudaEvent_t> freeEvents;
    size_t allocations = 0;
    size_t cachedBytes = 0;
    size_t maxCachedBytes = std::numeric_limits<size_t>::max();
    mutable std::mutex mutex;
};

template <typename T>
struct DeviceBufferDeleter {
    void operator()(T* ptr) const noexcept {
        if (ptr) {
            DeviceBufferPool::instance().release(ptr);
        }
    }
};

template <typename T>
using DeviceBufferPtr = std::unique_ptr<T, DeviceBufferDeleter<T>>;

template <typename T>
DeviceBufferPtr<T> acquireDeviceBuffer(size_t count) {
    return DeviceBufferPtr<T>(static_cast<T*>(DeviceBufferPool::instance().acquire(count * sizeof(T))));
}

template <typename T>
struct DeviceBufferAsyncDeleter {
    cudaStream_t stream = 0;
    void operator()(T* ptr) const noexcept {
        if (ptr) {
            DeviceBufferPool::instance().releaseAsync(ptr, stream);
        }
    }
};

template <typename T>
using DeviceBufferPtrAsync = std::unique_ptr<T, DeviceBufferAsyncDeleter<T>>;

template <typename T>
DeviceBufferPtrAsync<T> acquireDeviceBufferAsync(size_t count, cudaStream_t stream) {
    return DeviceBufferPtrAsync<T>(
        static_cast<T*>(DeviceBufferPool::instance().acquire(count * sizeof(T))),
        DeviceBufferAsyncDeleter<T>{stream});
}

#endif
