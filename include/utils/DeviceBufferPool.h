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

namespace tacuda {
namespace detail {

// True while `stream` is being captured into a CUDA graph (also while a
// capture that already failed has not been ended yet).
bool stream_is_capturing(cudaStream_t stream) noexcept;

// How call-local scratch memory for a stream is allocated. Every mode is
// ordered on the stream; none synchronizes the host.
enum class ScratchMode : unsigned char {
    // Library-private cudaMemPool (cudaMallocFromPoolAsync / cudaFreeAsync).
    // A block freed on a stream is reused by the next allocation on that
    // stream at once, so a batch that enqueues series far ahead of the GPU
    // still needs about one series' scratch per stream. (DeviceBufferPool
    // reuses a block only after its event fired: every series in flight
    // would take fresh memory.)
    StreamOrdered,
    // Stream capture: cudaMallocAsync / cudaFreeAsync become memory nodes
    // owned by the graph. Pooled memory must not be used here: the graph
    // would keep writing to it on every replay after the pool handed it to
    // someone else, and querying an event recorded on a capturing stream
    // invalidates the capture.
    Graph,
    // Devices without memory-pool support: DeviceBufferPool + release event.
    Pooled,
};

// Picks the mode for `stream` now. Evaluate once per buffer (or allocator)
// and pass the same value to the matching scratch_release.
ScratchMode scratch_mode(cudaStream_t stream) noexcept;
void* scratch_acquire(size_t bytes, cudaStream_t stream, ScratchMode mode);
void scratch_release(void* ptr, cudaStream_t stream, ScratchMode mode) noexcept;

// Releases unused memory held by the private scratch pools (all devices);
// DeviceBufferPool::cleanup() calls it.
void scratch_trim() noexcept;
// Memory the scratch pools keep reserved across synchronizations
// (cudaMemPoolAttrReleaseThreshold); DeviceBufferPool::setMaxCacheBytes()
// applies its limit here too.
void scratch_set_release_threshold(size_t bytes) noexcept;

}  // namespace detail
}  // namespace tacuda

template <typename T>
struct DeviceBufferAsyncDeleter {
    cudaStream_t stream = 0;
    tacuda::detail::ScratchMode mode = tacuda::detail::ScratchMode::Pooled;
    void operator()(T* ptr) const noexcept {
        if (ptr) {
            tacuda::detail::scratch_release(ptr, stream, mode);
        }
    }
};

template <typename T>
using DeviceBufferPtrAsync = std::unique_ptr<T, DeviceBufferAsyncDeleter<T>>;

template <typename T>
DeviceBufferPtrAsync<T> acquireDeviceBufferAsync(size_t count, cudaStream_t stream) {
    const tacuda::detail::ScratchMode mode = tacuda::detail::scratch_mode(stream);
    return DeviceBufferPtrAsync<T>(
        static_cast<T*>(tacuda::detail::scratch_acquire(count * sizeof(T), stream, mode)),
        DeviceBufferAsyncDeleter<T>{stream, mode});
}

#endif
