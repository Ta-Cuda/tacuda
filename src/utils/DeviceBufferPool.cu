/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <utils/DeviceBufferPool.h>
#include <utils/CudaUtils.h>

#include <exception>
#include <iostream>

DeviceBufferPool& DeviceBufferPool::instance() {
    static DeviceBufferPool pool;
    return pool;
}

cudaEvent_t DeviceBufferPool::acquireEventLocked() {
    if (!freeEvents.empty()) {
        cudaEvent_t ev = freeEvents.back();
        freeEvents.pop_back();
        return ev;
    }
    cudaEvent_t ev = nullptr;
    cudaError_t err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        return nullptr;
    }
    return ev;
}

void DeviceBufferPool::recycleEventLocked(cudaEvent_t ev) {
    if (!ev) return;
    freeEvents.push_back(ev);
}

void DeviceBufferPool::freeLocked(void* ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err == cudaSuccess) {
        allocs.erase(ptr);
    }
}

void DeviceBufferPool::drainPendingLocked(size_t sizeFilter) {
    for (auto it = pendingBuffers.begin(); it != pendingBuffers.end();) {
        if (sizeFilter != 0 && it->size != sizeFilter) {
            ++it;
            continue;
        }

        cudaError_t q = cudaEventQuery(it->readyEvent);
        if (q == cudaSuccess) {
            // Work completed; recycle event and either cache or free.
            void* ptr = it->ptr;
            const size_t bytes = it->size;
            const int device = it->device;
            const bool cache = it->cacheWhenReady;
            cudaEvent_t ev = it->readyEvent;
            it = pendingBuffers.erase(it);

            recycleEventLocked(ev);

            if (cache) {
                freeBuffers[{device, bytes}].push_back(ptr);
            } else {
                freeLocked(ptr);
            }
            continue;
        }

        if (q == cudaErrorNotReady) {
            ++it;
            continue;
        }

        // Unknown failure (often during shutdown). Best-effort cleanup.
        cudaEventDestroy(it->readyEvent);
        cudaFree(it->ptr);
        allocs.erase(it->ptr);
        it = pendingBuffers.erase(it);
    }
}

void* DeviceBufferPool::acquire(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    int device = 0;
    (void)cudaGetDevice(&device);

    std::lock_guard<std::mutex> lock(mutex);
    // Move any stream-deferred buffers that are now ready into the free list.
    drainPendingLocked(bytes);
    auto it = freeBuffers.find({device, bytes});
    if (it != freeBuffers.end() && !it->second.empty()) {
        void* ptr = it->second.back();
        it->second.pop_back();
        if (cachedBytes >= bytes) {
            cachedBytes -= bytes;
        } else {
            cachedBytes = 0;
        }
        return ptr;
    }
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    allocs[ptr] = AllocInfo{bytes, device};
    ++allocations;
    return ptr;
}

void DeviceBufferPool::release(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex);
    // Opportunistically drain all ready pending buffers to keep memory usage bounded.
    drainPendingLocked();
    auto it = allocs.find(ptr);
    if (it != allocs.end()) {
        const size_t size = it->second.size;
        const int device = it->second.device;
        if (cachedBytes + size <= maxCachedBytes) {
            freeBuffers[{device, size}].push_back(ptr);
            cachedBytes += size;
        } else {
            // Free immediately when exceeding cache limit. Do not throw if cudaFree fails
            // to avoid propagating exceptions during pool teardown or test cleanup.
            freeLocked(ptr);
        }
    }
}

void DeviceBufferPool::releaseAsync(void* ptr, cudaStream_t stream) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex);

    // Drain pending buffers of the same size first to reduce allocation pressure.
    auto sizeIt = allocs.find(ptr);
    if (sizeIt == allocs.end()) {
        return;
    }
    const size_t bytes = sizeIt->second.size;
    const int device = sizeIt->second.device;
    drainPendingLocked(bytes);

    const bool cache = (cachedBytes + bytes <= maxCachedBytes);
    if (cache) {
        cachedBytes += bytes;
    }

    cudaEvent_t ev = acquireEventLocked();
    if (!ev) {
        // Fallback: only a SUCCESSFUL synchronize proves the buffer is idle.
        // If the sync fails (destroyed stream, dying context) the pointer
        // must never re-enter the cache — leak-on-error is safe,
        // use-after-free is not. cudaFree device-syncs on success.
        if (cache && cachedBytes >= bytes) cachedBytes -= bytes;
        if (cudaStreamSynchronize(stream) == cudaSuccess) {
            if (cache) {
                cachedBytes += bytes;
                freeBuffers[{device, bytes}].push_back(ptr);
                return;
            }
        }
        freeLocked(ptr);
        return;
    }

    cudaError_t rec = cudaEventRecord(ev, stream);
    if (rec != cudaSuccess) {
        // Recording failed (e.g. the stream was destroyed): same rule —
        // successful sync or the pointer leaves the pool entirely.
        cudaEventDestroy(ev);
        if (cache && cachedBytes >= bytes) cachedBytes -= bytes;
        if (cudaStreamSynchronize(stream) == cudaSuccess) {
            if (cache) {
                cachedBytes += bytes;
                freeBuffers[{device, bytes}].push_back(ptr);
                return;
            }
        }
        freeLocked(ptr);
        return;
    }

    pendingBuffers.push_back(PendingBuffer{ptr, bytes, device, ev, cache});
}

void DeviceBufferPool::cleanup() {
    std::lock_guard<std::mutex> lock(mutex);
    // Make sure no in-flight kernels are still using pooled buffers.
    // This is an explicit "tear down the pool" operation, so a device-wide sync
    // is acceptable and prevents freeing memory still referenced by a stream.
    (void)cudaDeviceSynchronize();

    // Dispose of pending buffers (their work is done after the sync).
    for (auto& pending : pendingBuffers) {
        cudaEventDestroy(pending.readyEvent);
        cudaFree(pending.ptr);
        allocs.erase(pending.ptr);
    }
    pendingBuffers.clear();
    for (cudaEvent_t ev : freeEvents) {
        cudaEventDestroy(ev);
    }
    freeEvents.clear();

    // Free cached (free-listed) buffers only. Allocations still held by
    // callers stay valid and re-enter the pool via release()/releaseAsync().
    for (auto& kv : freeBuffers) {
        for (void* ptr : kv.second) {
            cudaFree(ptr);
            allocs.erase(ptr);
        }
    }
    freeBuffers.clear();
    // allocations counter is cumulative - don't reset it
    cachedBytes = 0;
}

size_t DeviceBufferPool::setMaxCacheBytes(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex);
    size_t oldLimit = maxCachedBytes;
    maxCachedBytes = bytes;

    // First, move any ready stream-deferred buffers into the free list.
    drainPendingLocked();

    // If we exceed the new cache limit, prefer evicting immediately-free buffers.
    if (cachedBytes <= maxCachedBytes) {
        return oldLimit;
    }

    for (auto it = freeBuffers.begin(); it != freeBuffers.end() && cachedBytes > maxCachedBytes;) {
        auto& vec = it->second;
        const size_t bufferSize = it->first.second;
        while (!vec.empty() && cachedBytes > maxCachedBytes) {
            void* ptr = vec.back();
            vec.pop_back();
            cudaError_t err = cudaFree(ptr);
            if (err == cudaSuccess) {
                cachedBytes -= bufferSize;
                allocs.erase(ptr);
            }
            // If cudaFree fails, just continue (may happen during shutdown)
        }
        if (vec.empty()) {
            it = freeBuffers.erase(it);
        } else {
            ++it;
        }
    }

    // If we still exceed the limit, mark pending cached buffers for eviction
    // once their stream work completes.
    for (auto& pending : pendingBuffers) {
        if (!pending.cacheWhenReady) continue;
        if (cachedBytes <= maxCachedBytes) break;
        pending.cacheWhenReady = false;
        if (cachedBytes >= pending.size) {
            cachedBytes -= pending.size;
        } else {
            cachedBytes = 0;
        }
    }
    return oldLimit;
}

DeviceBufferPool::~DeviceBufferPool() {
    // During program shutdown, CUDA may already be shutting down.
    // We silently attempt to free resources, but don't throw exceptions.
    std::lock_guard<std::mutex> lock(mutex);

    // Best-effort cleanup of outstanding events.
    for (auto& pending : pendingBuffers) {
        cudaEventDestroy(pending.readyEvent);
    }
    for (cudaEvent_t ev : freeEvents) {
        cudaEventDestroy(ev);
    }

    // Free every live allocation exactly once (including buffers still held
    // by callers: at process teardown nothing may touch them anymore).
    for (auto& kv : allocs) {
        cudaFree(kv.first);  // Ignore errors during shutdown
    }
    // No need to clear containers - destructor is being called anyway
}

size_t DeviceBufferPool::allocationCount() const {
    std::lock_guard<std::mutex> lock(mutex);
    return allocations;
}
