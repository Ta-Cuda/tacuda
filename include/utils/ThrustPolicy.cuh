/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_THRUST_POLICY_CUH
#define TACUDA_THRUST_POLICY_CUH

/**
 * @file ThrustPolicy.cuh
 * @brief The only execution policy library code may pass to Thrust.
 *
 * `thrust::cuda::par.on(stream)` is unusable inside `ct_*_d`: every algorithm
 * call does cudaMalloc for its temporary storage, then cudaStreamSynchronize,
 * then cudaFree. The host blocks once per scan (a batch call blocks once per
 * series, so its stream pool never overlaps anything), and the synchronize is
 * illegal while the stream is captured into a CUDA graph.
 *
 * thrust_on(stream) returns `par_nosync` with stream-ordered temporary
 * storage (see tacuda::detail::scratch_acquire): no host synchronization, and
 * graph-owned memory nodes when the stream is being captured.
 *
 * Algorithms that return a value to the host (reduce, transform_reduce,
 * count, *_element, ...) synchronize regardless of the policy. Do not use
 * them in device paths: compute such values on the device instead
 * (utils/DeviceSeed.cuh).
 */

#include <cstddef>

#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/version.h>

#include <utils/DeviceBufferPool.h>

namespace tacuda {
namespace detail {

/// Thrust temporary-storage allocator ordered on one stream. The scratch
/// mode (capture state included) is fixed at construction so every release
/// takes the same path as its allocation.
class ThrustScratchAllocator {
public:
    using value_type = char;

    explicit ThrustScratchAllocator(cudaStream_t stream)
        : stream_(stream), mode_(scratch_mode(stream)) {}

    char* allocate(std::size_t bytes) {
        // Never hand CUB a null temp-storage pointer: with nullptr it only
        // reports the size it needs and skips the algorithm.
        return static_cast<char*>(scratch_acquire(bytes > 0 ? bytes : 1, stream_, mode_));
    }

    void deallocate(char* ptr, std::size_t) noexcept {
        scratch_release(ptr, stream_, mode_);
    }

private:
    cudaStream_t stream_;
    ScratchMode mode_;
};

inline auto thrust_on(cudaStream_t stream) {
#if THRUST_VERSION >= 101600
    return thrust::cuda::par_nosync(ThrustScratchAllocator(stream)).on(stream);
#else
    // No par_nosync before Thrust 1.16: still synchronizes, but keeps the
    // stream-ordered temporary storage.
    return thrust::cuda::par(ThrustScratchAllocator(stream)).on(stream);
#endif
}

}  // namespace detail
}  // namespace tacuda

#endif  // TACUDA_THRUST_POLICY_CUH
