/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_AFFINE_SCAN_CUH
#define TACUDA_AFFINE_SCAN_CUH

/**
 * @file AffineScan.cuh
 * @brief Shared primitives for parallelising affine recurrences via prefix scan.
 *
 * An affine recurrence  y[t] = A * y[t-1] + b[t]  can be expressed as the
 * composition of affine transforms (A, b) and evaluated in parallel via
 * inclusive scan with an associative composition operator.
 *
 * This header provides:
 *  - AffineTransform1<T>  — single-channel  y = A*y + b
 *  - AffineTransform2<T>  — two-channel     yG = A*yG + bG,  yL = A*yL + bL
 *  - AffineCompose1<T>    — composition operator for AffineTransform1
 *  - AffineCompose2<T>    — composition operator for AffineTransform2
 *  - affine_inclusive_scan / affine_inclusive_scan_by_key — convenience wrappers
 */

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/functional.h>

namespace tacuda {
namespace detail {

// ============================================================================
// Single-channel affine transform:  y = A * y_prev + b
// ============================================================================
template <typename T = double>
struct AffineTransform1 {
    T A;  // multiplier
    T b;  // addend
};

// Composition:  applying x first, then y  =>  (y.A * x.A,  y.A * x.b + y.b)
template <typename T = double>
struct AffineCompose1 {
    __host__ __device__
    AffineTransform1<T> operator()(const AffineTransform1<T>& x,
                                   const AffineTransform1<T>& y) const {
        AffineTransform1<T> r;
        r.A = y.A * x.A;
        r.b = y.A * x.b + y.b;
        return r;
    }
};

// ============================================================================
// Two-channel affine transform:  yG = A * yG_prev + bG
//                                yL = A * yL_prev + bL
// Used by RSI (gain/loss channels share the same multiplier A).
// ============================================================================
template <typename T = float>
struct AffineTransform2 {
    T A;
    T bG;
    T bL;
};

template <typename T = float>
struct AffineCompose2 {
    __host__ __device__
    AffineTransform2<T> operator()(const AffineTransform2<T>& x,
                                   const AffineTransform2<T>& y) const {
        AffineTransform2<T> r;
        r.A  = y.A * x.A;
        r.bG = y.A * x.bG + y.bG;
        r.bL = y.A * x.bL + y.bL;
        return r;
    }
};

// ============================================================================
// Convenience wrappers around thrust::inclusive_scan
// ============================================================================

/// Non-segmented inclusive scan of affine transforms.
template <typename PairT, typename ComposeT>
inline void affine_inclusive_scan(PairT* d_input, PairT* d_output, int count,
                                 cudaStream_t stream) {
    thrust::device_ptr<PairT> in(d_input);
    thrust::device_ptr<PairT> out(d_output);
    thrust::inclusive_scan(thrust::cuda::par.on(stream),
                           in, in + count, out, ComposeT{});
}

/// Segmented (by-key) inclusive scan of affine transforms.
template <typename PairT, typename ComposeT>
inline void affine_inclusive_scan_by_key(const int* d_keys,
                                        PairT* d_input,
                                        PairT* d_output,
                                        int count,
                                        cudaStream_t stream) {
    thrust::device_ptr<const int> kBeg(d_keys);
    thrust::device_ptr<PairT> in(d_input);
    thrust::device_ptr<PairT> out(d_output);
    thrust::inclusive_scan_by_key(thrust::cuda::par.on(stream),
                                  kBeg, kBeg + count,
                                  in, out,
                                  thrust::equal_to<int>(),
                                  ComposeT{});
}

} // namespace detail
} // namespace tacuda

#endif // TACUDA_AFFINE_SCAN_CUH
