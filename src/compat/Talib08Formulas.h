/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#pragma once
// Private CUDA adaptation of pinned TA-Lib 0.8.3 batch formulas.
// The source license and import manifest are in third_party/ta_lib_0_8_3.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <climits>
#include "../../third_party/ta_lib_0_8_3/ta_defs.h"

#ifdef __CUDACC__
#define TA08_HD __host__ __device__ __noinline__ inline
#else
#define TA08_HD inline
#endif

namespace tacuda { namespace talib08 { namespace ref {

// Per-worker arena over caller-stream scratch. There is no device malloc,
// global mutable state, or workspace shared by different series. Out-of-order
// frees coalesce when the live top allocation is released.
struct Arena {
    struct Allocation {
        void* pointer;
        template<class T> TA08_HD operator T*() const { return static_cast<T*>(pointer); }
    };
    struct alignas(16) Block { size_t previousUsed, previousBlock, free, reserved; };
    unsigned char* data;
    size_t capacity, used = 0, last = SIZE_MAX, highWater = 0;
    bool failed = false;
    TA08_HD Allocation allocate(size_t bytes) {
        const size_t offset = (used + 15) & ~size_t(15);
        if (offset > capacity || sizeof(Block) > capacity - offset ||
            bytes > capacity - offset - sizeof(Block)) { failed = true; return {nullptr}; }
        auto* header = reinterpret_cast<Block*>(data + offset);
        *header = Block{used, last, 0, 0};
        last = offset;
        used = offset + sizeof(Block) + bytes;
        if (used > highWater) highWater = used;
        return {header + 1};
    }
    TA08_HD void release(void* pointer) {
        if (!pointer) return;
        (static_cast<Block*>(pointer) - 1)->free = 1;
        while (last != SIZE_MAX) {
            auto* header = reinterpret_cast<Block*>(data + last);
            if (!header->free) break;
            used = header->previousUsed;
            last = header->previousBlock;
        }
    }
};

template<class T, class U> TA08_HD auto min(T a, U b) { return a < b ? a : b; }
template<class T, class U> TA08_HD auto max(T a, U b) { return a > b ? a : b; }
using std::isfinite;
TA08_HD void* memcpy(void* dst, const void* src, size_t bytes) {
    auto* d = static_cast<unsigned char*>(dst);
    const auto* s = static_cast<const unsigned char*>(src);
    for (size_t i = 0; i < bytes; ++i) d[i] = s[i];
    return dst;
}
TA08_HD void* memmove(void* dst, const void* src, size_t bytes) {
    auto* d = static_cast<unsigned char*>(dst);
    const auto* s = static_cast<const unsigned char*>(src);
    if (reinterpret_cast<uintptr_t>(d) < reinterpret_cast<uintptr_t>(s)) {
        for (size_t i = 0; i < bytes; ++i) d[i] = s[i];
    } else {
        for (size_t i = bytes; i > 0; --i) d[i - 1] = s[i - 1];
    }
    return dst;
}
#define TA_INTERNAL_ERROR(id) TA_UNKNOWN_ERR
#define TA_IS_FINITE(v) (isfinite(v))
#define TA_EPSILON 1e-14
#define TA_IS_ZERO(v) ((-TA_EPSILON < (v)) && ((v) < TA_EPSILON))
#define TA_IS_ZERO_SCALED(v, scale) (fabs(v) <= TA_EPSILON * (scale))
#define TA_MIN_LT(a, b) (((a) < (b)) ? (a) : (b))
#define TA_MAX_GT(a, b) (((a) > (b)) ? (a) : (b))
#define TA_UNLIKELY(c) (c)
#define TA_UNROLL(n)
#define TA_GLOBALS_UNSTABLE_OFFSET(...) 0u
#define TA_GLOBALS_UNSTABLE(...) 0u
#include "talib08_formulas.inc"
#undef TA_INTERNAL_ERROR
#undef TA_IS_FINITE
#undef TA_EPSILON
#undef TA_IS_ZERO
#undef TA_IS_ZERO_SCALED
#undef TA_MIN_LT
#undef TA_MAX_GT
#undef TA_UNLIKELY
#undef TA_UNROLL
#undef TA_GLOBALS_UNSTABLE_OFFSET
#undef TA_GLOBALS_UNSTABLE
}}}
#undef TA08_HD
