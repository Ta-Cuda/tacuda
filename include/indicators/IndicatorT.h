/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef INDICATOR_T_H
#define INDICATOR_T_H

#include <cuda_runtime.h>

namespace tacuda {

/**
 * @brief Templated base class for indicators supporting float/double.
 *
 * Parallel hierarchy to Indicator (float-only). Use IndicatorT<double> for
 * double-precision pipelines; IndicatorT<float> is equivalent in functionality
 * to Indicator but lives in a separate type hierarchy.
 */
template<typename T>
class IndicatorT {
public:
    virtual void calculate(const T* input, T* output, int size,
                           cudaStream_t stream = 0) noexcept(false) = 0;
    virtual ~IndicatorT() = default;
};

} // namespace tacuda

#endif // INDICATOR_T_H
