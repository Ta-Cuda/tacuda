/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#pragma once
#include <tacuda.h>
#include <utils/UnsupportedOperation.h>
#include <stdexcept>

namespace tacuda { namespace talib08 {
// Generated enum mirrors the pinned catalog, not a floating installed oracle.
#include "Talib08Functions.inc"
struct Params { double value[16] = {}; };
ctStatus_t run_device(Function function, const float* const* inputs,
                      void* const* outputs, int length, int series,
                      const Params& params, cudaStream_t stream);
ctStatus_t run_host(Function function, const float* const* inputs,
                    void* const* outputs, int length, int series,
                    const Params& params, cudaStream_t stream);
inline void require_success(ctStatus_t status) {
    if (status == CT_STATUS_SUCCESS) return;
    if (status == CT_STATUS_INVALID_ARGUMENT) throw std::invalid_argument(ct_get_last_error());
    if (status == CT_STATUS_UNSUPPORTED) throw detail::UnsupportedOperation(ct_get_last_error());
    throw std::runtime_error(ct_get_last_error());
}
}}
