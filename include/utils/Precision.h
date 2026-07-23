/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_PRECISION_H
#define TACUDA_PRECISION_H

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <tacuda.h>

namespace tacuda::precision {

inline bool is_valid_dtype(ctDataType_t dtype) {
  return dtype == CT_DTYPE_F16 || dtype == CT_DTYPE_F32 || dtype == CT_DTYPE_F64;
}

// Returns 0 for an unknown dtype — callers must validate (an out-of-range
// enum silently treated as float would corrupt sizes and reinterpret data).
inline size_t element_size(ctDataType_t dtype) {
  switch (dtype) {
    case CT_DTYPE_F16:
      return sizeof(__half);
    case CT_DTYPE_F64:
      return sizeof(double);
    case CT_DTYPE_F32:
      return sizeof(float);
    default:
      return 0;
  }
}

inline bool is_supported_compute_pair(ctDataType_t storage, ctDataType_t compute) {
  if (!is_valid_dtype(storage) || !is_valid_dtype(compute)) {
    return false;
  }
  if (storage == CT_DTYPE_F16) {
    return compute == CT_DTYPE_F32;
  }
  return compute == CT_DTYPE_F32 || compute == CT_DTYPE_F64;
}

}  // namespace tacuda::precision

#endif  // TACUDA_PRECISION_H
