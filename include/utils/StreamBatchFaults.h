/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_STREAM_BATCH_FAULTS_H
#define TACUDA_STREAM_BATCH_FAULTS_H

// Test-only hook: GPU failures cannot be provoked on demand, so the batch
// engine's error paths are driven through named sites. The symbol exists only
// when the library is configured with TACUDA_STREAM_FAULT_INJECTION=ON and is
// not part of the public ABI.

#include "tacuda_stream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ctStreamBatchFault {
  CT_STREAM_BATCH_FAULT_NONE = 0,
  CT_STREAM_BATCH_FAULT_STATE_UPLOAD = 1,
  CT_STREAM_BATCH_FAULT_IO_ALLOC = 2,
  CT_STREAM_BATCH_FAULT_INPUT_UPLOAD = 3,
  CT_STREAM_BATCH_FAULT_KERNEL = 4,
  CT_STREAM_BATCH_FAULT_OUTPUT_DOWNLOAD = 5,
  CT_STREAM_BATCH_FAULT_STATE_DOWNLOAD = 6
} ctStreamBatchFault_t;

// The next operation reaching `site` reports failure, once.
CTAPI_EXPORT void ct_test_stream_batch_fail_next(ctStreamBatchFault_t site);

#ifdef __cplusplus
}
#endif

#endif  // TACUDA_STREAM_BATCH_FAULTS_H
