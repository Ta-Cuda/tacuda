/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "tacuda.h"
#include <string>

// Thread-local storage for the last error message
static thread_local std::string tls_last_error;

extern "C" {

const char* ct_status_string(ctStatus_t status) {
    switch (status) {
        case CT_STATUS_SUCCESS:          return "success";
        case CT_STATUS_ALLOC_FAILED:     return "memory allocation failed";
        case CT_STATUS_COPY_FAILED:      return "host/device memory copy failed";
        case CT_STATUS_KERNEL_FAILED:    return "CUDA kernel execution failed";
        case CT_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case CT_STATUS_DEVICE_ERROR:     return "CUDA device error";
        case CT_STATUS_UNSUPPORTED:      return "unsupported operation";
        default:                         return "unknown error";
    }
}

const char* ct_get_last_error(void) {
    return tls_last_error.c_str();
}

const char* ct_version(void) {
    return CT_VERSION_STRING;
}

} // extern "C"

// Internal helpers (not part of the C API)
namespace tacuda {
namespace detail {

void set_last_error(const std::string& msg) {
    tls_last_error = msg;
}

void set_last_error(const char* msg) {
    tls_last_error = msg ? msg : "";
}

void clear_last_error() {
    tls_last_error.clear();
}

} // namespace detail
} // namespace tacuda
