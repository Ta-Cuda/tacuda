/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_ERROR_STATE_H
#define TACUDA_ERROR_STATE_H

#include <string>

namespace tacuda {
namespace detail {

/**
 * @brief Store an error message in thread-local storage.
 * Called by API wrappers when an operation fails.
 */
void set_last_error(const std::string& msg);
void set_last_error(const char* msg);

/**
 * @brief Clear the stored error (called on success).
 */
void clear_last_error();

} // namespace detail
} // namespace tacuda

#endif // TACUDA_ERROR_STATE_H
