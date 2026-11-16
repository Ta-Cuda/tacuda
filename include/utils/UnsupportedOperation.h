/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_UNSUPPORTED_OPERATION_H
#define TACUDA_UNSUPPORTED_OPERATION_H

#include <stdexcept>

namespace tacuda {
namespace detail {

// An unavailable runtime capability, distinct from allocation/device failure.
// Internal exception only: wrappers translate it to the existing C status.
class UnsupportedOperation : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

}  // namespace detail
}  // namespace tacuda

#endif
