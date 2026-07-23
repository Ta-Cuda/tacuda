/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_BACKEND_BACKEND_H
#define TACUDA_OPT_BACKEND_BACKEND_H

#include <memory>
#include <string>
#include <vector>

#include "core/Types.h"

namespace topt {

// Computes indicator lines for a fixed OHLCV series. Implementations are
// TacudaBackend (GPU via the TaCuda C API) and TalibBackend (CPU fallback
// via a dynamically loaded TA-Lib). Not required to be thread-safe: the
// IndicatorCache serializes compute calls.
class IIndicatorBackend {
public:
  virtual ~IIndicatorBackend() = default;
  virtual const char* name() const = 0;

  // Computes req over the bound series into outs (family_outputs(req.family)
  // pre-allocated arrays of series length). Warm-up prefixes are NaN.
  // Throws std::runtime_error on failure.
  virtual void compute(const IndicatorRequest& req,
                       const std::vector<float*>& outs) = 0;
};

// TacudaDevice forces the single-series device API even in portfolio mode
// (one ct_*_d call per symbol instead of one ct_*_batch_d call per request)
// — the baseline the batch API is benchmarked against. For single-series
// commands it behaves exactly like Tacuda.
enum class BackendKind { Auto, Tacuda, TacudaDevice, Talib };

BackendKind parse_backend_kind(const std::string& text);

// Creates the indicator backend bound to `bars`:
// - Tacuda: requires a usable CUDA device, throws otherwise;
// - Talib: requires a loadable TA-Lib shared library, throws otherwise;
// - Auto: tacuda when a CUDA device is present, else TA-Lib fallback,
//   throws only when neither works. `log` receives one line describing
//   the choice (may be empty).
std::unique_ptr<IIndicatorBackend> create_backend(BackendKind kind,
                                                  const Series& bars,
                                                  std::string& log);

} // namespace topt

#endif // TACUDA_OPT_BACKEND_BACKEND_H
