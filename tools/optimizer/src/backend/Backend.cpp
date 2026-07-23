/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "backend/Backend.h"

#include <stdexcept>

#include "backend/TacudaBackend.h"
#include "backend/TalibBackend.h"

namespace topt {

BackendKind parse_backend_kind(const std::string& text) {
  if (text == "auto") return BackendKind::Auto;
  if (text == "tacuda" || text == "gpu" || text == "cuda") return BackendKind::Tacuda;
  if (text == "tacuda-device") return BackendKind::TacudaDevice;
  if (text == "talib" || text == "cpu") return BackendKind::Talib;
  throw std::invalid_argument("unknown backend '" + text +
                              "' (use auto|tacuda|tacuda-device|talib)");
}

std::unique_ptr<IIndicatorBackend> create_backend(BackendKind kind,
                                                  const Series& bars,
                                                  std::string& log) {
  switch (kind) {
  case BackendKind::Tacuda:
  case BackendKind::TacudaDevice: // same thing for a single series
    log = "backend: tacuda (GPU, forced)";
    return std::make_unique<TacudaBackend>(bars);
  case BackendKind::Talib:
    log = "backend: talib (CPU, forced)";
    return std::make_unique<TalibBackend>(bars);
  case BackendKind::Auto:
    break;
  }

  if (TacudaBackend::available()) {
    log = "backend: tacuda (GPU)";
    return std::make_unique<TacudaBackend>(bars);
  }
  std::string why;
  if (TalibBackend::available(why)) {
    log = "backend: talib (CPU fallback, no CUDA device)";
    return std::make_unique<TalibBackend>(bars);
  }
  throw std::runtime_error(
      "no usable indicator backend: no CUDA device for tacuda and " + why);
}

} // namespace topt
