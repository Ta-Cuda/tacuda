/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_BACKEND_TACUDABACKEND_H
#define TACUDA_OPT_BACKEND_TACUDABACKEND_H

#include <array>
#include <cstddef>

#include "backend/Backend.h"

namespace topt {

// GPU backend on top of the TaCuda device API (tacuda_device.h).
// OHLCV columns are uploaded to the GPU once (lazily, per column) and reused
// by every compute call; per call only the requested output lines travel
// back over PCIe. This is the optimizer's M1 fast path.
class TacudaBackend : public IIndicatorBackend {
public:
  explicit TacudaBackend(const Series& bars);
  ~TacudaBackend() override;

  TacudaBackend(const TacudaBackend&) = delete;
  TacudaBackend& operator=(const TacudaBackend&) = delete;

  const char* name() const override { return "tacuda"; }
  void compute(const IndicatorRequest& req, const std::vector<float*>& outs) override;

  // True when a CUDA device is usable in this process.
  static bool available();

private:
  float* column_device(Column c);
  // Computes any single-output request (club families and derived chains
  // included) into a device buffer without host round-trips.
  void compute_line_device(const IndicatorRequest& req, float* dOut);

  const Series& bars_;
  std::array<float*, 5> deviceColumns_{}; // indexed by Column, lazy upload
};

} // namespace topt

#endif // TACUDA_OPT_BACKEND_TACUDABACKEND_H
