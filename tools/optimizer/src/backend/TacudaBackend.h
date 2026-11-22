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
// back over PCIe, straight into the caller's buffers. Output lines and
// chain scratch come from a process-wide workspace shared by every
// TacudaBackend (see TacudaBackend.cpp), so a compute() call performs no
// cudaMalloc / cudaFree and the per-symbol device loop of `portfolio
// --backend tacuda-device` does not multiply that memory by the symbol
// count. This is the optimizer's M1 fast path.
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
  // included) into a device buffer without host round-trips. `depth` picks
  // the workspace scratch line that holds a chained upstream result.
  void compute_line_device(const IndicatorRequest& req, float* dOut,
                           std::size_t depth);
  // Downloads workspace lines 0..nOut-1 into outs, one sync for all lines.
  void download_lines(const std::vector<float*>& outs, int nOut);

  const Series& bars_;
  std::array<float*, 5> deviceColumns_{}; // indexed by Column, lazy upload
};

} // namespace topt

#endif // TACUDA_OPT_BACKEND_TACUDABACKEND_H
