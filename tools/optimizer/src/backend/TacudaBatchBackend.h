/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_BACKEND_TACUDABATCHBACKEND_H
#define TACUDA_OPT_BACKEND_TACUDABATCHBACKEND_H

#include <array>
#include <vector>

#include "backend/BatchBackend.h"

namespace topt {

// GPU portfolio backend on top of the TaCuda batch API. Per used OHLCV
// column one numSeries x seriesLen matrix (series-major) is uploaded to the
// device once and reused; each compute() issues a single ct_*_batch_d call
// that fans the per-symbol kernels out over TaCuda's internal stream pool,
// then downloads each output line as one matrix-sized transfer and scatters
// it into the per-symbol buffers.
class TacudaBatchBackend final : public IBatchIndicatorBackend {
public:
  // All series must be non-empty and share one length.
  explicit TacudaBatchBackend(const std::vector<Series>& series);
  ~TacudaBatchBackend() override;

  TacudaBatchBackend(const TacudaBatchBackend&) = delete;
  TacudaBatchBackend& operator=(const TacudaBatchBackend&) = delete;

  const char* name() const override { return "tacuda-batch"; }
  int num_series() const override { return numSeries_; }
  void compute(const IndicatorRequest& req,
               const std::vector<std::vector<float*>>& outs) override;

private:
  float* matrix_device(Column c);
  float* line_device(int k);
  // Computes any single-output request (club families and derived chains
  // included) into a numSeries x seriesLen device matrix.
  void compute_matrix_device(const IndicatorRequest& req, float* dOut);

  const std::vector<Series>& series_;
  int numSeries_ = 0;
  int seriesLen_ = 0;
  std::array<float*, 5> deviceMatrices_{}; // indexed by Column, lazy upload
  std::array<float*, 3> lineMatrices_{};   // reusable output matrices
  std::vector<float> staging_;             // reusable download buffer
};

} // namespace topt

#endif // TACUDA_OPT_BACKEND_TACUDABATCHBACKEND_H
