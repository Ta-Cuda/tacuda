/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_BACKEND_BATCHBACKEND_H
#define TACUDA_OPT_BACKEND_BATCHBACKEND_H

#include <memory>
#include <string>
#include <vector>

#include "backend/Backend.h"
#include "core/Types.h"

namespace topt {

// Computes one indicator request across a whole portfolio of equal-length
// series (same parameters for every symbol — the ct_*_batch contract).
// outs[symbol] holds family_outputs(req.family) pre-allocated arrays of
// series length. Implementations: TacudaBatchBackend (GPU, one call per
// request over a numSeries x seriesLen device matrix) and
// PerSymbolBatchBackend (loops per-series backends; the CPU fallback).
class IBatchIndicatorBackend {
public:
  virtual ~IBatchIndicatorBackend() = default;
  virtual const char* name() const = 0;
  virtual int num_series() const = 0;

  // Throws std::runtime_error on failure.
  virtual void compute(const IndicatorRequest& req,
                       const std::vector<std::vector<float*>>& outs) = 0;
};

// Fallback that serves each symbol through its own single-series backend
// (TA-Lib in production, TestBackend in unit tests). With threads > 1 the
// symbols are split across a thread pool — each worker only touches its own
// backend instances, so the per-symbol backends need no internal locking.
class PerSymbolBatchBackend final : public IBatchIndicatorBackend {
public:
  PerSymbolBatchBackend(std::vector<std::unique_ptr<IIndicatorBackend>> backends,
                        std::string name, int threads = 1);

  const char* name() const override { return name_.c_str(); }
  int num_series() const override { return static_cast<int>(backends_.size()); }
  int threads() const { return threads_; }
  void compute(const IndicatorRequest& req,
               const std::vector<std::vector<float*>>& outs) override;

private:
  std::vector<std::unique_ptr<IIndicatorBackend>> backends_;
  std::string name_;
  int threads_ = 1;
};

// Portfolio counterpart of create_backend(): tacuda batch API on the GPU,
// per-symbol TA-Lib on the CPU, auto prefers the GPU. All series must have
// the same length (align first). `series` must outlive the backend.
// `cpuThreads` parallelizes the per-symbol CPU path across symbols
// (0 = all hardware cores, 1 = sequential); GPU paths ignore it.
std::unique_ptr<IBatchIndicatorBackend> create_batch_backend(
    BackendKind kind, const std::vector<Series>& series, std::string& log,
    int cpuThreads = 1);

} // namespace topt

#endif // TACUDA_OPT_BACKEND_BATCHBACKEND_H
