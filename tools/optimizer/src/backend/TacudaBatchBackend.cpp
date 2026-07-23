/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "backend/TacudaBatchBackend.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <tacuda.h>
#include <tacuda_device.h>

#include "backend/TacudaBackend.h"

namespace topt {

namespace {

[[noreturn]] void raise(const std::string& what, ctStatus_t st) {
  std::string detail = ct_get_last_error();
  std::string msg =
      "tacuda batch backend: " + what + " failed: " + ct_status_string(st);
  if (!detail.empty()) msg += " (" + detail + ")";
  throw std::runtime_error(msg);
}

void check(ctStatus_t st, const char* what) {
  if (st != CT_STATUS_SUCCESS) raise(what, st);
}

// RAII scratch matrix for chained intermediate lines.
struct ScratchMatrix {
  float* ptr = nullptr;
  explicit ScratchMatrix(int elems) : ptr(ct_device_alloc(elems)) {
    if (!ptr) {
      throw std::runtime_error("tacuda batch backend: device allocation failed");
    }
  }
  ~ScratchMatrix() { ct_device_free(ptr); }
  ScratchMatrix(const ScratchMatrix&) = delete;
  ScratchMatrix& operator=(const ScratchMatrix&) = delete;
};

} // namespace

TacudaBatchBackend::TacudaBatchBackend(const std::vector<Series>& series)
    : series_(series), numSeries_(static_cast<int>(series.size())) {
  if (series_.empty()) {
    throw std::invalid_argument("tacuda batch backend: no series");
  }
  seriesLen_ = series_.front().size();
  for (const Series& s : series_) {
    if (s.size() <= 0 || s.size() != seriesLen_) {
      throw std::invalid_argument(
          "tacuda batch backend: series must be non-empty and equal-length");
    }
  }
  if (static_cast<long long>(numSeries_) * seriesLen_ >
      std::numeric_limits<int>::max()) {
    throw std::invalid_argument("tacuda batch backend: matrix too large");
  }
  if (!TacudaBackend::available()) {
    throw std::runtime_error("tacuda batch backend: no usable CUDA device");
  }
}

TacudaBatchBackend::~TacudaBatchBackend() {
  for (float* p : deviceMatrices_) {
    if (p) ct_device_free(p);
  }
  for (float* p : lineMatrices_) {
    if (p) ct_device_free(p);
  }
}

float* TacudaBatchBackend::line_device(int k) {
  auto idx = static_cast<std::size_t>(k);
  if (!lineMatrices_[idx]) {
    lineMatrices_[idx] = ct_device_alloc(numSeries_ * seriesLen_);
    if (!lineMatrices_[idx]) {
      throw std::runtime_error(
          "tacuda batch backend: device allocation failed");
    }
  }
  return lineMatrices_[idx];
}

float* TacudaBatchBackend::matrix_device(Column c) {
  auto idx = static_cast<std::size_t>(c);
  if (deviceMatrices_[idx]) return deviceMatrices_[idx];

  const int elems = numSeries_ * seriesLen_;
  std::vector<float> host(static_cast<std::size_t>(elems));
  for (int s = 0; s < numSeries_; ++s) {
    const std::vector<float>& col =
        series_[static_cast<std::size_t>(s)].column(c);
    std::copy(col.begin(), col.end(),
              host.begin() + static_cast<std::size_t>(s) * seriesLen_);
  }

  float* d = ct_device_alloc(elems);
  if (!d) {
    throw std::runtime_error("tacuda batch backend: device allocation failed");
  }
  ctStatus_t st = ct_device_upload(d, host.data(), elems, 0);
  // The ct_*_batch_d entry points dispatch onto non-blocking pool streams that
  // do NOT implicitly order against the legacy stream this upload runs on, so
  // the matrix must be resident before compute() issues the first batch call
  // (per the batch API contract: "synchronize the producing stream first").
  if (st == CT_STATUS_SUCCESS) st = ct_device_sync(0);
  if (st != CT_STATUS_SUCCESS) {
    ct_device_free(d);
    raise("matrix upload", st);
  }
  deviceMatrices_[idx] = d;
  return d;
}

// Computes any single-output request (club families and derived chains
// included) into a numSeries x seriesLen device matrix; chained upstream
// lines stay on the GPU.
void TacudaBatchBackend::compute_matrix_device(const IndicatorRequest& req,
                                               float* dOut) {
  const int n = seriesLen_;
  const int elems = numSeries_ * n;
  const int p0 = req.params.empty() ? 0 : static_cast<int>(req.params.at(0));

  std::unique_ptr<ScratchMatrix> chained;
  const float* dIn = nullptr;
  if (family_single_input(req.family)) {
    if (req.chain) {
      chained = std::make_unique<ScratchMatrix>(elems);
      compute_matrix_device(*req.chain, chained->ptr);
      dIn = chained->ptr;
    } else {
      dIn = matrix_device(req.input);
    }
  }

  switch (req.family) {
  case Family::SMA:
    check(ct_sma_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_sma_batch_d");
    break;
  case Family::EMA:
    check(ct_ema_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_ema_batch_d");
    break;
  case Family::RSI:
    check(ct_rsi_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_rsi_batch_d");
    break;
  case Family::ATR:
    check(ct_atr_batch_d(matrix_device(Column::High), matrix_device(Column::Low),
                         matrix_device(Column::Close), dOut, n,
                         numSeries_, p0, 0.0f, 0, 0),
          "ct_atr_batch_d");
    break;
  case Family::WMA:
    check(ct_wma_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_wma_batch_d");
    break;
  case Family::TRIMA:
    check(ct_trima_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_trima_batch_d");
    break;
  case Family::MOM:
    check(ct_momentum_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_momentum_batch_d");
    break;
  case Family::ROC:
    check(ct_roc_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_roc_batch_d");
    break;
  case Family::STDDEV:
    check(ct_stddev_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_stddev_batch_d");
    break;
  case Family::LINEARREG:
    check(ct_linearreg_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_linearreg_batch_d");
    break;
  case Family::LINEARREG_SLOPE:
    check(ct_linearreg_slope_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_linearreg_slope_batch_d");
    break;
  case Family::MAX:
    check(ct_max_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_max_batch_d");
    break;
  case Family::MIN:
    check(ct_min_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_min_batch_d");
    break;
  case Family::CCI:
    check(ct_cci_batch_d(matrix_device(Column::High), matrix_device(Column::Low),
                         matrix_device(Column::Close), dOut, n,
                         numSeries_, p0, 0, 0),
          "ct_cci_batch_d");
    break;
  case Family::ADX:
    check(ct_adx_batch_d(matrix_device(Column::High), matrix_device(Column::Low),
                         matrix_device(Column::Close), dOut, n,
                         numSeries_, p0, 0, 0),
          "ct_adx_batch_d");
    break;
  case Family::PLUS_DI:
    check(ct_plus_di_batch_d(matrix_device(Column::High),
                             matrix_device(Column::Low),
                             matrix_device(Column::Close), dOut, n,
                             numSeries_, p0, 0, 0),
          "ct_plus_di_batch_d");
    break;
  case Family::MINUS_DI:
    check(ct_minus_di_batch_d(matrix_device(Column::High),
                              matrix_device(Column::Low),
                              matrix_device(Column::Close), dOut, n,
                              numSeries_, p0, 0, 0),
          "ct_minus_di_batch_d");
    break;
  case Family::OBV:
    check(ct_obv_batch_d(matrix_device(Column::Close),
                         matrix_device(Column::Volume), dOut, n,
                         numSeries_, 0, 0),
          "ct_obv_batch_d");
    break;
  case Family::TRANGE:
    check(ct_trange_batch_d(matrix_device(Column::High),
                            matrix_device(Column::Low),
                            matrix_device(Column::Close), dOut, n,
                            numSeries_, 0, 0),
          "ct_trange_batch_d");
    break;
  case Family::ASI:
    check(ct_asi_batch_d(matrix_device(Column::Open),
                         matrix_device(Column::High),
                         matrix_device(Column::Low),
                         matrix_device(Column::Close), dOut, n, numSeries_,
                         0, 0),
          "ct_asi_batch_d");
    break;
  case Family::SAR_EL:
    check(ct_sar_el_batch_d(matrix_device(Column::High),
                            matrix_device(Column::Low),
                            matrix_device(Column::Close), dOut, n, numSeries_,
                            static_cast<float>(req.params.at(0)),
                            static_cast<float>(req.params.at(1)), 0, 0),
          "ct_sar_el_batch_d");
    break;
  case Family::OBTR:
    check(ct_obtr_batch_d(matrix_device(Column::High),
                          matrix_device(Column::Low),
                          matrix_device(Column::Close), dOut, n, numSeries_,
                          0, 0),
          "ct_obtr_batch_d");
    break;
  case Family::WOBV:
    check(ct_wobv_batch_d(matrix_device(Column::Open),
                          matrix_device(Column::High),
                          matrix_device(Column::Low),
                          matrix_device(Column::Close),
                          matrix_device(Column::Volume), dOut, n, numSeries_,
                          0, 0),
          "ct_wobv_batch_d");
    break;
  case Family::HPI:
    check(ct_hpi_batch_d(matrix_device(Column::High),
                         matrix_device(Column::Low),
                         matrix_device(Column::Volume), dOut, n, numSeries_,
                         static_cast<float>(req.params.at(0)), 0, 0),
          "ct_hpi_batch_d");
    break;
  case Family::MARKET_STRENGTH:
    check(ct_market_strength_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_market_strength_batch_d");
    break;
  case Family::TRENDSCORE:
    check(ct_trendscore_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_trendscore_batch_d");
    break;
  case Family::REFDEV:
    check(ct_refdev_batch_d(dIn, dOut, n, numSeries_, p0,
                            static_cast<int>(req.params.at(1)), 0, 0),
          "ct_refdev_batch_d");
    break;
  case Family::SKEW:
    check(ct_skew_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_skew_batch_d");
    break;
  case Family::KURTOSIS:
    check(ct_kurtosis_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_kurtosis_batch_d");
    break;
  case Family::PIVOT_HIGH:
    check(ct_pivot_high_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_pivot_high_batch_d");
    break;
  case Family::PIVOT_LOW:
    check(ct_pivot_low_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_pivot_low_batch_d");
    break;
  case Family::XAVERAGE:
    check(ct_xaverage_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_xaverage_batch_d");
    break;
  case Family::ROLLING_MEAN:
    check(ct_rolling_mean_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_rolling_mean_batch_d");
    break;
  case Family::ROLLING_MAX:
    check(ct_rolling_max_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_rolling_max_batch_d");
    break;
  case Family::ROLLING_MIN:
    check(ct_rolling_min_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_rolling_min_batch_d");
    break;
  case Family::ROLLING_STD:
    check(ct_rolling_std_batch_d(dIn, dOut, n, numSeries_, p0, 0, 0),
          "ct_rolling_std_batch_d");
    break;
  default:
    throw std::invalid_argument(
        "tacuda batch backend: family is not single-output");
  }
}

void TacudaBatchBackend::compute(const IndicatorRequest& req,
                                 const std::vector<std::vector<float*>>& outs) {
  const int n = seriesLen_;
  const int elems = numSeries_ * n;
  const int nOut = family_outputs(req.family);
  if (static_cast<int>(outs.size()) != numSeries_) {
    throw std::invalid_argument("tacuda batch backend: wrong symbol count");
  }
  for (const auto& symbolOuts : outs) {
    if (static_cast<int>(symbolOuts.size()) != nOut) {
      throw std::invalid_argument(
          "tacuda batch backend: wrong number of output buffers");
    }
  }
  validate_chain(req);

  // One ct_*_batch_d call over the device matrices (maxConcurrency 0 =
  // library-managed stream pool) into persistent per-line output matrices
  // (allocated once, reused across compute() calls).
  std::array<float*, 3> lines{};
  for (int k = 0; k < nOut; ++k) {
    lines[static_cast<std::size_t>(k)] = line_device(k);
  }

  int p0 = req.params.empty() ? 0 : static_cast<int>(req.params.at(0));

  if (nOut == 1) {
    compute_matrix_device(req, lines[0]);
  } else {
    switch (req.family) {
    case Family::BBANDS:
      check(ct_bbands_batch_d(matrix_device(req.input), lines[0], lines[1],
                              lines[2], n, numSeries_, p0,
                              static_cast<float>(req.params.at(1)),
                              static_cast<float>(req.params.at(2)), 0, 0),
            "ct_bbands_batch_d");
      break;
    case Family::MACD:
      check(ct_macd_batch_d(matrix_device(req.input), lines[0], lines[1],
                            lines[2], n, numSeries_, p0,
                            static_cast<int>(req.params.at(1)),
                            static_cast<int>(req.params.at(2)), CT_MA_EMA, 0, 0),
            "ct_macd_batch_d");
      break;
    case Family::STOCH:
      check(ct_stochastic_batch_d(matrix_device(Column::High),
                                  matrix_device(Column::Low),
                                  matrix_device(Column::Close), lines[0],
                                  lines[1], n, numSeries_, p0,
                                  static_cast<int>(req.params.at(1)), 0, 0),
            "ct_stochastic_batch_d");
      break;
    default:
      throw std::invalid_argument("tacuda batch backend: unhandled family");
    }
  }

  // One download per output line (the whole numSeries x n matrix), then a
  // host-side scatter into the per-symbol buffers — numSeries small PCIe
  // transfers per line would be latency-bound.
  staging_.resize(static_cast<std::size_t>(elems));
  for (int k = 0; k < nOut; ++k) {
    check(ct_device_download(staging_.data(),
                             lines[static_cast<std::size_t>(k)], elems, 0),
          "download");
    check(ct_device_sync(0), "stream sync");
    for (int s = 0; s < numSeries_; ++s) {
      std::copy_n(staging_.data() + static_cast<std::size_t>(s) * n, n,
                  outs[static_cast<std::size_t>(s)][static_cast<std::size_t>(k)]);
    }
  }

  // Align warm-ups with TA-Lib conventions (see TacudaBackend::compute).
  auto mask = [&](int line, int firstValid) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (auto& symbolOuts : outs) {
      float* dst = symbolOuts[static_cast<std::size_t>(line)];
      for (int i = 0; i < n && i < firstValid; ++i) dst[i] = nan;
    }
  };
  if (req.family == Family::MACD) {
    const int firstValid =
        static_cast<int>(req.params.at(1)) + static_cast<int>(req.params.at(2)) - 2;
    for (int k = 0; k < 3; ++k) mask(k, firstValid);
  } else if (req.family == Family::STOCH) {
    const int firstValid = p0 + 2 * (static_cast<int>(req.params.at(1)) - 1) - 1;
    mask(0, firstValid);
    mask(1, firstValid);
  }
  // (ct_trange reports NaN at bar 0 natively since 0.5.0 — no mask needed.)
}

} // namespace topt
