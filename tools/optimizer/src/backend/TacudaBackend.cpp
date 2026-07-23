/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "backend/TacudaBackend.h"

#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include <tacuda.h>
#include <tacuda_device.h>

namespace topt {

namespace {

// Aligns a downloaded line with TA-Lib warm-up conventions when the tacuda
// kernel emits values earlier than TA-Lib would (e.g. the raw MACD line is
// defined from slow-1, but TA-Lib reports all three lines from
// slow+signal-2). Keeps every backend NaN-comparable.
void mask_warmup(float* line, int n, int firstValid) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (int i = 0; i < n && i < firstValid; ++i) line[i] = nan;
}

[[noreturn]] void raise(const std::string& what, ctStatus_t st) {
  std::string detail = ct_get_last_error();
  std::string msg = "tacuda backend: " + what + " failed: " + ct_status_string(st);
  if (!detail.empty()) msg += " (" + detail + ")";
  throw std::runtime_error(msg);
}

void check(ctStatus_t st, const char* what) {
  if (st != CT_STATUS_SUCCESS) raise(what, st);
}

// RAII device buffer for per-call outputs.
struct DeviceArray {
  float* ptr = nullptr;
  explicit DeviceArray(int n) : ptr(ct_device_alloc(n)) {
    if (!ptr) throw std::runtime_error("tacuda backend: device allocation failed");
  }
  ~DeviceArray() { ct_device_free(ptr); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
};

} // namespace

bool TacudaBackend::available() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
}

TacudaBackend::TacudaBackend(const Series& bars) : bars_(bars) {
  if (bars_.size() <= 0) {
    throw std::invalid_argument("tacuda backend: empty series");
  }
  if (!available()) {
    throw std::runtime_error("tacuda backend: no usable CUDA device");
  }
}

TacudaBackend::~TacudaBackend() {
  for (float* p : deviceColumns_) {
    if (p) ct_device_free(p);
  }
}

float* TacudaBackend::column_device(Column c) {
  auto idx = static_cast<std::size_t>(c);
  if (deviceColumns_[idx]) return deviceColumns_[idx];

  const std::vector<float>& host = bars_.column(c);
  int n = bars_.size();
  float* d = ct_device_alloc(n);
  if (!d) throw std::runtime_error("tacuda backend: device allocation failed");
  ctStatus_t st = ct_device_upload(d, host.data(), n, 0);
  if (st != CT_STATUS_SUCCESS) {
    ct_device_free(d);
    raise("column upload", st);
  }
  deviceColumns_[idx] = d;
  return d;
}

// Computes any single-output request (including club families and derived
// chains) into a device buffer — no host round-trip anywhere in the chain.
void TacudaBackend::compute_line_device(const IndicatorRequest& req,
                                        float* dOut) {
  const int n = bars_.size();
  const int p0 = req.params.empty() ? 0 : static_cast<int>(req.params.at(0));

  // Resolve the input line: OHLCV column, or the chained upstream line
  // computed recursively into a scratch device buffer.
  std::unique_ptr<DeviceArray> chained;
  const float* dIn = nullptr;
  if (family_single_input(req.family)) {
    if (req.chain) {
      chained = std::make_unique<DeviceArray>(n);
      compute_line_device(*req.chain, chained->ptr);
      dIn = chained->ptr;
    } else {
      dIn = column_device(req.input);
    }
  }

  switch (req.family) {
  case Family::SMA:
    check(ct_sma_d(dIn, dOut, n, p0, 0), "ct_sma_d");
    break;
  case Family::EMA:
    check(ct_ema_d(dIn, dOut, n, p0, 0), "ct_ema_d");
    break;
  case Family::RSI:
    check(ct_rsi_d(dIn, dOut, n, p0, 0), "ct_rsi_d");
    break;
  case Family::ATR:
    check(ct_atr_d(column_device(Column::High), column_device(Column::Low),
                   column_device(Column::Close), dOut, n, p0, 0.0f, 0),
          "ct_atr_d");
    break;
  case Family::WMA:
  case Family::TRIMA:
  case Family::MOM:
  case Family::ROC:
  case Family::STDDEV:
  case Family::LINEARREG:
  case Family::LINEARREG_SLOPE:
  case Family::MAX:
  case Family::MIN:
  case Family::MARKET_STRENGTH:
  case Family::TRENDSCORE:
  case Family::SKEW:
  case Family::KURTOSIS:
  case Family::PIVOT_HIGH:
  case Family::PIVOT_LOW:
  case Family::XAVERAGE:
  case Family::ROLLING_MEAN:
  case Family::ROLLING_MAX:
  case Family::ROLLING_MIN:
  case Family::ROLLING_STD: {
    using UnaryDeviceFn = ctStatus_t (*)(const float*, float*, int, int, cudaStream_t);
    UnaryDeviceFn fn = nullptr;
    const char* what = "";
    switch (req.family) {
    case Family::WMA: fn = ct_wma_d; what = "ct_wma_d"; break;
    case Family::TRIMA: fn = ct_trima_d; what = "ct_trima_d"; break;
    case Family::MOM: fn = ct_momentum_d; what = "ct_momentum_d"; break;
    case Family::ROC: fn = ct_roc_d; what = "ct_roc_d"; break;
    case Family::STDDEV: fn = ct_stddev_d; what = "ct_stddev_d"; break;
    case Family::LINEARREG: fn = ct_linearreg_d; what = "ct_linearreg_d"; break;
    case Family::LINEARREG_SLOPE:
      fn = ct_linearreg_slope_d; what = "ct_linearreg_slope_d"; break;
    case Family::MAX: fn = ct_max_d; what = "ct_max_d"; break;
    case Family::MIN: fn = ct_min_d; what = "ct_min_d"; break;
    case Family::MARKET_STRENGTH:
      fn = ct_market_strength_d; what = "ct_market_strength_d"; break;
    case Family::TRENDSCORE:
      fn = ct_trendscore_d; what = "ct_trendscore_d"; break;
    case Family::SKEW: fn = ct_skew_d; what = "ct_skew_d"; break;
    case Family::KURTOSIS: fn = ct_kurtosis_d; what = "ct_kurtosis_d"; break;
    case Family::PIVOT_HIGH:
      fn = ct_pivot_high_d; what = "ct_pivot_high_d"; break;
    case Family::PIVOT_LOW:
      fn = ct_pivot_low_d; what = "ct_pivot_low_d"; break;
    case Family::XAVERAGE: fn = ct_xaverage_d; what = "ct_xaverage_d"; break;
    case Family::ROLLING_MEAN:
      fn = ct_rolling_mean_d; what = "ct_rolling_mean_d"; break;
    case Family::ROLLING_MAX:
      fn = ct_rolling_max_d; what = "ct_rolling_max_d"; break;
    case Family::ROLLING_MIN:
      fn = ct_rolling_min_d; what = "ct_rolling_min_d"; break;
    case Family::ROLLING_STD:
      fn = ct_rolling_std_d; what = "ct_rolling_std_d"; break;
    default: break;
    }
    check(fn(dIn, dOut, n, p0, 0), what);
    break;
  }
  case Family::REFDEV:
    check(ct_refdev_d(dIn, dOut, n, p0, static_cast<int>(req.params.at(1)), 0),
          "ct_refdev_d");
    break;
  case Family::CCI:
  case Family::ADX:
  case Family::PLUS_DI:
  case Family::MINUS_DI: {
    using HlcDeviceFn =
        ctStatus_t (*)(const float*, const float*, const float*, float*, int, int,
                       cudaStream_t);
    HlcDeviceFn fn = nullptr;
    const char* what = "";
    switch (req.family) {
    case Family::CCI: fn = ct_cci_d; what = "ct_cci_d"; break;
    case Family::ADX: fn = ct_adx_d; what = "ct_adx_d"; break;
    case Family::PLUS_DI: fn = ct_plus_di_d; what = "ct_plus_di_d"; break;
    case Family::MINUS_DI: fn = ct_minus_di_d; what = "ct_minus_di_d"; break;
    default: break;
    }
    check(fn(column_device(Column::High), column_device(Column::Low),
             column_device(Column::Close), dOut, n, p0, 0), what);
    break;
  }
  case Family::OBV:
    check(ct_obv_d(column_device(Column::Close), column_device(Column::Volume),
                   dOut, n, 0), "ct_obv_d");
    break;
  case Family::TRANGE:
    check(ct_trange_d(column_device(Column::High), column_device(Column::Low),
                      column_device(Column::Close), dOut, n, 0), "ct_trange_d");
    break;
  case Family::ASI:
    check(ct_asi_d(column_device(Column::Open), column_device(Column::High),
                   column_device(Column::Low), column_device(Column::Close),
                   dOut, n, 0), "ct_asi_d");
    break;
  case Family::SAR_EL:
    check(ct_sar_el_d(column_device(Column::High), column_device(Column::Low),
                      column_device(Column::Close), dOut, n,
                      static_cast<float>(req.params.at(0)),
                      static_cast<float>(req.params.at(1)), 0), "ct_sar_el_d");
    break;
  case Family::OBTR:
    check(ct_obtr_d(column_device(Column::High), column_device(Column::Low),
                    column_device(Column::Close), dOut, n, 0), "ct_obtr_d");
    break;
  case Family::WOBV:
    check(ct_wobv_d(column_device(Column::Open), column_device(Column::High),
                    column_device(Column::Low), column_device(Column::Close),
                    column_device(Column::Volume), dOut, n, 0), "ct_wobv_d");
    break;
  case Family::HPI:
    check(ct_hpi_d(column_device(Column::High), column_device(Column::Low),
                   column_device(Column::Volume), dOut, n,
                   static_cast<float>(req.params.at(0)), 0), "ct_hpi_d");
    break;
  default:
    throw std::invalid_argument(
        "tacuda backend: family is not single-output");
  }
}

void TacudaBackend::compute(const IndicatorRequest& req, const std::vector<float*>& outs) {
  const int n = bars_.size();
  if (static_cast<int>(outs.size()) != family_outputs(req.family)) {
    throw std::invalid_argument("tacuda backend: wrong number of output buffers");
  }
  validate_chain(req);

  // Single-output families (incl. the club pack and derived chains) share
  // one device-side path; only the final line is downloaded. (Since 0.5.0
  // ct_trange_d itself reports NaN at bar 0, TA-Lib style — no mask needed.)
  if (family_outputs(req.family) == 1) {
    DeviceArray out(n);
    compute_line_device(req, out.ptr);
    check(ct_device_download(outs[0], out.ptr, n, 0), "download");
    check(ct_device_sync(0), "stream sync");
    return;
  }

  switch (req.family) {
  case Family::BBANDS: {
    DeviceArray upper(n), middle(n), lower(n);
    check(ct_bbands_d(column_device(req.input), upper.ptr, middle.ptr, lower.ptr, n,
                      static_cast<int>(req.params.at(0)),
                      static_cast<float>(req.params.at(1)),
                      static_cast<float>(req.params.at(2)), 0), "ct_bbands_d");
    check(ct_device_download(outs[0], upper.ptr, n, 0), "download");
    check(ct_device_download(outs[1], middle.ptr, n, 0), "download");
    check(ct_device_download(outs[2], lower.ptr, n, 0), "download");
    break;
  }
  case Family::MACD: {
    const int fast = static_cast<int>(req.params.at(0));
    const int slow = static_cast<int>(req.params.at(1));
    const int sig = static_cast<int>(req.params.at(2));
    DeviceArray macd(n), signal(n), hist(n);
    check(ct_macd_d(column_device(req.input), macd.ptr, signal.ptr, hist.ptr, n,
                    fast, slow, sig, CT_MA_EMA, 0), "ct_macd_d");
    check(ct_device_download(outs[0], macd.ptr, n, 0), "download");
    check(ct_device_download(outs[1], signal.ptr, n, 0), "download");
    check(ct_device_download(outs[2], hist.ptr, n, 0), "download");
    check(ct_device_sync(0), "stream sync");
    const int firstValid = slow + sig - 2; // TA-Lib MACD lookback
    for (int k = 0; k < 3; ++k) mask_warmup(outs[static_cast<std::size_t>(k)], n, firstValid);
    break;
  }
  case Family::STOCH: {
    const int kPeriod = static_cast<int>(req.params.at(0));
    const int dPeriod = static_cast<int>(req.params.at(1));
    DeviceArray slowK(n), slowD(n);
    check(ct_stochastic_d(column_device(Column::High), column_device(Column::Low),
                          column_device(Column::Close), slowK.ptr, slowD.ptr, n,
                          kPeriod, dPeriod, 0), "ct_stochastic_d");
    check(ct_device_download(outs[0], slowK.ptr, n, 0), "download");
    check(ct_device_download(outs[1], slowD.ptr, n, 0), "download");
    check(ct_device_sync(0), "stream sync");
    // TA-Lib reports %K and %D from the same index; tacuda's raw slowK
    // starts dPeriod-1 bars earlier.
    const int firstValid = kPeriod + 2 * (dPeriod - 1) - 1;
    mask_warmup(outs[0], n, firstValid);
    mask_warmup(outs[1], n, firstValid);
    break;
  }
  default:
    throw std::invalid_argument("tacuda backend: unhandled family");
  }

  check(ct_device_sync(0), "stream sync");
}

} // namespace topt
