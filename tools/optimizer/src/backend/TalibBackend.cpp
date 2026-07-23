/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "backend/TalibBackend.h"

#include "backend/ClubCpu.h"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace topt {

namespace {

void* open_library(const char* name) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(LoadLibraryA(name));
#else
  return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void close_library(void* lib) {
  if (!lib) return;
#if defined(_WIN32)
  FreeLibrary(reinterpret_cast<HMODULE>(lib));
#else
  dlclose(lib);
#endif
}

void* find_symbol(void* lib, const char* name) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(
      GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
  return dlsym(lib, name);
#endif
}

std::vector<std::string> candidate_names() {
  std::vector<std::string> names;
  for (const char* env : {"TACUDA_OPT_TALIB", "TALIB_LIBRARY"}) {
    const char* v = std::getenv(env);
    if (v && *v) names.push_back(v);
  }
#if defined(_WIN32)
  names.insert(names.end(), {"ta-lib.dll", "ta_lib.dll", "libta-lib.dll", "talib.dll"});
#elif defined(__APPLE__)
  names.insert(names.end(), {"libta-lib.dylib", "libta_lib.dylib"});
#else
  names.insert(names.end(), {"libta-lib.so.0", "libta-lib.so", "libta_lib.so.0",
                             "libta_lib.so", "libta_lib.so.0.0.0"});
#endif
  return names;
}

void* load_talib(std::string& why) {
  std::string tried;
  for (const auto& name : candidate_names()) {
    if (void* lib = open_library(name.c_str())) {
      // Some distros ship the library without TA_Initialize being mandatory
      // to call, but resolving it proves this really is TA-Lib.
      if (find_symbol(lib, "TA_Initialize")) {
        using InitFn = int (*)(void);
        auto init = reinterpret_cast<InitFn>(find_symbol(lib, "TA_Initialize"));
        init(); // idempotent, returns TA_ALREADY_INITIALIZED on repeat calls
        why.clear();
        return lib;
      }
      close_library(lib);
    }
    if (!tried.empty()) tried += ", ";
    tried += name;
  }
  why = "TA-Lib shared library not found (tried: " + tried +
        "); set TACUDA_OPT_TALIB or TALIB_LIBRARY to the full path";
  return nullptr;
}

void check_ta(int rc, const char* what) {
  if (rc != 0) {
    throw std::runtime_error(std::string("talib backend: ") + what +
                             " failed with TA_RetCode=" + std::to_string(rc));
  }
}

// Writes a TA-Lib result (outBegIdx offset, nElem values) into a float line
// with a NaN warm-up prefix, matching tacuda's output convention.
void scatter(const std::vector<double>& tmp, int begIdx, int nElem, float* out, int n) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (int i = 0; i < n; ++i) out[i] = nan;
  for (int i = 0; i < nElem && begIdx + i < n; ++i) {
    out[begIdx + i] = static_cast<float>(tmp[static_cast<std::size_t>(i)]);
  }
}

} // namespace

bool TalibBackend::available(std::string& why) {
  static std::string cachedWhy;
  static void* probe = load_talib(cachedWhy);
  why = cachedWhy;
  return probe != nullptr;
}

TalibBackend::TalibBackend(const Series& bars) : bars_(bars) {
  if (bars_.size() <= 0) {
    throw std::invalid_argument("talib backend: empty series");
  }
  std::string why;
  lib_ = load_talib(why);
  if (!lib_) {
    throw std::runtime_error("talib backend: " + why);
  }
  taSma_ = reinterpret_cast<UnaryFn>(find_symbol(lib_, "TA_SMA"));
  taEma_ = reinterpret_cast<UnaryFn>(find_symbol(lib_, "TA_EMA"));
  taRsi_ = reinterpret_cast<UnaryFn>(find_symbol(lib_, "TA_RSI"));
  taBbands_ = reinterpret_cast<BbandsFn>(find_symbol(lib_, "TA_BBANDS"));
  taAtr_ = reinterpret_cast<AtrFn>(find_symbol(lib_, "TA_ATR"));
  taWma_ = reinterpret_cast<UnaryFn>(find_symbol(lib_, "TA_WMA"));
  taTrima_ = reinterpret_cast<UnaryFn>(find_symbol(lib_, "TA_TRIMA"));
  taMom_ = reinterpret_cast<UnaryFn>(find_symbol(lib_, "TA_MOM"));
  taRoc_ = reinterpret_cast<UnaryFn>(find_symbol(lib_, "TA_ROC"));
  taLinearreg_ = reinterpret_cast<UnaryFn>(find_symbol(lib_, "TA_LINEARREG"));
  taLinearregSlope_ =
      reinterpret_cast<UnaryFn>(find_symbol(lib_, "TA_LINEARREG_SLOPE"));
  taMax_ = reinterpret_cast<UnaryFn>(find_symbol(lib_, "TA_MAX"));
  taMin_ = reinterpret_cast<UnaryFn>(find_symbol(lib_, "TA_MIN"));
  taStddev_ = reinterpret_cast<StdDevFn>(find_symbol(lib_, "TA_STDDEV"));
  taMacd_ = reinterpret_cast<MacdFn>(find_symbol(lib_, "TA_MACD"));
  taStoch_ = reinterpret_cast<StochFn>(find_symbol(lib_, "TA_STOCH"));
  taCci_ = reinterpret_cast<AtrFn>(find_symbol(lib_, "TA_CCI"));
  taAdx_ = reinterpret_cast<AtrFn>(find_symbol(lib_, "TA_ADX"));
  taPlusDi_ = reinterpret_cast<AtrFn>(find_symbol(lib_, "TA_PLUS_DI"));
  taMinusDi_ = reinterpret_cast<AtrFn>(find_symbol(lib_, "TA_MINUS_DI"));
  taObv_ = reinterpret_cast<ObvFn>(find_symbol(lib_, "TA_OBV"));
  taTrange_ = reinterpret_cast<TrangeFn>(find_symbol(lib_, "TA_TRANGE"));
  const bool ok = taSma_ && taEma_ && taRsi_ && taBbands_ && taAtr_ && taWma_ &&
                  taTrima_ && taMom_ && taRoc_ && taLinearreg_ &&
                  taLinearregSlope_ && taMax_ && taMin_ && taStddev_ &&
                  taMacd_ && taStoch_ && taCci_ && taAdx_ && taPlusDi_ &&
                  taMinusDi_ && taObv_ && taTrange_;
  if (!ok) {
    close_library(lib_);
    lib_ = nullptr;
    throw std::runtime_error("talib backend: required TA_* symbols missing");
  }
}

TalibBackend::~TalibBackend() {
  close_library(lib_);
}

const std::vector<double>& TalibBackend::column_f64(Column c) {
  auto idx = static_cast<std::size_t>(c);
  auto& cache = columnsF64_[idx];
  if (cache.empty()) {
    const std::vector<float>& src = bars_.column(c);
    cache.assign(src.begin(), src.end());
  }
  return cache;
}

void TalibBackend::compute(const IndicatorRequest& req, const std::vector<float*>& outs) {
  const int n = bars_.size();
  if (static_cast<int>(outs.size()) != family_outputs(req.family)) {
    throw std::invalid_argument("talib backend: wrong number of output buffers");
  }
  validate_chain(req);

  // Club-pack families (and their derived-series chains) have no TA-Lib
  // counterpart; they run on the shared plain-C++ implementation.
  {
    std::vector<float> chained;
    const float* clubIn = nullptr;
    if (family_single_input(req.family)) {
      if (req.chain) {
        chained = club::resolve_input(bars_, req, *this);
        clubIn = chained.data();
      } else {
        clubIn = bars_.column(req.input).data();
      }
    }
    if (club::compute_family(bars_, req, clubIn, outs.at(0))) {
      return;
    }
  }

  int begIdx = 0, nElem = 0;
  std::vector<double> tmp(static_cast<std::size_t>(n));

  switch (req.family) {
  case Family::SMA:
  case Family::EMA:
  case Family::RSI:
  case Family::WMA:
  case Family::TRIMA:
  case Family::MOM:
  case Family::ROC:
  case Family::LINEARREG:
  case Family::LINEARREG_SLOPE:
  case Family::MAX:
  case Family::MIN: {
    UnaryFn fn = nullptr;
    const char* what = "";
    switch (req.family) {
    case Family::SMA: fn = taSma_; what = "TA_SMA"; break;
    case Family::EMA: fn = taEma_; what = "TA_EMA"; break;
    case Family::RSI: fn = taRsi_; what = "TA_RSI"; break;
    case Family::WMA: fn = taWma_; what = "TA_WMA"; break;
    case Family::TRIMA: fn = taTrima_; what = "TA_TRIMA"; break;
    case Family::MOM: fn = taMom_; what = "TA_MOM"; break;
    case Family::ROC: fn = taRoc_; what = "TA_ROC"; break;
    case Family::LINEARREG: fn = taLinearreg_; what = "TA_LINEARREG"; break;
    case Family::LINEARREG_SLOPE:
      fn = taLinearregSlope_; what = "TA_LINEARREG_SLOPE"; break;
    case Family::MAX: fn = taMax_; what = "TA_MAX"; break;
    case Family::MIN: fn = taMin_; what = "TA_MIN"; break;
    default: break;
    }
    check_ta(fn(0, n - 1, column_f64(req.input).data(),
                static_cast<int>(req.params.at(0)), &begIdx, &nElem, tmp.data()),
             what);
    scatter(tmp, begIdx, nElem, outs[0], n);
    break;
  }
  case Family::BBANDS: {
    std::vector<double> mid(static_cast<std::size_t>(n)), low(static_cast<std::size_t>(n));
    // MAType 0 == SMA, the same middle-band definition tacuda's ct_bbands uses.
    check_ta(taBbands_(0, n - 1, column_f64(req.input).data(),
                       static_cast<int>(req.params.at(0)), req.params.at(1),
                       req.params.at(2), 0, &begIdx, &nElem, tmp.data(),
                       mid.data(), low.data()),
             "TA_BBANDS");
    scatter(tmp, begIdx, nElem, outs[0], n);
    scatter(mid, begIdx, nElem, outs[1], n);
    scatter(low, begIdx, nElem, outs[2], n);
    break;
  }
  case Family::ATR:
  case Family::CCI:
  case Family::ADX:
  case Family::PLUS_DI:
  case Family::MINUS_DI: {
    AtrFn fn = nullptr;
    const char* what = "";
    switch (req.family) {
    case Family::ATR: fn = taAtr_; what = "TA_ATR"; break;
    case Family::CCI: fn = taCci_; what = "TA_CCI"; break;
    case Family::ADX: fn = taAdx_; what = "TA_ADX"; break;
    case Family::PLUS_DI: fn = taPlusDi_; what = "TA_PLUS_DI"; break;
    case Family::MINUS_DI: fn = taMinusDi_; what = "TA_MINUS_DI"; break;
    default: break;
    }
    check_ta(fn(0, n - 1, column_f64(Column::High).data(),
                column_f64(Column::Low).data(), column_f64(Column::Close).data(),
                static_cast<int>(req.params.at(0)), &begIdx, &nElem, tmp.data()),
             what);
    scatter(tmp, begIdx, nElem, outs[0], n);
    break;
  }
  case Family::STDDEV: {
    check_ta(taStddev_(0, n - 1, column_f64(req.input).data(),
                       static_cast<int>(req.params.at(0)), 1.0, &begIdx, &nElem,
                       tmp.data()),
             "TA_STDDEV");
    scatter(tmp, begIdx, nElem, outs[0], n);
    break;
  }
  case Family::MACD: {
    std::vector<double> sig(static_cast<std::size_t>(n)),
        hist(static_cast<std::size_t>(n));
    check_ta(taMacd_(0, n - 1, column_f64(req.input).data(),
                     static_cast<int>(req.params.at(0)),
                     static_cast<int>(req.params.at(1)),
                     static_cast<int>(req.params.at(2)), &begIdx, &nElem,
                     tmp.data(), sig.data(), hist.data()),
             "TA_MACD");
    scatter(tmp, begIdx, nElem, outs[0], n);
    scatter(sig, begIdx, nElem, outs[1], n);
    scatter(hist, begIdx, nElem, outs[2], n);
    break;
  }
  case Family::STOCH: {
    std::vector<double> dLine(static_cast<std::size_t>(n));
    // fastK = kPeriod; slowK smoothing = dPeriod SMA; slowD = dPeriod SMA,
    // mirroring tacuda's ct_stochastic semantics.
    check_ta(taStoch_(0, n - 1, column_f64(Column::High).data(),
                      column_f64(Column::Low).data(),
                      column_f64(Column::Close).data(),
                      static_cast<int>(req.params.at(0)),
                      static_cast<int>(req.params.at(1)), 0,
                      static_cast<int>(req.params.at(1)), 0, &begIdx, &nElem,
                      tmp.data(), dLine.data()),
             "TA_STOCH");
    scatter(tmp, begIdx, nElem, outs[0], n);
    scatter(dLine, begIdx, nElem, outs[1], n);
    break;
  }
  case Family::OBV: {
    check_ta(taObv_(0, n - 1, column_f64(Column::Close).data(),
                    column_f64(Column::Volume).data(), &begIdx, &nElem,
                    tmp.data()),
             "TA_OBV");
    scatter(tmp, begIdx, nElem, outs[0], n);
    break;
  }
  case Family::TRANGE: {
    check_ta(taTrange_(0, n - 1, column_f64(Column::High).data(),
                       column_f64(Column::Low).data(),
                       column_f64(Column::Close).data(), &begIdx, &nElem,
                       tmp.data()),
             "TA_TRANGE");
    scatter(tmp, begIdx, nElem, outs[0], n);
    break;
  }
  default:
    // Club families returned above; anything else is a wiring bug.
    throw std::invalid_argument("talib backend: unhandled family");
  }
}

} // namespace topt
