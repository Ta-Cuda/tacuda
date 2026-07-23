/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_BACKEND_TALIBBACKEND_H
#define TACUDA_OPT_BACKEND_TALIBBACKEND_H

#include <array>
#include <string>
#include <vector>

#include "backend/Backend.h"

namespace topt {

// CPU fallback backend: TA-Lib loaded dynamically at runtime
// (LoadLibrary/dlopen), so the optimizer binary carries no hard TA-Lib
// dependency. Used when no CUDA device is available or when the user
// forces --backend talib. Library lookup order: TACUDA_OPT_TALIB and
// TALIB_LIBRARY environment variables, then platform-typical names
// (ta-lib.dll / libta-lib.so / libta_lib.so...).
class TalibBackend : public IIndicatorBackend {
public:
  explicit TalibBackend(const Series& bars);
  ~TalibBackend() override;

  TalibBackend(const TalibBackend&) = delete;
  TalibBackend& operator=(const TalibBackend&) = delete;

  const char* name() const override { return "talib"; }
  void compute(const IndicatorRequest& req, const std::vector<float*>& outs) override;

  // Attempts to load TA-Lib; on failure returns false and fills `why`.
  static bool available(std::string& why);

private:
  const std::vector<double>& column_f64(Column c);

  const Series& bars_;
  void* lib_ = nullptr;
  std::array<std::vector<double>, 5> columnsF64_{}; // lazy float->double copies

  // Resolved TA-Lib entrypoints (single-input share a signature).
  using UnaryFn = int (*)(int, int, const double*, int, int*, int*, double*);
  using BbandsFn = int (*)(int, int, const double*, int, double, double, int,
                           int*, int*, double*, double*, double*);
  using AtrFn = int (*)(int, int, const double*, const double*, const double*,
                        int, int*, int*, double*);
  using StdDevFn = int (*)(int, int, const double*, int, double, int*, int*,
                           double*);
  using MacdFn = int (*)(int, int, const double*, int, int, int, int*, int*,
                         double*, double*, double*);
  using StochFn = int (*)(int, int, const double*, const double*, const double*,
                          int, int, int, int, int, int*, int*, double*, double*);
  using ObvFn = int (*)(int, int, const double*, const double*, int*, int*,
                        double*);
  using TrangeFn = int (*)(int, int, const double*, const double*, const double*,
                           int*, int*, double*);
  UnaryFn taSma_ = nullptr;
  UnaryFn taEma_ = nullptr;
  UnaryFn taRsi_ = nullptr;
  BbandsFn taBbands_ = nullptr;
  AtrFn taAtr_ = nullptr;
  UnaryFn taWma_ = nullptr;
  UnaryFn taTrima_ = nullptr;
  UnaryFn taMom_ = nullptr;
  UnaryFn taRoc_ = nullptr;
  UnaryFn taLinearreg_ = nullptr;
  UnaryFn taLinearregSlope_ = nullptr;
  UnaryFn taMax_ = nullptr;
  UnaryFn taMin_ = nullptr;
  StdDevFn taStddev_ = nullptr;
  MacdFn taMacd_ = nullptr;
  StochFn taStoch_ = nullptr;
  AtrFn taCci_ = nullptr;
  AtrFn taAdx_ = nullptr;
  AtrFn taPlusDi_ = nullptr;
  AtrFn taMinusDi_ = nullptr;
  ObvFn taObv_ = nullptr;
  TrangeFn taTrange_ = nullptr;
};

} // namespace topt

#endif // TACUDA_OPT_BACKEND_TALIBBACKEND_H
