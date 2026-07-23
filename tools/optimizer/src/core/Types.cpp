/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "core/Types.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace topt {

const char* to_string(Column c) {
  switch (c) {
  case Column::Open: return "open";
  case Column::High: return "high";
  case Column::Low: return "low";
  case Column::Close: return "close";
  case Column::Volume: return "volume";
  }
  return "?";
}

const char* to_string(Family f) {
  switch (f) {
  case Family::SMA: return "SMA";
  case Family::EMA: return "EMA";
  case Family::RSI: return "RSI";
  case Family::BBANDS: return "BBANDS";
  case Family::ATR: return "ATR";
  case Family::WMA: return "WMA";
  case Family::TRIMA: return "TRIMA";
  case Family::MACD: return "MACD";
  case Family::STOCH: return "STOCH";
  case Family::CCI: return "CCI";
  case Family::ADX: return "ADX";
  case Family::PLUS_DI: return "PLUS_DI";
  case Family::MINUS_DI: return "MINUS_DI";
  case Family::OBV: return "OBV";
  case Family::MOM: return "MOM";
  case Family::ROC: return "ROC";
  case Family::STDDEV: return "STDDEV";
  case Family::LINEARREG: return "LINEARREG";
  case Family::LINEARREG_SLOPE: return "LINEARREG_SLOPE";
  case Family::MAX: return "MAX";
  case Family::MIN: return "MIN";
  case Family::TRANGE: return "TRANGE";
  case Family::ASI: return "ASI";
  case Family::SAR_EL: return "SAR_EL";
  case Family::OBTR: return "OBTR";
  case Family::WOBV: return "WOBV";
  case Family::HPI: return "HPI";
  case Family::MARKET_STRENGTH: return "MARKET_STRENGTH";
  case Family::TRENDSCORE: return "TRENDSCORE";
  case Family::REFDEV: return "REFDEV";
  case Family::SKEW: return "SKEW";
  case Family::KURTOSIS: return "KURTOSIS";
  case Family::PIVOT_HIGH: return "PIVOT_HIGH";
  case Family::PIVOT_LOW: return "PIVOT_LOW";
  case Family::XAVERAGE: return "XAVERAGE";
  case Family::ROLLING_MEAN: return "ROLLING_MEAN";
  case Family::ROLLING_MAX: return "ROLLING_MAX";
  case Family::ROLLING_MIN: return "ROLLING_MIN";
  case Family::ROLLING_STD: return "ROLLING_STD";
  }
  return "?";
}

int family_outputs(Family f) {
  switch (f) {
  case Family::BBANDS:
  case Family::MACD:
    return 3;
  case Family::STOCH:
    return 2;
  default:
    return 1;
  }
}

bool family_single_input(Family f) {
  switch (f) {
  case Family::ATR:
  case Family::STOCH:
  case Family::CCI:
  case Family::ADX:
  case Family::PLUS_DI:
  case Family::MINUS_DI:
  case Family::OBV:
  case Family::TRANGE:
  case Family::ASI:
  case Family::SAR_EL:
  case Family::OBTR:
  case Family::WOBV:
  case Family::HPI:
    return false;
  default:
    return true;
  }
}

bool family_chainable(Family f) {
  switch (f) {
  case Family::MARKET_STRENGTH:
  case Family::TRENDSCORE:
  case Family::REFDEV:
  case Family::SKEW:
  case Family::KURTOSIS:
  case Family::PIVOT_HIGH:
  case Family::PIVOT_LOW:
  case Family::XAVERAGE:
  case Family::ROLLING_MEAN:
  case Family::ROLLING_MAX:
  case Family::ROLLING_MIN:
  case Family::ROLLING_STD:
    return true;
  default:
    return false;
  }
}

void validate_chain(const IndicatorRequest& req) {
  if (!req.chain) return;
  if (!family_chainable(req.family)) {
    throw std::invalid_argument(
        std::string("indicator request: family ") + to_string(req.family) +
        " cannot consume a chained input");
  }
  if (family_outputs(req.chain->family) != 1 || req.chainLine != 0) {
    throw std::invalid_argument(
        "indicator request: chained input must be the single output line of "
        "the upstream request");
  }
  validate_chain(*req.chain);
}

const std::vector<float>& Series::column(Column c) const {
  switch (c) {
  case Column::Open: return open;
  case Column::High: return high;
  case Column::Low: return low;
  case Column::Close: return close;
  case Column::Volume: return volume;
  }
  throw std::invalid_argument("Series::column: bad column");
}

double ParamSet::get(const std::string& name) const {
  for (const auto& kv : values) {
    if (kv.first == name) return kv.second;
  }
  throw std::out_of_range("ParamSet: missing parameter '" + name + "'");
}

int ParamSet::get_int(const std::string& name) const {
  return static_cast<int>(std::llround(get(name)));
}

bool ParamSet::has(const std::string& name) const {
  for (const auto& kv : values) {
    if (kv.first == name) return true;
  }
  return false;
}

void ParamSet::set(const std::string& name, double v) {
  for (auto& kv : values) {
    if (kv.first == name) {
      kv.second = v;
      return;
    }
  }
  values.emplace_back(name, v);
}

std::string ParamSet::to_string() const {
  std::ostringstream os;
  bool first = true;
  for (const auto& kv : values) {
    if (!first) os << ' ';
    first = false;
    double v = kv.second;
    os << kv.first << '=';
    if (v == std::floor(v) && std::fabs(v) < 1e15) {
      os << static_cast<long long>(v);
    } else {
      os << v;
    }
  }
  return os.str();
}

} // namespace topt
