/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "core/ParamSpace.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace topt {

namespace {

double parse_number(const std::string& text, const std::string& context) {
  char* end = nullptr;
  double v = std::strtod(text.c_str(), &end);
  if (!end || *end != '\0' || text.empty()) {
    throw std::invalid_argument("bad number '" + text + "' in " + context);
  }
  return v;
}

} // namespace

GridAxis parse_grid_spec(const std::string& spec) {
  auto eq = spec.find('=');
  if (eq == std::string::npos || eq == 0) {
    throw std::invalid_argument("grid spec must look like name=from..to:step, got '" + spec + "'");
  }
  GridAxis axis;
  axis.name = spec.substr(0, eq);
  std::string body = spec.substr(eq + 1);

  auto dots = body.find("..");
  if (dots != std::string::npos) {
    std::string fromStr = body.substr(0, dots);
    std::string rest = body.substr(dots + 2);
    double step = 1.0;
    auto colon = rest.find(':');
    std::string toStr = rest;
    if (colon != std::string::npos) {
      toStr = rest.substr(0, colon);
      step = parse_number(rest.substr(colon + 1), spec);
    }
    double from = parse_number(fromStr, spec);
    double to = parse_number(toStr, spec);
    if (step <= 0.0) {
      throw std::invalid_argument("grid step must be > 0 in '" + spec + "'");
    }
    if (to < from) {
      throw std::invalid_argument("grid upper bound below lower bound in '" + spec + "'");
    }
    // Half-step tolerance so from..to:step includes `to` despite fp error.
    for (double v = from; v <= to + step * 0.5; v += step) {
      axis.values.push_back(v);
    }
  } else {
    // Comma-separated list of explicit values.
    std::size_t pos = 0;
    while (pos <= body.size()) {
      auto comma = body.find(',', pos);
      std::string tok = comma == std::string::npos ? body.substr(pos)
                                                   : body.substr(pos, comma - pos);
      axis.values.push_back(parse_number(tok, spec));
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  }
  if (axis.values.empty()) {
    throw std::invalid_argument("grid spec produced no values: '" + spec + "'");
  }
  return axis;
}

std::pair<std::string, double> parse_set_spec(const std::string& spec) {
  auto eq = spec.find('=');
  if (eq == std::string::npos || eq == 0) {
    throw std::invalid_argument("set spec must look like name=value, got '" + spec + "'");
  }
  return {spec.substr(0, eq), parse_number(spec.substr(eq + 1), spec)};
}

std::vector<std::vector<double>> build_value_lists(
    const std::vector<ParamDef>& defs, const std::vector<GridAxis>& axes,
    const std::vector<std::pair<std::string, double>>& sets) {
  auto find_def = [&](const std::string& name) -> const ParamDef* {
    for (const auto& d : defs) {
      if (d.name == name) return &d;
    }
    return nullptr;
  };
  for (const auto& a : axes) {
    if (!find_def(a.name)) {
      throw std::invalid_argument("unknown grid parameter '" + a.name + "'");
    }
  }
  for (const auto& s : sets) {
    if (!find_def(s.first)) {
      throw std::invalid_argument("unknown set parameter '" + s.first + "'");
    }
  }

  std::vector<std::vector<double>> valueLists;
  valueLists.reserve(defs.size());
  for (const auto& d : defs) {
    const GridAxis* axis = nullptr;
    for (const auto& a : axes) {
      if (a.name == d.name) axis = &a;
    }
    if (axis) {
      valueLists.push_back(axis->values);
      continue;
    }
    const std::pair<std::string, double>* fixed = nullptr;
    for (const auto& s : sets) {
      if (s.first == d.name) fixed = &s;
    }
    valueLists.push_back({fixed ? fixed->second : d.defval});
  }
  return valueLists;
}

ParamSet materialize(const std::vector<ParamDef>& defs,
                     const std::vector<std::vector<double>>& valueLists,
                     const std::vector<std::size_t>& indices) {
  ParamSet ps;
  ps.values.reserve(defs.size());
  for (std::size_t i = 0; i < defs.size(); ++i) {
    double v = valueLists[i][indices[i]];
    if (defs[i].integer) v = std::round(v);
    ps.values.emplace_back(defs[i].name, v);
  }
  return ps;
}

std::vector<ParamSet> enumerate_variants(
    const std::vector<ParamDef>& defs, const std::vector<GridAxis>& axes,
    const std::vector<std::pair<std::string, double>>& sets) {
  std::vector<std::vector<double>> valueLists = build_value_lists(defs, axes, sets);

  std::size_t total = 1;
  for (const auto& list : valueLists) {
    total *= list.size();
  }

  std::vector<ParamSet> out;
  out.reserve(total);
  std::vector<std::size_t> idx(defs.size(), 0);
  for (std::size_t k = 0; k < total; ++k) {
    out.push_back(materialize(defs, valueLists, idx));
    // Increment mixed-radix counter; last declared parameter varies fastest.
    for (std::size_t i = defs.size(); i-- > 0;) {
      if (++idx[i] < valueLists[i].size()) break;
      idx[i] = 0;
    }
  }
  return out;
}

} // namespace topt
