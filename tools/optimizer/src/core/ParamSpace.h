/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OPT_CORE_PARAMSPACE_H
#define TACUDA_OPT_CORE_PARAMSPACE_H

#include <string>
#include <vector>

#include "core/Types.h"

namespace topt {

// One axis of the optimization grid: an explicit list of values for a name.
struct GridAxis {
  std::string name;
  std::vector<double> values;
};

// Parses "name=from..to:step" (grid) or "name=v1,v2,v3" (list).
// Throws std::invalid_argument with a readable message on bad syntax.
GridAxis parse_grid_spec(const std::string& spec);

// Parses "name=value". Throws on bad syntax.
std::pair<std::string, double> parse_set_spec(const std::string& spec);

// Per declared parameter (strategy order): the list of values it can take —
// the grid axis when one is given, otherwise a single value from --set or
// the declared default. Validates that every axis/set name is declared.
// This is the search space shared by grid enumeration and the genetic
// optimizer (genes are indices into these lists).
std::vector<std::vector<double>> build_value_lists(
    const std::vector<ParamDef>& defs, const std::vector<GridAxis>& axes,
    const std::vector<std::pair<std::string, double>>& sets);

// Assembles a ParamSet from per-parameter value indices (declaration order).
// Integer parameters are rounded.
ParamSet materialize(const std::vector<ParamDef>& defs,
                     const std::vector<std::vector<double>>& valueLists,
                     const std::vector<std::size_t>& indices);

// Builds the full variant list for a strategy:
// - axes from --grid specs (must match declared parameter names),
// - fixed overrides from --set,
// - remaining parameters take their declared defaults.
// Enumeration order is deterministic: parameters keep the strategy's
// declaration order, the first declared gridded parameter varies slowest.
std::vector<ParamSet> enumerate_variants(const std::vector<ParamDef>& defs,
                                         const std::vector<GridAxis>& axes,
                                         const std::vector<std::pair<std::string, double>>& sets);

} // namespace topt

#endif // TACUDA_OPT_CORE_PARAMSPACE_H
