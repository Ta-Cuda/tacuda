/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <gtest/gtest.h>

#include "core/ParamSpace.h"

using namespace topt;

TEST(ParamSpace, ParsesRangeSpec) {
  GridAxis axis = parse_grid_spec("fast=5..20:5");
  EXPECT_EQ(axis.name, "fast");
  ASSERT_EQ(axis.values.size(), 4u);
  EXPECT_DOUBLE_EQ(axis.values[0], 5.0);
  EXPECT_DOUBLE_EQ(axis.values[3], 20.0);
}

TEST(ParamSpace, RangeIncludesUpperBoundDespiteFpError) {
  GridAxis axis = parse_grid_spec("w=0.5..4.0:0.25");
  ASSERT_EQ(axis.values.size(), 15u);
  EXPECT_NEAR(axis.values.back(), 4.0, 1e-9);
}

TEST(ParamSpace, ParsesValueList) {
  GridAxis axis = parse_grid_spec("period=7,14,21");
  ASSERT_EQ(axis.values.size(), 3u);
  EXPECT_DOUBLE_EQ(axis.values[1], 14.0);
}

TEST(ParamSpace, DefaultStepIsOne) {
  GridAxis axis = parse_grid_spec("p=1..3");
  ASSERT_EQ(axis.values.size(), 3u);
}

TEST(ParamSpace, RejectsBadSpecs) {
  EXPECT_THROW(parse_grid_spec("fast"), std::invalid_argument);
  EXPECT_THROW(parse_grid_spec("=5..10"), std::invalid_argument);
  EXPECT_THROW(parse_grid_spec("fast=10..5"), std::invalid_argument);
  EXPECT_THROW(parse_grid_spec("fast=5..10:0"), std::invalid_argument);
  EXPECT_THROW(parse_grid_spec("fast=5..x"), std::invalid_argument);
}

TEST(ParamSpace, ParsesSetSpec) {
  auto kv = parse_set_spec("slow=48");
  EXPECT_EQ(kv.first, "slow");
  EXPECT_DOUBLE_EQ(kv.second, 48.0);
  EXPECT_THROW(parse_set_spec("slow"), std::invalid_argument);
}

TEST(ParamSpace, EnumeratesCartesianProductInDeclaredOrder) {
  std::vector<ParamDef> defs = {
      {"fast", 12, 2, 100, 1, true},
      {"slow", 48, 5, 250, 1, true},
  };
  std::vector<GridAxis> axes = {
      parse_grid_spec("fast=5..10:5"),
      parse_grid_spec("slow=20..30:10"),
  };
  auto variants = enumerate_variants(defs, axes, {});
  ASSERT_EQ(variants.size(), 4u);
  // First declared parameter varies slowest, values ascending.
  EXPECT_EQ(variants[0].to_string(), "fast=5 slow=20");
  EXPECT_EQ(variants[1].to_string(), "fast=5 slow=30");
  EXPECT_EQ(variants[2].to_string(), "fast=10 slow=20");
  EXPECT_EQ(variants[3].to_string(), "fast=10 slow=30");
}

TEST(ParamSpace, UsesDefaultsAndSetsForUngriddedParams) {
  std::vector<ParamDef> defs = {
      {"period", 14, 2, 100, 1, true},
      {"low", 30, 5, 50, 1, true},
      {"high", 70, 50, 95, 1, true},
  };
  auto variants = enumerate_variants(defs, {parse_grid_spec("period=7,14")},
                                     {{"high", 80.0}});
  ASSERT_EQ(variants.size(), 2u);
  EXPECT_DOUBLE_EQ(variants[0].get("low"), 30.0);  // default
  EXPECT_DOUBLE_EQ(variants[0].get("high"), 80.0); // --set override
}

TEST(ParamSpace, RejectsUnknownParameterNames) {
  std::vector<ParamDef> defs = {{"fast", 12, 2, 100, 1, true}};
  EXPECT_THROW(enumerate_variants(defs, {parse_grid_spec("nope=1..2")}, {}),
               std::invalid_argument);
  EXPECT_THROW(enumerate_variants(defs, {}, {{"nope", 1.0}}), std::invalid_argument);
}

TEST(ParamSpace, RoundsIntegerParams) {
  std::vector<ParamDef> defs = {{"period", 14, 2, 100, 1, true}};
  auto variants = enumerate_variants(defs, {parse_grid_spec("period=7.4,7.6")}, {});
  EXPECT_DOUBLE_EQ(variants[0].get("period"), 7.0);
  EXPECT_DOUBLE_EQ(variants[1].get("period"), 8.0);
}
