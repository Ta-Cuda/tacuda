/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include "data/CsvLoader.h"
#include "data/Synthetic.h"

using namespace topt;

namespace {

std::string write_temp(const std::string& name, const std::string& content) {
  std::string path = ::testing::TempDir() + name;
  std::ofstream f(path, std::ios::binary);
  f << content;
  return path;
}

} // namespace

TEST(CsvLoader, ParsesHeaderedFile) {
  std::string path = write_temp("headered.csv",
                                "timestamp,open,high,low,close,volume\n"
                                "2024-01-02 09:30:00,100,101,99,100.5,1200\n"
                                "2024-01-02 09:35:00,100.5,102,100,101.5,900\n");
  Series s = load_csv(path);
  ASSERT_EQ(s.size(), 2);
  EXPECT_FLOAT_EQ(s.open[0], 100.0f);
  EXPECT_FLOAT_EQ(s.close[1], 101.5f);
  EXPECT_FLOAT_EQ(s.volume[0], 1200.0f);
  EXPECT_EQ(s.ts[1] - s.ts[0], 300);
}

TEST(CsvLoader, ParsesHeaderlessOhlcv) {
  std::string path = write_temp("headerless.csv",
                                "100,101,99,100.5,1200\n"
                                "100.5,102,100,101.5,900\n");
  Series s = load_csv(path);
  ASSERT_EQ(s.size(), 2);
  EXPECT_FLOAT_EQ(s.high[1], 102.0f);
  EXPECT_FLOAT_EQ(s.volume[1], 900.0f);
}

TEST(CsvLoader, ParsesHeaderlessWithUnixTimestamp) {
  std::string path = write_temp("unix_ts.csv",
                                "1700000000,100,101,99,100.5,10\n"
                                "1700000300,101,102,100,101.5,11\n");
  Series s = load_csv(path);
  ASSERT_EQ(s.size(), 2);
  EXPECT_EQ(s.ts[0], 1700000000);
  EXPECT_FLOAT_EQ(s.open[0], 100.0f);
}

TEST(CsvLoader, ParsesSemicolonDelimiter) {
  std::string path = write_temp("semi.csv",
                                "open;high;low;close\n"
                                "1;2;0.5;1.5\n");
  Series s = load_csv(path);
  ASSERT_EQ(s.size(), 1);
  EXPECT_FLOAT_EQ(s.low[0], 0.5f);
  EXPECT_FLOAT_EQ(s.volume[0], 0.0f); // volume optional
}

TEST(CsvLoader, SkipsCommentsAndBlankLines) {
  std::string path = write_temp("comments.csv",
                                "# generated\n\nopen,high,low,close\n1,2,0.5,1.5\n\n");
  Series s = load_csv(path);
  EXPECT_EQ(s.size(), 1);
}

TEST(CsvLoader, RejectsHighBelowLow) {
  std::string path = write_temp("badhl.csv",
                                "open,high,low,close\n1,0.5,2,1.5\n");
  EXPECT_THROW(load_csv(path), std::runtime_error);
}

TEST(CsvLoader, RejectsGarbageValues) {
  std::string path = write_temp("garbage.csv",
                                "open,high,low,close\n1,x,0.5,1.5\n");
  EXPECT_THROW(load_csv(path), std::runtime_error);
}

TEST(CsvLoader, RejectsMissingFile) {
  EXPECT_THROW(load_csv(::testing::TempDir() + "no_such_file.csv"), std::runtime_error);
}

TEST(Synthetic, DeterministicForSameSeed) {
  Series a = generate_synthetic(500, 7);
  Series b = generate_synthetic(500, 7);
  Series c = generate_synthetic(500, 8);
  ASSERT_EQ(a.size(), 500);
  EXPECT_EQ(a.close, b.close);
  EXPECT_NE(a.close, c.close);
}

TEST(Synthetic, BarsAreConsistent) {
  Series s = generate_synthetic(1000, 42);
  for (int i = 0; i < s.size(); ++i) {
    auto idx = static_cast<std::size_t>(i);
    EXPECT_GE(s.high[idx], s.low[idx]);
    EXPECT_GE(s.high[idx], s.close[idx]);
    EXPECT_LE(s.low[idx], s.open[idx]);
    EXPECT_GT(s.close[idx], 0.0f);
  }
}
