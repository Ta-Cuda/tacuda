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

namespace {

std::int64_t single_timestamp(const std::string& file, const std::string& stamp) {
  Series s = load_csv(write_temp(file, "timestamp,open,high,low,close\n" + stamp + ",1,2,0.5,1.5\n"));
  return s.ts.at(0);
}

void expect_rejected_at_line(const std::string& file, const std::string& csv, const char* line) {
  try {
    load_csv(write_temp(file, csv));
    ADD_FAILURE() << csv << " was accepted";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find(line), std::string::npos) << e.what();
  }
}

} // namespace

TEST(CsvTimestamps, SameInstantInAnyOffsetIsOneTimestamp) {
  const std::int64_t utc = single_timestamp("ts_utc.csv", "2024-03-01T00:00:00Z");
  EXPECT_EQ(single_timestamp("ts_plus.csv", "2024-03-01T03:00:00+03:00"), utc);
  EXPECT_EQ(single_timestamp("ts_compact.csv", "2024-03-01T03:00:00+0300"), utc);
  EXPECT_EQ(single_timestamp("ts_hours.csv", "2024-03-01 03:00:00+03"), utc);
  EXPECT_EQ(single_timestamp("ts_space.csv", "2024-03-01 03:00:00 +03:00"), utc);
  EXPECT_EQ(single_timestamp("ts_minus.csv", "2024-02-29T19:00:00-05:00"), utc);
  EXPECT_EQ(single_timestamp("ts_naive.csv", "2024-03-01 00:00:00"), utc);
}

TEST(CsvTimestamps, FractionsAreFlooredAndLeapDaysAreValid) {
  EXPECT_EQ(single_timestamp("ts_frac.csv", "2024-02-29T00:00:00.750Z"),
            single_timestamp("ts_whole.csv", "2024-02-29T00:00:00Z"));
}

TEST(CsvTimestamps, ImpossibleOrUnsupportedSpellingsAreRejected) {
  const char* bad[] = {
      "2024-03-01T00:00:00 EST",     "2024-02-30",
      "2023-02-29",                  "2024-13-01",
      "2024-01-01 24:00:00",         "2024-01-01 10:60:00",
      "2024-01-01 10:00:60",         "2024-01-01T00:00:00+15:00",
      "2024-01-01T00:00:00+03:00x",  "2024-01-01T00:00:00.",
  };
  for (const char* stamp : bad) {
    EXPECT_THROW(single_timestamp("ts_bad.csv", stamp), std::runtime_error) << stamp;
  }
}

TEST(CsvTimestamps, RowsMustBeStrictlyIncreasing) {
  const std::string header = "timestamp,open,high,low,close\n";
  expect_rejected_at_line("ts_desc.csv",
                          header + "1700000300,1,2,0.5,1.5\n1700000000,1,2,0.5,1.5\n", ":3:");
  expect_rejected_at_line("ts_dup.csv",
                          header + "1700000000,1,2,0.5,1.5\n1700000000,1,2,0.5,1.5\n", ":3:");
  // Two spellings of one instant are duplicates once the offset is honoured.
  expect_rejected_at_line("ts_alias.csv",
                          header + "2024-03-01T03:00:00+03:00,1,2,0.5,1.5\n"
                                   "2024-03-01T00:00:00Z,1,2,0.5,1.5\n", ":3:");
}

TEST(CsvValues, RejectsValuesOutsideFloat32) {
  expect_rejected_at_line("big.csv", "open,high,low,close\n1,1e50,0.5,1.5\n", ":2:");
  expect_rejected_at_line("bigvol.csv", "open,high,low,close,volume\n1,2,0.5,1.5,1e50\n", ":2:");
}

TEST(CsvValues, BarGeometryIsEnforced) {
  expect_rejected_at_line("open.csv", "open,high,low,close\n3,2,0.5,1.5\n", ":2:");
  expect_rejected_at_line("close.csv", "open,high,low,close\n1,2,0.5,0.1\n", ":2:");
  expect_rejected_at_line("vol.csv", "open,high,low,close,volume\n1,2,0.5,1.5,-1\n", ":2:");
}

TEST(CsvValues, NegativePricesRemainValid) {
  Series s = load_csv(write_temp("neg.csv", "open,high,low,close\n-3,-1,-5,-2\n"));
  ASSERT_EQ(s.size(), 1);
  EXPECT_FLOAT_EQ(s.close[0], -2.0f);
}
