/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "data/CsvLoader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace topt {

namespace {

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string trim(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && (std::isspace(static_cast<unsigned char>(s[e - 1])) || s[e - 1] == '\r')) --e;
  return s.substr(b, e - b);
}

std::vector<std::string> split_row(const std::string& line, char delim) {
  std::vector<std::string> out;
  std::size_t pos = 0;
  while (pos <= line.size()) {
    auto next = line.find(delim, pos);
    std::string tok = next == std::string::npos ? line.substr(pos)
                                                : line.substr(pos, next - pos);
    out.push_back(trim(tok));
    if (next == std::string::npos) break;
    pos = next + 1;
  }
  return out;
}

bool parse_double(const std::string& s, double& out) {
  if (s.empty()) return false;
  char* end = nullptr;
  out = std::strtod(s.c_str(), &end);
  return end && *end == '\0';
}

// Days since 1970-01-01 for a civil date (Howard Hinnant's algorithm),
// avoids timegm/_mkgmtime portability differences.
std::int64_t days_from_civil(int y, int m, int d) {
  y -= m <= 2;
  const int era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(y - era * 400);
  const unsigned doy = (153u * static_cast<unsigned>(m + (m > 2 ? -3 : 9)) + 2u) / 5u +
                       static_cast<unsigned>(d) - 1u;
  const unsigned doe = yoe * 365u + yoe / 4u - yoe / 100u + doy;
  return static_cast<std::int64_t>(era) * 146097 + static_cast<std::int64_t>(doe) - 719468;
}

bool read_int(const std::string& s, std::size_t& i, int maxDigits, int& out) {
  int digits = 0, value = 0;
  while (i < s.size() && digits < maxDigits && std::isdigit(static_cast<unsigned char>(s[i]))) {
    value = value * 10 + (s[i] - '0');
    ++i;
    ++digits;
  }
  if (digits == 0) return false;
  out = value;
  return true;
}

int days_in_month(int year, int month) {
  static const int days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  const bool leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
  return month == 2 && leap ? 29 : days[month - 1];
}

// Optional UTC designator: Z, +HH, +HH:MM or +HHMM (seconds east of UTC).
bool parse_utc_offset(const std::string& s, std::size_t& i, int& seconds) {
  seconds = 0;
  if (i == s.size()) return true;
  if (s[i] == ' ' && ++i == s.size()) return false;
  if (s[i] == 'Z' || s[i] == 'z') return ++i == s.size();
  if (s[i] != '+' && s[i] != '-') return false;
  const int sign = s[i] == '-' ? -1 : 1;
  ++i;
  int hours = 0, minutes = 0;
  if (!read_int(s, i, 2, hours)) return false;
  if (i < s.size()) {
    if (s[i] == ':') ++i;
    if (!read_int(s, i, 2, minutes)) return false;
  }
  if (i != s.size() || minutes > 59 || hours * 60 + minutes > 14 * 60) return false;
  seconds = sign * (hours * 3600 + minutes * 60);
  return true;
}

// YYYY-MM-DD[( |T)HH[:MM[:SS[.fff]]]][offset]. A value without an offset is
// taken as UTC. Fractions are floored; a trailing token is an error so that an
// unsupported spelling cannot silently change the instant.
bool parse_iso8601(const std::string& s, std::int64_t& out) {
  std::size_t i = 0;
  int y = 0, mo = 0, d = 0, h = 0, mi = 0, sec = 0, offset = 0;
  if (!read_int(s, i, 4, y) || i >= s.size() || s[i++] != '-') return false;
  if (!read_int(s, i, 2, mo) || i >= s.size() || s[i++] != '-') return false;
  if (!read_int(s, i, 2, d)) return false;
  if (mo < 1 || mo > 12 || d < 1 || d > days_in_month(y, mo)) return false;
  if (i < s.size() && (s[i] == ' ' || s[i] == 'T' || s[i] == 't')) {
    ++i;
    if (!read_int(s, i, 2, h) || h > 23) return false;
    if (i < s.size() && s[i] == ':') {
      ++i;
      if (!read_int(s, i, 2, mi) || mi > 59) return false;
      if (i < s.size() && s[i] == ':') {
        ++i;
        if (!read_int(s, i, 2, sec) || sec > 59) return false;
        if (i < s.size() && s[i] == '.') {
          const std::size_t start = ++i;
          while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
          if (i == start) return false;
        }
      }
    }
  }
  if (!parse_utc_offset(s, i, offset)) return false;
  out = days_from_civil(y, mo, d) * 86400ll + h * 3600ll + mi * 60ll + sec - offset;
  return true;
}

// Parses timestamps: unix seconds, unix millis, ISO date / datetime.
bool parse_timestamp(const std::string& raw, std::int64_t& out) {
  if (raw.empty()) return false;
  bool numeric = true;
  for (char c : raw) {
    if (!std::isdigit(static_cast<unsigned char>(c)) && c != '-' && c != '+' && c != '.') {
      numeric = false;
      break;
    }
  }
  if (numeric && raw.find('-', 1) == std::string::npos) {
    char* end = nullptr;
    double v = std::strtod(raw.c_str(), &end);
    if (!end || *end != '\0') return false;
    // Heuristic: values past year 33658 in seconds are actually milliseconds.
    if (std::fabs(v) >= 1e12) v /= 1000.0;
    out = static_cast<std::int64_t>(v);
    return true;
  }
  return parse_iso8601(raw, out);
}

struct ColumnMap {
  int ts = -1, open = -1, high = -1, low = -1, close = -1, volume = -1;
};

bool map_header(const std::vector<std::string>& cells, ColumnMap& map) {
  bool any = false;
  for (std::size_t i = 0; i < cells.size(); ++i) {
    std::string name = lower(cells[i]);
    int idx = static_cast<int>(i);
    if (name == "timestamp" || name == "date" || name == "time" || name == "datetime" ||
        name == "ts") {
      map.ts = idx; any = true;
    } else if (name == "open" || name == "o") {
      map.open = idx; any = true;
    } else if (name == "high" || name == "h") {
      map.high = idx; any = true;
    } else if (name == "low" || name == "l") {
      map.low = idx; any = true;
    } else if (name == "close" || name == "c" || name == "adj close" || name == "price") {
      map.close = idx; any = true;
    } else if (name == "volume" || name == "vol" || name == "v") {
      map.volume = idx; any = true;
    }
  }
  return any && map.open >= 0 && map.high >= 0 && map.low >= 0 && map.close >= 0;
}

} // namespace

Series load_csv(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open CSV file: " + path);
  }

  Series series;
  std::string line;
  std::size_t lineNo = 0;
  char delim = ',';
  bool delimKnown = false;
  ColumnMap map;
  bool headerDone = false;
  bool headerless = false;

  while (std::getline(in, line)) {
    ++lineNo;
    std::string t = trim(line);
    if (t.empty() || t[0] == '#') continue;

    if (!delimKnown) {
      delim = t.find(';') != std::string::npos && t.find(',') == std::string::npos ? ';' : ',';
      delimKnown = true;
    }
    std::vector<std::string> cells = split_row(t, delim);

    if (!headerDone) {
      headerDone = true;
      if (map_header(cells, map)) {
        continue; // consumed the header row
      }
      // No header: infer layout from the first data row.
      headerless = true;
      double tmp;
      bool firstNumeric = parse_double(cells.empty() ? "" : cells[0], tmp);
      std::int64_t tsTmp;
      bool firstIsTs = !firstNumeric && !cells.empty() && parse_timestamp(cells[0], tsTmp);
      if (cells.size() >= 6 || firstIsTs ||
          (cells.size() == 5 && firstNumeric && tmp > 1e8)) {
        // [ts], o, h, l, c, [v]
        map.ts = 0; map.open = 1; map.high = 2; map.low = 3; map.close = 4;
        if (cells.size() >= 6) map.volume = 5;
      } else if (cells.size() == 5) {
        map.open = 0; map.high = 1; map.low = 2; map.close = 3; map.volume = 4;
      } else if (cells.size() == 4) {
        map.open = 0; map.high = 1; map.low = 2; map.close = 3;
      } else {
        throw std::runtime_error(path + ":" + std::to_string(lineNo) +
                                 ": cannot infer CSV layout (need o,h,l,c columns)");
      }
    }

    int maxIdx = std::max({map.ts, map.open, map.high, map.low, map.close, map.volume});
    if (static_cast<int>(cells.size()) <= maxIdx) {
      throw std::runtime_error(path + ":" + std::to_string(lineNo) + ": too few columns");
    }

    auto cell_value = [&](int idx, const char* what) {
      double v;
      // Columns are stored as float32, so a finite double beyond its range
      // would otherwise become Inf only after this check.
      if (!parse_double(cells[static_cast<std::size_t>(idx)], v) || !std::isfinite(v) ||
          std::fabs(v) > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::runtime_error(path + ":" + std::to_string(lineNo) + ": bad " +
                                 what + " value '" + cells[static_cast<std::size_t>(idx)] + "'");
      }
      return v;
    };
    auto fail = [&](const std::string& why) {
      throw std::runtime_error(path + ":" + std::to_string(lineNo) + ": " + why);
    };

    double o = cell_value(map.open, "open");
    double h = cell_value(map.high, "high");
    double l = cell_value(map.low, "low");
    double c = cell_value(map.close, "close");
    if (h < l) {
      throw std::runtime_error(path + ":" + std::to_string(lineNo) + ": high < low");
    }
    // Bar geometry is a universal invariant; the sign of a price is not, since
    // spreads and some energy contracts legitimately trade below zero.
    if (o < l || o > h) fail("open outside [low, high]");
    if (c < l || c > h) fail("close outside [low, high]");
    double v = map.volume >= 0 ? cell_value(map.volume, "volume") : 0.0;
    if (v < 0.0) fail("negative volume");

    std::int64_t ts = 0;
    if (map.ts >= 0) {
      if (!parse_timestamp(cells[static_cast<std::size_t>(map.ts)], ts)) {
        throw std::runtime_error(path + ":" + std::to_string(lineNo) + ": bad timestamp '" +
                                 cells[static_cast<std::size_t>(map.ts)] + "'");
      }
      if (!series.ts.empty() && ts <= series.ts.back()) {
        fail("timestamp '" + cells[static_cast<std::size_t>(map.ts)] +
             "' is not after the previous bar (rows must be strictly increasing in time)");
      }
    }

    series.ts.push_back(ts);
    series.open.push_back(static_cast<float>(o));
    series.high.push_back(static_cast<float>(h));
    series.low.push_back(static_cast<float>(l));
    series.close.push_back(static_cast<float>(c));
    series.volume.push_back(static_cast<float>(v));
  }

  (void)headerless;
  if (series.size() == 0) {
    throw std::runtime_error(path + ": no data rows");
  }
  return series;
}

} // namespace topt
