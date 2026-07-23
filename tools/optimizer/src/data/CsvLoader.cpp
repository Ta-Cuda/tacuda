/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#include "data/CsvLoader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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
  int y = 0, mo = 0, d = 0, h = 0, mi = 0, s = 0;
  int matched = std::sscanf(raw.c_str(), "%d-%d-%d%*1[ T]%d:%d:%d", &y, &mo, &d, &h, &mi, &s);
  if (matched < 3) return false;
  if (mo < 1 || mo > 12 || d < 1 || d > 31) return false;
  out = days_from_civil(y, mo, d) * 86400ll + h * 3600ll + mi * 60ll + s;
  return true;
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
      if (!parse_double(cells[static_cast<std::size_t>(idx)], v) || !std::isfinite(v)) {
        throw std::runtime_error(path + ":" + std::to_string(lineNo) + ": bad " +
                                 what + " value '" + cells[static_cast<std::size_t>(idx)] + "'");
      }
      return v;
    };

    double o = cell_value(map.open, "open");
    double h = cell_value(map.high, "high");
    double l = cell_value(map.low, "low");
    double c = cell_value(map.close, "close");
    if (h < l) {
      throw std::runtime_error(path + ":" + std::to_string(lineNo) + ": high < low");
    }
    double v = map.volume >= 0 ? cell_value(map.volume, "volume") : 0.0;

    std::int64_t ts = 0;
    if (map.ts >= 0) {
      if (!parse_timestamp(cells[static_cast<std::size_t>(map.ts)], ts)) {
        throw std::runtime_error(path + ":" + std::to_string(lineNo) + ": bad timestamp '" +
                                 cells[static_cast<std::size_t>(map.ts)] + "'");
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
