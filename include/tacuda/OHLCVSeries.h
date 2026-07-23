/* ======================================================================== */
/*  TaCuda | GPU-accelerated technical analysis | https://tacuda.com        */
/*  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>               */
/*  SPDX-License-Identifier: Apache-2.0                                     */
/* ======================================================================== */
#ifndef TACUDA_OHLCVSERIES_H
#define TACUDA_OHLCVSERIES_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tacuda {

class OHLCVSeries {
public:
  enum class Column : std::size_t { Open = 0, High = 1, Low = 2, Close = 3, Volume = 4 };
  static constexpr std::size_t ColumnCount = 5;

  OHLCVSeries() = default;

  explicit OHLCVSeries(std::size_t size) { resize(size); }

  OHLCVSeries(const float *open, const float *high, const float *low,
              const float *close, const float *volume, std::size_t size) {
    assign_from_raw(open, high, low, close, volume, size);
  }

  OHLCVSeries(std::vector<float> open, std::vector<float> high,
              std::vector<float> low, std::vector<float> close,
              std::vector<float> volume = {}) {
    initialise_from_vectors(std::move(open), std::move(high), std::move(low),
                            std::move(close), std::move(volume));
  }

  OHLCVSeries(const float *column_major, std::size_t size,
              bool include_volume) {
    if (!column_major && size != 0) {
      throw std::invalid_argument(
          "tacuda::OHLCVSeries: column-major pointer must not be null");
    }
    resize(size);
    if (size == 0) {
      return;
    }
    std::size_t columns = include_volume ? ColumnCount : ColumnCount - 1;
    for (std::size_t c = 0; c < columns; ++c) {
      std::copy_n(column_major + c * size_, size_, columns_[c].begin());
    }
    if (!include_volume) {
      std::fill(columns_[index(Column::Volume)].begin(),
                columns_[index(Column::Volume)].end(), 0.0f);
    }
  }

  static OHLCVSeries from_rows(const float *rows, std::size_t count,
                               std::size_t stride = ColumnCount) {
    if (!rows && count != 0) {
      throw std::invalid_argument(
          "tacuda::OHLCVSeries: row pointer must not be null when count > 0");
    }
    if (stride != 4 && stride != ColumnCount) {
      throw std::invalid_argument(
          "tacuda::OHLCVSeries: stride must be 4 (OHLC) or 5 (OHLCV)");
    }
    OHLCVSeries series(count);
    for (std::size_t i = 0; i < count; ++i) {
      const float *row = rows + i * stride;
      float volume = stride == ColumnCount ? row[4] : 0.0f;
      series.set_row(i, row[0], row[1], row[2], row[3], volume);
    }
    return series;
  }

  void resize(std::size_t size) {
    size_ = size;
    for (auto &column : columns_) {
      column.assign(size_, 0.0f);
    }
  }

  std::size_t size() const noexcept { return size_; }

  bool empty() const noexcept { return size_ == 0; }

  std::vector<float> &open() noexcept { return columns_[index(Column::Open)]; }
  const std::vector<float> &open() const noexcept {
    return columns_[index(Column::Open)];
  }
  std::vector<float> &high() noexcept { return columns_[index(Column::High)]; }
  const std::vector<float> &high() const noexcept {
    return columns_[index(Column::High)];
  }
  std::vector<float> &low() noexcept { return columns_[index(Column::Low)]; }
  const std::vector<float> &low() const noexcept {
    return columns_[index(Column::Low)];
  }
  std::vector<float> &close() noexcept { return columns_[index(Column::Close)]; }
  const std::vector<float> &close() const noexcept {
    return columns_[index(Column::Close)];
  }
  std::vector<float> &volume() noexcept {
    return columns_[index(Column::Volume)];
  }
  const std::vector<float> &volume() const noexcept {
    return columns_[index(Column::Volume)];
  }

  float *open_data() noexcept { return data(Column::Open); }
  const float *open_data() const noexcept { return data(Column::Open); }
  float *high_data() noexcept { return data(Column::High); }
  const float *high_data() const noexcept { return data(Column::High); }
  float *low_data() noexcept { return data(Column::Low); }
  const float *low_data() const noexcept { return data(Column::Low); }
  float *close_data() noexcept { return data(Column::Close); }
  const float *close_data() const noexcept { return data(Column::Close); }
  float *volume_data() noexcept { return data(Column::Volume); }
  const float *volume_data() const noexcept { return data(Column::Volume); }

  void assign_column(Column column, const float *values, std::size_t count) {
    if (!values && count != 0) {
      throw std::invalid_argument(
          "tacuda::OHLCVSeries: input pointer must not be null when count > 0");
    }
    if (count != size_) {
      throw std::invalid_argument(
          "tacuda::OHLCVSeries: column length mismatch");
    }
    std::copy_n(values, size_, columns_[index(column)].begin());
  }

  void assign_column(Column column, std::vector<float> values) {
    if (values.size() != size_) {
      throw std::invalid_argument(
          "tacuda::OHLCVSeries: column length mismatch");
    }
    columns_[index(column)] = std::move(values);
  }

  void set_row(std::size_t idx, float open_value, float high_value,
               float low_value, float close_value, float volume_value = 0.0f) {
    if (idx >= size_) {
      throw std::out_of_range("tacuda::OHLCVSeries: row index out of range");
    }
    open()[idx] = open_value;
    high()[idx] = high_value;
    low()[idx] = low_value;
    close()[idx] = close_value;
    volume()[idx] = volume_value;
  }

  std::vector<float> column_major(bool include_volume = true) const {
    std::size_t columns = include_volume ? ColumnCount : ColumnCount - 1;
    std::vector<float> packed(columns * size_);
    copy_column_major(packed.data(), include_volume);
    return packed;
  }

  void copy_column_major(float *dest, bool include_volume = true) const {
    if (!dest && size_ != 0) {
      throw std::invalid_argument(
          "tacuda::OHLCVSeries: destination pointer must not be null");
    }
    std::size_t columns = include_volume ? ColumnCount : ColumnCount - 1;
    for (std::size_t c = 0; c < columns; ++c) {
      std::copy_n(columns_[c].begin(), size_, dest + c * size_);
    }
  }

  const std::vector<float> &column(Column column) const {
    return columns_[index(column)];
  }

  std::vector<float> &column(Column column) { return columns_[index(column)]; }

private:
  std::size_t index(Column column) const noexcept {
    return static_cast<std::size_t>(column);
  }

  float *data(Column column) noexcept {
    auto &vec = columns_[index(column)];
    return vec.empty() ? nullptr : vec.data();
  }

  const float *data(Column column) const noexcept {
    const auto &vec = columns_[index(column)];
    return vec.empty() ? nullptr : vec.data();
  }

  void initialise_from_vectors(std::vector<float> open, std::vector<float> high,
                               std::vector<float> low,
                               std::vector<float> close,
                               std::vector<float> volume) {
    std::size_t size = open.size();
    ensure_same_size(size, high.size(), "high");
    ensure_same_size(size, low.size(), "low");
    ensure_same_size(size, close.size(), "close");
    if (!volume.empty() && volume.size() != size) {
      throw std::invalid_argument(
          "tacuda::OHLCVSeries: volume column length mismatch");
    }
    size_ = size;
    columns_[index(Column::Open)] = std::move(open);
    columns_[index(Column::High)] = std::move(high);
    columns_[index(Column::Low)] = std::move(low);
    columns_[index(Column::Close)] = std::move(close);
    if (volume.empty()) {
      columns_[index(Column::Volume)].assign(size_, 0.0f);
    } else {
      columns_[index(Column::Volume)] = std::move(volume);
    }
  }

  void assign_from_raw(const float *open, const float *high, const float *low,
                       const float *close, const float *volume,
                       std::size_t size) {
    if ((!open || !high || !low || !close) && size != 0) {
      throw std::invalid_argument(
          "tacuda::OHLCVSeries: OHLC pointers must not be null");
    }
    resize(size);
    if (size == 0) {
      return;
    }
    std::copy_n(open, size_, columns_[index(Column::Open)].begin());
    std::copy_n(high, size_, columns_[index(Column::High)].begin());
    std::copy_n(low, size_, columns_[index(Column::Low)].begin());
    std::copy_n(close, size_, columns_[index(Column::Close)].begin());
    if (volume) {
      std::copy_n(volume, size_, columns_[index(Column::Volume)].begin());
    }
  }

  static void ensure_same_size(std::size_t expected, std::size_t actual,
                               const char *name) {
    if (actual != expected) {
      throw std::invalid_argument(
          std::string("tacuda::OHLCVSeries: column '") + name +
          "' length mismatch");
    }
  }

  std::size_t size_{0};
  std::array<std::vector<float>, ColumnCount> columns_{};
};

} // namespace tacuda

#endif // TACUDA_OHLCVSERIES_H
