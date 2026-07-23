#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark ATR (Average True Range): tacuda vs TA-Lib
Сравниваем производительность на разных размерах периодов
ATR требует OHLC данные (High, Low, Close)
"""

import time
import numpy as np
import talib
import ctypes
import os
import sys

# Загрузка tacuda
tacuda_lib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'Release', 'tacuda.dll')
if not os.path.exists(tacuda_lib_path):
    tacuda_lib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'libtacuda.so')

# Добавляем путь к CUDA DLLs
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
if os.path.exists(cuda_path):
    os.environ['PATH'] = cuda_path + os.pathsep + os.environ.get('PATH', '')
    try:
        os.add_dll_directory(cuda_path)
    except (AttributeError, FileNotFoundError):
        pass

# Добавляем каталог tacuda.dll для корректной загрузки зависимостей (Python 3.8+)
tacuda_dir = os.path.dirname(tacuda_lib_path)
if os.path.isdir(tacuda_dir):
    try:
        os.add_dll_directory(os.path.abspath(tacuda_dir))
    except (AttributeError, FileNotFoundError):
        pass

try:
    tacuda = ctypes.CDLL(tacuda_lib_path)
except Exception as e:
    print(f"Failed to load tacuda library: {e}")
    print(f"Library path: {tacuda_lib_path}")
    print(f"Make sure CUDA runtime DLLs are in PATH")
    sys.exit(1)

try:
    tacuda_atr = tacuda.ct_atr
except AttributeError as exc:
    raise SystemExit("ct_atr symbol not found in tacuda library") from exc

# ct_atr(const float* high, const float* low, const float* close,
#        float* output, int size, int period, float initial, cudaStream_t stream)
tacuda_atr.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # high
    ctypes.POINTER(ctypes.c_float),  # low
    ctypes.POINTER(ctypes.c_float),  # close
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # size
    ctypes.c_int,                    # period
    ctypes.c_float,                  # initial (default 0.0)
    ctypes.c_void_p                  # stream
]
tacuda_atr.restype = ctypes.c_int

def generate_ohlc_data(size, seed=42):
    """
    Генерация реалистичных OHLC данных
    """
    np.random.seed(seed)
    
    # Генерируем базовую цену с трендом и волатильностью
    t = np.linspace(0, 100, size)
    base_price = 100 + 0.5 * t + 10 * np.sin(t / 5)
    
    # Добавляем волатильность
    volatility = 2.0 + 0.5 * np.sin(t / 10)
    noise = np.random.randn(size) * volatility
    
    # Close price
    close = base_price + noise
    close = np.maximum(close, 1.0)  # Prevent negative prices
    
    # High: close + random positive movement
    high = close + np.abs(np.random.randn(size)) * volatility * 0.5
    
    # Low: close - random positive movement
    low = close - np.abs(np.random.randn(size)) * volatility * 0.5
    
    # Ensure High >= Low >= 0
    low = np.minimum(low, close)
    high = np.maximum(high, close)
    low = np.maximum(low, 0.1)
    
    return high.astype(np.float32), low.astype(np.float32), close.astype(np.float32)

def benchmark_talib(high, low, close, period, iterations=100):
    """Бенчмарк TA-Lib ATR"""
    high_arr = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)
    close_arr = np.asarray(close, dtype=np.float64)
    
    # Прогрев
    for _ in range(5):
        _ = talib.ATR(high_arr, low_arr, close_arr, timeperiod=period)
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = talib.ATR(high_arr, low_arr, close_arr, timeperiod=period)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return result.astype(np.float32), avg_time_ms

def benchmark_tacuda(high, low, close, period, iterations=100):
    """Бенчмарк tacuda ATR"""
    size = len(high)
    high_arr = np.asarray(high, dtype=np.float32)
    low_arr = np.asarray(low, dtype=np.float32)
    close_arr = np.asarray(close, dtype=np.float32)
    output_arr = np.full(size, np.nan, dtype=np.float32)

    high_ptr = high_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    low_ptr = low_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    close_ptr = close_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Прогрев GPU
    for _ in range(5):
        tacuda_atr(high_ptr, low_ptr, close_ptr, output_ptr, size, period, ctypes.c_float(0.0), None)

    start = time.perf_counter()
    for _ in range(iterations):
        tacuda_atr(high_ptr, low_ptr, close_ptr, output_ptr, size, period, ctypes.c_float(0.0), None)
    end = time.perf_counter()

    avg_time_ms = (end - start) / iterations * 1000
    return output_arr, avg_time_ms

def compare_results(ta_result, cuda_result, period, tolerance=1e-5):
    """Сравнение результатов"""
    # ATR начинается с индекса period-1 (TA-Lib convention)
    valid_start = period - 1
    ta_valid = ta_result[valid_start:]
    cuda_valid = cuda_result[valid_start:]

    # Убираем оставшиеся NaN
    mask = ~(np.isnan(ta_valid) | np.isnan(cuda_valid))
    ta_valid = ta_valid[mask]
    cuda_valid = cuda_valid[mask]

    if len(ta_valid) == 0:
        return False, 0.0, 0.0

    diff = np.abs(ta_valid - cuda_valid)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    match = np.allclose(ta_valid, cuda_valid, rtol=tolerance, atol=tolerance)
    return match, max_diff, mean_diff

def print_separator():
    print("=" * 110)

def main():
    print_separator()
    print("ATR (Average True Range) Benchmark: tacuda vs TA-Lib".center(110))
    print_separator()
    print()

    data_size = 500000
    print(f"Data size: {data_size:,} bars (OHLC)")
    print()

    # Генерация OHLC данных
    print("Generating OHLC data...")
    high, low, close = generate_ohlc_data(data_size)
    print(f"  High range:  [{high.min():.2f}, {high.max():.2f}]")
    print(f"  Low range:   [{low.min():.2f}, {low.max():.2f}]")
    print(f"  Close range: [{close.min():.2f}, {close.max():.2f}]")
    print()

    # Разные размеры периодов для ATR
    test_periods = [5, 7, 10, 14, 20, 30, 50, 100, 200]

    print_separator()
    print(f"{'Period':<10} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15} {'Mean Diff':<15}")
    print_separator()

    results = []

    for period in test_periods:
        if period >= data_size:
            continue

        # TA-Lib benchmark
        ta_result, ta_time = benchmark_talib(high, low, close, period, iterations=50)

        # tacuda benchmark
        cuda_result, cuda_time = benchmark_tacuda(high, low, close, period, iterations=50)

        # Сравнение результатов
        match, max_diff, mean_diff = compare_results(ta_result, cuda_result, period)

        speedup = ta_time / cuda_time if cuda_time > 0 else 0
        match_str = "PASS" if match else "FAIL"

        print(f"{period:<10} {ta_time:<15.3f} {cuda_time:<15.3f} {speedup:<12.2f}x {match_str:<8} {max_diff:<15.6e} {mean_diff:<15.6e}")

        results.append({
            'period': period,
            'ta_time': ta_time,
            'cuda_time': cuda_time,
            'speedup': speedup,
            'match': match,
            'max_diff': max_diff,
            'mean_diff': mean_diff
        })

    print_separator()
    print()

    # Статистика
    speedups = [r['speedup'] for r in results]
    avg_speedup = np.mean(speedups) if speedups else 0.0
    max_speedup = np.max(speedups) if speedups else 0.0
    min_speedup = np.min(speedups) if speedups else 0.0

    all_match = all(r['match'] for r in results)

    print("Summary:")
    print(f"  Average speedup: {avg_speedup:.2f}x")
    print(f"  Max speedup:     {max_speedup:.2f}x")
    print(f"  Min speedup:     {min_speedup:.2f}x")
    print(f"  All tests match: {'YES' if all_match else 'NO'}")
    print()

    # Throughput анализ
    print("Throughput Analysis (tacuda):")
    for r in results:
        # Размер данных: 3 inputs (HLC) + 1 output = 4x для float32
        data_bytes = data_size * 4 * 4
        time_sec = r['cuda_time'] / 1000
        throughput_gbps = (data_bytes / time_sec) / (1024**3)
        print(f"  Period {r['period']:3d}: {throughput_gbps:6.2f} GB/s")
    print()

    # Детальный анализ для популярного периода 14
    if 14 in test_periods:
        print("Detailed Analysis for ATR(14) - Most Common Period:")
        idx = test_periods.index(14)
        r = results[idx]
        
        # Пересчитываем для детального анализа
        ta_result, _ = benchmark_talib(high, low, close, 14, iterations=1)
        cuda_result, _ = benchmark_tacuda(high, low, close, 14, iterations=1)
        
        valid_start = 14 - 1
        ta_slice = ta_result[valid_start:valid_start+10]
        cuda_slice = cuda_result[valid_start:valid_start+10]
        
        print(f"  First 10 valid values comparison:")
        print(f"  {'Index':<8} {'TA-Lib':<15} {'tacuda':<15} {'Diff':<15}")
        print("  " + "-" * 53)
        for i in range(min(10, len(ta_slice))):
            idx = valid_start + i
            diff = abs(ta_slice[i] - cuda_slice[i])
            print(f"  {idx:<8} {ta_slice[i]:<15.6f} {cuda_slice[i]:<15.6f} {diff:<15.6e}")
        print()

    print_separator()

if __name__ == '__main__':
    main()

