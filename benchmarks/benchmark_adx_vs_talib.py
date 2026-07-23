#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark ADX: tacuda vs TA-Lib
Average Directional Movement Index - индикатор силы тренда
Требует: High, Low, Close
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
    tacuda_adx = tacuda.ct_adx
except AttributeError as exc:
    raise SystemExit("ct_adx symbol not found in tacuda library") from exc

# ct_adx(const float* high, const float* low, const float* close,
#        float* output, int size, int period, cudaStream_t stream)
tacuda_adx.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # high
    ctypes.POINTER(ctypes.c_float),  # low
    ctypes.POINTER(ctypes.c_float),  # close
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # size
    ctypes.c_int,                    # period
    ctypes.c_void_p                  # stream
]
tacuda_adx.restype = ctypes.c_int

def generate_ohlc_data(size, seed=42):
    """Генерация реалистичных OHLC данных"""
    np.random.seed(seed)
    
    # Базовая цена с трендом и циклами
    x = np.linspace(0, 100, size)
    base_price = 100 + 20 * np.sin(x * 0.5) + x * 0.05
    
    # Добавляем волатильность
    volatility = np.random.randn(size) * 2
    
    # Close
    close = base_price + volatility
    
    # High и Low с реалистичными значениями
    high = close + np.abs(np.random.randn(size) * 1.5)
    low = close - np.abs(np.random.randn(size) * 1.5)
    
    # Убедимся, что High >= Close >= Low
    close = np.maximum(close, low)
    close = np.minimum(close, high)
    
    return high.astype(np.float32), low.astype(np.float32), close.astype(np.float32)

def benchmark_talib(high, low, close, period, iterations=100):
    """Бенчмарк TA-Lib ADX"""
    # TA-Lib требует float64
    high_f64 = high.astype(np.float64)
    low_f64 = low.astype(np.float64)
    close_f64 = close.astype(np.float64)
    
    # Прогрев
    for _ in range(5):
        _ = talib.ADX(high_f64, low_f64, close_f64, timeperiod=period)
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = talib.ADX(high_f64, low_f64, close_f64, timeperiod=period)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return result.astype(np.float32), avg_time_ms

def benchmark_tacuda(high, low, close, period, iterations=100):
    """Бенчмарк tacuda ADX"""
    size = len(high)
    
    # Входные данные уже float32
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
        tacuda_adx(high_ptr, low_ptr, close_ptr, output_ptr, size, period, None)
    
    start = time.perf_counter()
    for _ in range(iterations):
        tacuda_adx(high_ptr, low_ptr, close_ptr, output_ptr, size, period, None)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return output_arr, avg_time_ms

def compare_results(ta_result, cuda_result, period, tolerance=1e-4):
    """Сравнение результатов
    
    ADX имеет lookback = 2*period - 2
    """
    lookback = 2 * period - 2
    
    # Убираем lookback период
    valid_start = lookback
    if valid_start >= len(ta_result):
        return False, 0.0, 0.0
    
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
    
    # ADX может иметь небольшие различия из-за накопления погрешностей
    match = np.allclose(ta_valid, cuda_valid, rtol=tolerance, atol=tolerance)
    return match, max_diff, mean_diff

def print_separator():
    print("=" * 110)

def main():
    print("=" * 110)
    print("ADX Benchmark: tacuda vs TA-Lib".center(110))
    print("Average Directional Movement Index - Trend Strength Indicator".center(110))
    print("=" * 110)
    print()
    
    data_size = 500000
    print(f"Data size: {data_size:,} elements (High, Low, Close arrays)")
    print()
    
    # Генерация тестовых OHLC данных
    print("Generating realistic OHLC data...")
    high, low, close = generate_ohlc_data(data_size)
    print(f"Price range: Low={low.min():.2f}, High={high.max():.2f}, Close avg={close.mean():.2f}")
    print()
    
    # Разные размеры периодов (ADX обычно используют 14)
    test_periods = [5, 7, 10, 14, 20, 25, 30, 50, 100]
    
    print_separator()
    print(f"{'Period':<10} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15} {'Mean Diff':<15}")
    print_separator()
    
    results = []
    
    for period in test_periods:
        if 2 * period - 2 >= data_size:
            print(f"{period:<10} SKIPPED (lookback too large)")
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
    if results:
        speedups = [r['speedup'] for r in results]
        avg_speedup = np.mean(speedups)
        max_speedup = np.max(speedups)
        min_speedup = np.min(speedups)
        
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
            # Размер данных: 3 inputs (H,L,C) + 1 output = 4 массива float32
            data_bytes = data_size * 4 * 4
            time_sec = r['cuda_time'] / 1000
            throughput_gbps = (data_bytes / time_sec) / (1024**3)
            print(f"  Period {r['period']:3d}: {throughput_gbps:6.2f} GB/s")
        print()
        
        # Дополнительная информация
        print("Notes:")
        print(f"  - ADX lookback = 2*period - 2")
        print(f"  - For period=14: lookback = 26 bars")
        print(f"  - ADX measures trend strength (0-100)")
        print(f"  - Values > 25 indicate strong trend")
        print()
        
        # Показываем несколько значений ADX для проверки
        if results:
            period = 14  # Стандартный период ADX
            if any(r['period'] == period for r in results):
                print(f"Sample ADX values (period={period}):")
                ta_result, _ = benchmark_talib(high, low, close, period, iterations=1)
                cuda_result, _ = benchmark_tacuda(high, low, close, period, iterations=1)
                
                lookback = 2 * period - 2
                sample_indices = [lookback, lookback + 100, lookback + 1000, lookback + 10000, -1]
                
                print(f"{'Index':<12} {'TA-Lib':<12} {'tacuda':<12} {'Diff':<12}")
                print("-" * 48)
                for idx in sample_indices:
                    if idx < len(ta_result) and idx >= lookback:
                        ta_val = ta_result[idx]
                        cu_val = cuda_result[idx]
                        diff = abs(ta_val - cu_val)
                        print(f"{idx:<12} {ta_val:<12.4f} {cu_val:<12.4f} {diff:<12.6e}")
                print()

if __name__ == '__main__':
    main()

