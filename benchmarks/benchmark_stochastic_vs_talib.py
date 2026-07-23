#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark Stochastic Oscillator: tacuda vs TA-Lib
Сравниваем производительность на разных периодах %K и %D
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
    tacuda_stoch = tacuda.ct_stochastic
except AttributeError as exc:
    raise SystemExit("ct_stochastic symbol not found in tacuda library") from exc

# ct_stochastic(const float* high, const float* low, const float* close,
#               float* k_out, float* d_out, int size, int k_period, int d_period, cudaStream_t stream)
tacuda_stoch.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # high
    ctypes.POINTER(ctypes.c_float),  # low
    ctypes.POINTER(ctypes.c_float),  # close
    ctypes.POINTER(ctypes.c_float),  # k_out
    ctypes.POINTER(ctypes.c_float),  # d_out
    ctypes.c_int,                    # size
    ctypes.c_int,                    # k_period
    ctypes.c_int,                    # d_period
    ctypes.c_void_p                  # stream
]
tacuda_stoch.restype = ctypes.c_int

def benchmark_talib(high, low, close, k_period, d_period, iterations=50):
    """Бенчмарк TA-Lib Stochastic"""
    high_arr = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)
    close_arr = np.asarray(close, dtype=np.float64)
    
    # Прогрев
    for _ in range(5):
        _, _ = talib.STOCH(high_arr, low_arr, close_arr, 
                          fastk_period=k_period, slowk_period=d_period, 
                          slowk_matype=0, slowd_period=d_period, slowd_matype=0)
    
    start = time.perf_counter()
    for _ in range(iterations):
        k_result, d_result = talib.STOCH(high_arr, low_arr, close_arr,
                                         fastk_period=k_period, slowk_period=d_period,
                                         slowk_matype=0, slowd_period=d_period, slowd_matype=0)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return k_result.astype(np.float32), d_result.astype(np.float32), avg_time_ms

def benchmark_tacuda(high, low, close, k_period, d_period, iterations=50):
    """Бенчмарк tacuda Stochastic"""
    size = len(high)
    high_arr = np.asarray(high, dtype=np.float32)
    low_arr = np.asarray(low, dtype=np.float32)
    close_arr = np.asarray(close, dtype=np.float32)
    
    k_out = np.full(size, np.nan, dtype=np.float32)
    d_out = np.full(size, np.nan, dtype=np.float32)

    high_ptr = high_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    low_ptr = low_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    close_ptr = close_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    k_ptr = k_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    d_ptr = d_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Прогрев GPU
    for _ in range(5):
        tacuda_stoch(high_ptr, low_ptr, close_ptr, k_ptr, d_ptr, 
                     size, k_period, d_period, None)

    start = time.perf_counter()
    for _ in range(iterations):
        tacuda_stoch(high_ptr, low_ptr, close_ptr, k_ptr, d_ptr,
                     size, k_period, d_period, None)
    end = time.perf_counter()

    avg_time_ms = (end - start) / iterations * 1000
    return k_out, d_out, avg_time_ms

def compare_results(ta_k, ta_d, cuda_k, cuda_d, k_period, d_period, tolerance=1e-5):
    """Сравнение результатов Stochastic"""
    # Stochastic lookback = k_period + d_period - 2
    lookback = k_period + d_period - 2
    
    # Проверяем %K
    ta_k_valid = ta_k[lookback:]
    cuda_k_valid = cuda_k[lookback:]
    
    mask_k = ~(np.isnan(ta_k_valid) | np.isnan(cuda_k_valid))
    ta_k_valid = ta_k_valid[mask_k]
    cuda_k_valid = cuda_k_valid[mask_k]
    
    # Проверяем %D
    ta_d_valid = ta_d[lookback:]
    cuda_d_valid = cuda_d[lookback:]
    
    mask_d = ~(np.isnan(ta_d_valid) | np.isnan(cuda_d_valid))
    ta_d_valid = ta_d_valid[mask_d]
    cuda_d_valid = cuda_d_valid[mask_d]
    
    if len(ta_k_valid) == 0 or len(ta_d_valid) == 0:
        return False, 0.0, 0.0, 0.0, 0.0
    
    # Различия для %K
    diff_k = np.abs(ta_k_valid - cuda_k_valid)
    max_diff_k = np.max(diff_k)
    mean_diff_k = np.mean(diff_k)
    
    # Различия для %D
    diff_d = np.abs(ta_d_valid - cuda_d_valid)
    max_diff_d = np.max(diff_d)
    mean_diff_d = np.mean(diff_d)
    
    match_k = np.allclose(ta_k_valid, cuda_k_valid, rtol=tolerance, atol=tolerance)
    match_d = np.allclose(ta_d_valid, cuda_d_valid, rtol=tolerance, atol=tolerance)
    
    match = match_k and match_d
    max_diff = max(max_diff_k, max_diff_d)
    mean_diff = (mean_diff_k + mean_diff_d) / 2
    
    return match, max_diff, mean_diff, max_diff_k, max_diff_d

def print_separator():
    print("=" * 120)

def main():
    print_separator()
    print("Stochastic Oscillator Benchmark: tacuda vs TA-Lib".center(120))
    print_separator()
    print()

    data_size = 500000
    print(f"Data size: {data_size:,} elements")
    print()

    # Генерация реалистичных OHLC данных
    np.random.seed(42)
    
    # Симулируем цены с трендом и волатильностью
    x = np.linspace(0, 100, data_size)
    base_price = 100 + 0.1 * x + 10 * np.sin(2 * np.pi * x / 50)
    volatility = np.random.randn(data_size) * 2
    
    # Генерируем OHLC
    close = base_price + volatility
    high = close + np.abs(np.random.randn(data_size)) * 1.5
    low = close - np.abs(np.random.randn(data_size)) * 1.5
    
    # Убедимся что high >= close >= low
    close = np.clip(close, low + 0.01, high - 0.01)

    # Разные конфигурации периодов (fastk_period, slowk/slowd_period)
    test_configs = [
        (5, 3, "Fast Stochastic (5,3)"),
        (14, 3, "Standard Stochastic (14,3)"),
        (21, 5, "Medium Stochastic (21,5)"),
        (14, 5, "Standard-Medium (14,5)"),
        (9, 3, "Short-term (9,3)"),
        (21, 7, "Long-term (21,7)"),
    ]

    print_separator()
    print(f"{'Config':<30} {'%K Period':<12} {'%D Period':<12} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} "
          f"{'Speedup':<12} {'Match':<8}")
    print_separator()

    results = []

    for k_period, d_period, description in test_configs:
        lookback = k_period + d_period - 2
        if lookback >= data_size:
            continue

        try:
            # TA-Lib benchmark
            ta_k, ta_d, ta_time = benchmark_talib(high, low, close, k_period, d_period, iterations=50)

            # tacuda benchmark
            cuda_k, cuda_d, cuda_time = benchmark_tacuda(high, low, close, k_period, d_period, iterations=50)

            # Сравнение результатов
            match, max_diff, mean_diff, max_diff_k, max_diff_d = compare_results(
                ta_k, ta_d, cuda_k, cuda_d, k_period, d_period)

            speedup = ta_time / cuda_time if cuda_time > 0 else 0
            match_str = "PASS" if match else "FAIL"

            print(f"{description:<30} {k_period:<12} {d_period:<12} {ta_time:<15.3f} {cuda_time:<15.3f} "
                  f"{speedup:<12.2f}x {match_str:<8}")

            results.append({
                'description': description,
                'k_period': k_period,
                'd_period': d_period,
                'ta_time': ta_time,
                'cuda_time': cuda_time,
                'speedup': speedup,
                'match': match,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'max_diff_k': max_diff_k,
                'max_diff_d': max_diff_d
            })
        
        except Exception as e:
            print(f"{description:<30} ERROR: {e}")

    print_separator()
    print()

    if results:
        # Статистика
        speedups = [r['speedup'] for r in results]
        avg_speedup = np.mean(speedups)
        max_speedup = np.max(speedups)
        min_speedup = np.min(speedups)

        all_match = all(r['match'] for r in results)

        print("Summary:")
        print(f"  Average speedup: {avg_speedup:6.2f}x")
        print(f"  Max speedup:     {max_speedup:6.2f}x")
        print(f"  Min speedup:     {min_speedup:6.2f}x")
        print(f"  All tests match: {'YES' if all_match else 'NO'}")
        print()

        # Анализ погрешностей
        print("Accuracy Analysis:")
        for r in results:
            print(f"  {r['description']:<30}: Max diff %K: {r['max_diff_k']:.6e}, Max diff %D: {r['max_diff_d']:.6e}")
        print()

        # Throughput анализ
        print("Throughput Analysis (tacuda):")
        for r in results:
            # Размер данных: 3 inputs (H,L,C) + 2 outputs (%K,%D) = 5x для float32
            data_bytes = data_size * 4 * 5
            time_sec = r['cuda_time'] / 1000
            throughput_gbps = (data_bytes / time_sec) / (1024**3)
            print(f"  {r['description']:<30}: {throughput_gbps:6.2f} GB/s")
        print()

    print_separator()
    print("Stochastic Oscillator Notes:".center(120))
    print_separator()
    print()
    print("  * Stochastic Oscillator - популярный momentum индикатор")
    print("  * %K (Fast): показывает текущее положение цены относительно диапазона")
    print("  * %D (Slow): сглаженная версия %K (сигнальная линия)")
    print("  * Значения: 0-100. >80 = overbought, <20 = oversold")
    print("  * Lookback period = k_period + d_period - 2")
    print()

if __name__ == '__main__':
    main()

