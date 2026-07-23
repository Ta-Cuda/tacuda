#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark Momentum: tacuda vs TA-Lib
Rate of Change (Momentum) - измеряет изменение цены за период
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
    tacuda_momentum = tacuda.ct_momentum
except AttributeError as exc:
    raise SystemExit("ct_momentum symbol not found in tacuda library") from exc

# ct_momentum(const float* input, float* output, int size, int period, cudaStream_t stream)
tacuda_momentum.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # size
    ctypes.c_int,                    # period
    ctypes.c_void_p                  # stream
]
tacuda_momentum.restype = ctypes.c_int

def benchmark_talib(data, period, iterations=100):
    """Бенчмарк TA-Lib Momentum"""
    input_arr = np.asarray(data, dtype=np.float64)
    # Прогрев
    for _ in range(5):
        _ = talib.MOM(input_arr, timeperiod=period)
    start = time.perf_counter()
    for _ in range(iterations):
        result = talib.MOM(input_arr, timeperiod=period)
    end = time.perf_counter()
    avg_time_ms = (end - start) / iterations * 1000
    return result.astype(np.float32), avg_time_ms

def benchmark_tacuda(data, period, iterations=100):
    """Бенчмарк tacuda Momentum"""
    size = len(data)
    input_arr = np.asarray(data, dtype=np.float32)
    output_arr = np.full(size, np.nan, dtype=np.float32)

    input_ptr = input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Прогрев GPU
    for _ in range(5):
        tacuda_momentum(input_ptr, output_ptr, size, period, None)

    start = time.perf_counter()
    for _ in range(iterations):
        tacuda_momentum(input_ptr, output_ptr, size, period, None)
    end = time.perf_counter()

    avg_time_ms = (end - start) / iterations * 1000
    return output_arr, avg_time_ms

def compare_results(ta_result, cuda_result, period, tolerance=1e-5):
    """Сравнение результатов"""
    # Убираем NaN для сравнения (первые period значений - NaN)
    valid_start = period
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
    print("Momentum Benchmark: tacuda vs TA-Lib".center(110))
    print("Rate of Change - измеряет изменение цены за период".center(110))
    print_separator()
    print()

    data_size = 500000
    print(f"Data size: {data_size:,} elements")
    print()

    # Генерация реалистичных ценовых данных (тренд + циклы + шум)
    np.random.seed(42)
    x = np.linspace(0, 100, data_size)
    # Тренд + синусоида + шум
    trend = x * 0.5
    cycles = np.sin(x * 0.5) * 20 + np.sin(x * 0.1) * 50
    noise = np.random.randn(data_size) * 5
    data = 100 + trend + cycles + noise
    
    print(f"Price range: Min={data.min():.2f}, Max={data.max():.2f}, Mean={data.mean():.2f}")
    print()

    # Разные размеры периодов
    test_periods = [1, 5, 10, 14, 20, 30, 50, 100, 200]

    print_separator()
    print(f"{'Period':<10} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15} {'Mean Diff':<15}")
    print_separator()

    results = []

    for period in test_periods:
        if period >= data_size:
            continue

        # TA-Lib benchmark
        ta_result, ta_time = benchmark_talib(data.astype(np.float32), period, iterations=50)

        # tacuda benchmark
        cuda_result, cuda_time = benchmark_tacuda(data, period, iterations=50)

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
        # Размер данных: input + output (2x для float32)
        data_bytes = data_size * 4 * 2
        time_sec = r['cuda_time'] / 1000
        throughput_gbps = (data_bytes / time_sec) / (1024**3)
        print(f"  Period {r['period']:3d}: {throughput_gbps:6.2f} GB/s")
    print()

    # Пример значений для проверки
    print("Sample Momentum values (period=10):")
    print(f"{'Index':<12} {'TA-Lib':<12} {'tacuda':<12} {'Diff':<12}")
    print("-" * 48)
    
    period = 10
    ta_sample, _ = benchmark_talib(data.astype(np.float32), period, iterations=1)
    cuda_sample, _ = benchmark_tacuda(data, period, iterations=1)
    
    sample_indices = [period, 100, 1000, 10000, 100000]
    for idx in sample_indices:
        if idx < len(data):
            ta_val = ta_sample[idx]
            cuda_val = cuda_sample[idx]
            diff = abs(ta_val - cuda_val) if not (np.isnan(ta_val) or np.isnan(cuda_val)) else 0.0
            print(f"{idx:<12} {ta_val:<12.4f} {cuda_val:<12.4f} {diff:<12.6e}")
    print()

    # Информация об индикаторе
    print("Notes:")
    print("  - Momentum = Price[t] - Price[t-period]")
    print("  - Показывает абсолютное изменение цены")
    print("  - Положительные значения = восходящий тренд")
    print("  - Отрицательные значения = нисходящий тренд")
    print("  - Lookback = period баров")
    print_separator()

if __name__ == '__main__':
    main()

