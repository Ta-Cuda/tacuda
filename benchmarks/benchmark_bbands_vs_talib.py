#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark BBANDS (Bollinger Bands): tacuda vs TA-Lib
Сравниваем производительность на разных размерах окон и множителях стандартного отклонения
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
    tacuda_bbands = tacuda.ct_bbands
except AttributeError as exc:
    raise SystemExit("ct_bbands symbol not found in tacuda library") from exc

# ct_bbands(const float *host_input, float *host_upper, float *host_middle, 
#           float *host_lower, int size, int period, float upperMul, 
#           float lowerMul, cudaStream_t stream)
tacuda_bbands.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # upper
    ctypes.POINTER(ctypes.c_float),  # middle
    ctypes.POINTER(ctypes.c_float),  # lower
    ctypes.c_int,                    # size
    ctypes.c_int,                    # period
    ctypes.c_float,                  # upperMul
    ctypes.c_float,                  # lowerMul
    ctypes.c_void_p                  # stream
]
tacuda_bbands.restype = ctypes.c_int

def benchmark_talib(data, period, nbdevup=2.0, nbdevdn=2.0, iterations=100):
    """Бенчмарк TA-Lib BBANDS"""
    input_arr = np.asarray(data, dtype=np.float64)
    # Прогрев
    for _ in range(5):
        _ = talib.BBANDS(input_arr, timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn)
    
    start = time.perf_counter()
    for _ in range(iterations):
        upper, middle, lower = talib.BBANDS(input_arr, timeperiod=period, 
                                            nbdevup=nbdevup, nbdevdn=nbdevdn)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return (upper.astype(np.float32), 
            middle.astype(np.float32), 
            lower.astype(np.float32)), avg_time_ms

def benchmark_tacuda(data, period, nbdevup=2.0, nbdevdn=2.0, iterations=100):
    """Бенчмарк tacuda BBANDS"""
    size = len(data)
    input_arr = np.asarray(data, dtype=np.float32)
    upper_arr = np.full(size, np.nan, dtype=np.float32)
    middle_arr = np.full(size, np.nan, dtype=np.float32)
    lower_arr = np.full(size, np.nan, dtype=np.float32)

    input_ptr = input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    upper_ptr = upper_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    middle_ptr = middle_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lower_ptr = lower_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Прогрев GPU
    for _ in range(5):
        tacuda_bbands(input_ptr, upper_ptr, middle_ptr, lower_ptr, 
                     size, period, float(nbdevup), float(nbdevdn), None)

    start = time.perf_counter()
    for _ in range(iterations):
        tacuda_bbands(input_ptr, upper_ptr, middle_ptr, lower_ptr, 
                     size, period, float(nbdevup), float(nbdevdn), None)
    end = time.perf_counter()

    avg_time_ms = (end - start) / iterations * 1000
    return (upper_arr, middle_arr, lower_arr), avg_time_ms

def compare_results(ta_result, cuda_result, period, tolerance=1e-5):
    """
    Сравнение результатов BBANDS (3 выхода: upper, middle, lower)
    """
    ta_upper, ta_middle, ta_lower = ta_result
    cuda_upper, cuda_middle, cuda_lower = cuda_result
    
    # Убираем NaN для сравнения (первые period-1 значений)
    valid_start = period - 1
    
    # Создаем маски для валидных данных
    mask_upper = ~(np.isnan(ta_upper[valid_start:]) | np.isnan(cuda_upper[valid_start:]))
    mask_middle = ~(np.isnan(ta_middle[valid_start:]) | np.isnan(cuda_middle[valid_start:]))
    mask_lower = ~(np.isnan(ta_lower[valid_start:]) | np.isnan(cuda_lower[valid_start:]))
    
    ta_upper_valid = ta_upper[valid_start:][mask_upper]
    cuda_upper_valid = cuda_upper[valid_start:][mask_upper]
    
    ta_middle_valid = ta_middle[valid_start:][mask_middle]
    cuda_middle_valid = cuda_middle[valid_start:][mask_middle]
    
    ta_lower_valid = ta_lower[valid_start:][mask_lower]
    cuda_lower_valid = cuda_lower[valid_start:][mask_lower]
    
    if len(ta_upper_valid) == 0 or len(ta_middle_valid) == 0 or len(ta_lower_valid) == 0:
        return False, 0.0, 0.0
    
    # Вычисляем различия для каждой полосы
    diff_upper = np.abs(ta_upper_valid - cuda_upper_valid)
    diff_middle = np.abs(ta_middle_valid - cuda_middle_valid)
    diff_lower = np.abs(ta_lower_valid - cuda_lower_valid)
    
    max_diff_upper = np.max(diff_upper)
    max_diff_middle = np.max(diff_middle)
    max_diff_lower = np.max(diff_lower)
    max_diff = max(max_diff_upper, max_diff_middle, max_diff_lower)
    
    mean_diff_upper = np.mean(diff_upper)
    mean_diff_middle = np.mean(diff_middle)
    mean_diff_lower = np.mean(diff_lower)
    mean_diff = (mean_diff_upper + mean_diff_middle + mean_diff_lower) / 3.0
    
    # Проверяем совпадение всех трёх полос
    match_upper = np.allclose(ta_upper_valid, cuda_upper_valid, rtol=tolerance, atol=tolerance)
    match_middle = np.allclose(ta_middle_valid, cuda_middle_valid, rtol=tolerance, atol=tolerance)
    match_lower = np.allclose(ta_lower_valid, cuda_lower_valid, rtol=tolerance, atol=tolerance)
    
    match = match_upper and match_middle and match_lower
    
    return match, max_diff, mean_diff

def print_separator():
    print("=" * 120)

def main():
    print("=" * 120)
    print("BBANDS (Bollinger Bands) Benchmark: tacuda vs TA-Lib".center(120))
    print("=" * 120)
    print()

    data_size = 500000
    print(f"Data size: {data_size:,} elements")
    print()

    # Генерация тестовых данных (синусоида + шум + тренд)
    np.random.seed(42)
    x = np.linspace(0, 100, data_size)
    data = 100 + 0.5 * x + 10 * np.sin(x) + np.random.randn(data_size) * 2.0

    # Разные размеры окон
    test_configs = [
        # (period, nbdevup, nbdevdn)
        (5, 2.0, 2.0),
        (10, 2.0, 2.0),
        (20, 2.0, 2.0),
        (30, 2.0, 2.0),
        (50, 2.0, 2.0),
        (100, 2.0, 2.0),
        (200, 2.0, 2.0),
        (20, 1.0, 1.0),  # Узкие полосы
        (20, 3.0, 3.0),  # Широкие полосы
        (50, 1.5, 2.5),  # Асимметричные полосы
    ]

    print_separator()
    print(f"{'Period':<10} {'NbDevUp':<10} {'NbDevDn':<10} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} "
          f"{'Speedup':<12} {'Match':<8} {'Max Diff':<15} {'Mean Diff':<15}")
    print_separator()

    results = []

    for period, nbdevup, nbdevdn in test_configs:
        if period >= data_size:
            continue

        # TA-Lib benchmark
        ta_result, ta_time = benchmark_talib(data.astype(np.float32), period, 
                                             nbdevup, nbdevdn, iterations=50)

        # tacuda benchmark
        cuda_result, cuda_time = benchmark_tacuda(data, period, 
                                                  nbdevup, nbdevdn, iterations=50)

        # Сравнение результатов
        match, max_diff, mean_diff = compare_results(ta_result, cuda_result, period)

        speedup = ta_time / cuda_time if cuda_time > 0 else 0
        match_str = "PASS" if match else "FAIL"

        print(f"{period:<10} {nbdevup:<10.1f} {nbdevdn:<10.1f} {ta_time:<15.3f} {cuda_time:<15.3f} "
              f"{speedup:<12.2f}x {match_str:<8} {max_diff:<15.6e} {mean_diff:<15.6e}")

        results.append({
            'period': period,
            'nbdevup': nbdevup,
            'nbdevdn': nbdevdn,
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

    # Throughput анализ (3 выходных массива)
    print("Throughput Analysis (tacuda):")
    for r in results:
        # Размер данных: input (1x) + 3 outputs (upper, middle, lower)
        data_bytes = data_size * 4 * 4  # 1 input + 3 outputs, float32
        time_sec = r['cuda_time'] / 1000
        throughput_gbps = (data_bytes / time_sec) / (1024**3)
        print(f"  Period {r['period']:3d} (dev {r['nbdevup']:.1f}/{r['nbdevdn']:.1f}): "
              f"{throughput_gbps:6.2f} GB/s")
    print()

    # Дополнительная информация о BBANDS
    print("BBANDS Information:")
    print("  - Upper Band = Middle + (nbdevup * StdDev)")
    print("  - Middle Band = SMA(period)")
    print("  - Lower Band = Middle - (nbdevdn * StdDev)")
    print("  - StdDev = Population Standard Deviation over period")
    print("  - Output: 3 arrays (upper, middle, lower bands)")
    print()

if __name__ == '__main__':
    main()

