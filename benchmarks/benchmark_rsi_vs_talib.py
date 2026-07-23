#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark RSI: tacuda vs TA-Lib
Сравниваем производительность Relative Strength Index на разных периодах
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
    tacuda_rsi = tacuda.ct_rsi
except AttributeError as exc:
    raise SystemExit("ct_rsi symbol not found in tacuda library") from exc

# ct_rsi(const float* host_input, float* host_output, int size, int period, cudaStream_t stream)
tacuda_rsi.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # size
    ctypes.c_int,                    # period
    ctypes.c_void_p                  # stream
]
tacuda_rsi.restype = ctypes.c_int

def benchmark_talib(data, period, iterations=100):
    """Бенчмарк TA-Lib RSI"""
    input_arr = np.asarray(data, dtype=np.float64)
    # Прогрев
    for _ in range(5):
        _ = talib.RSI(input_arr, timeperiod=period)
    start = time.perf_counter()
    for _ in range(iterations):
        result = talib.RSI(input_arr, timeperiod=period)
    end = time.perf_counter()
    avg_time_ms = (end - start) / iterations * 1000
    return result.astype(np.float32), avg_time_ms

def benchmark_tacuda(data, period, iterations=100):
    """Бенчмарк tacuda RSI"""
    size = len(data)
    input_arr = np.asarray(data, dtype=np.float32)
    output_arr = np.full(size, np.nan, dtype=np.float32)

    input_ptr = input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Прогрев GPU
    for _ in range(5):
        tacuda_rsi(input_ptr, output_ptr, size, period, None)

    start = time.perf_counter()
    for _ in range(iterations):
        tacuda_rsi(input_ptr, output_ptr, size, period, None)
    end = time.perf_counter()

    avg_time_ms = (end - start) / iterations * 1000
    return output_arr, avg_time_ms

def compare_results(ta_result, cuda_result, period, tolerance=1e-4):
    """Сравнение результатов RSI
    
    Примечание: RSI может иметь небольшие расхождения из-за накопления погрешностей
    в вычислениях с плавающей точкой, поэтому используем более мягкий tolerance
    """
    # Убираем NaN для сравнения (первые period значений)
    valid_start = period
    ta_valid = ta_result[valid_start:]
    cuda_valid = cuda_result[valid_start:]

    # Убираем оставшиеся NaN
    mask = ~(np.isnan(ta_valid) | np.isnan(cuda_valid))
    ta_valid = ta_valid[mask]
    cuda_valid = cuda_valid[mask]

    if len(ta_valid) == 0:
        return False, 0.0, 0.0, 0.0

    diff = np.abs(ta_valid - cuda_valid)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Относительная погрешность для RSI (значения от 0 до 100)
    rel_diff = diff / (np.abs(ta_valid) + 1e-8)  # избегаем деления на 0
    max_rel_diff = np.max(rel_diff)

    # Проверка с учетом как абсолютной, так и относительной погрешности
    match = np.allclose(ta_valid, cuda_valid, rtol=tolerance, atol=tolerance)
    
    return match, max_diff, mean_diff, max_rel_diff

def print_separator():
    print("=" * 120)

def main():
    print("=" * 120)
    print("RSI Benchmark: tacuda vs TA-Lib".center(120))
    print("=" * 120)
    print()

    # Тестируем на разных размерах данных
    test_configs = [
        (10000, "Small dataset"),
        (100000, "Medium dataset"),
        (500000, "Large dataset"),
        (1000000, "Very large dataset"),
    ]

    # RSI периоды для тестирования
    test_periods = [7, 9, 14, 21, 28, 50, 100, 200]

    for data_size, data_desc in test_configs:
        print(f"\n{data_desc}: {data_size:,} elements")
        print("=" * 120)
        
        # Генерация реалистичных тестовых данных (цены с трендом и волатильностью)
        np.random.seed(42)
        # Симулируем цену с трендом, сезонностью и случайными флуктуациями
        x = np.linspace(0, 100, data_size)
        trend = 0.1 * x  # восходящий тренд
        seasonal = 10 * np.sin(2 * np.pi * x / 50)  # сезонные колебания
        noise = np.random.randn(data_size) * 2  # случайный шум
        data = 100 + trend + seasonal + noise  # базовая цена 100
        
        # Убедимся что нет отрицательных цен
        data = np.maximum(data, 1.0)

        print(f"\n{'Period':<10} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} "
              f"{'Max Diff':<12} {'Mean Diff':<12} {'Max Rel %':<12}")
        print_separator()

        results = []

        for period in test_periods:
            if period >= data_size:
                continue

            # TA-Lib benchmark
            ta_result, ta_time = benchmark_talib(data.astype(np.float32), period, iterations=30)

            # tacuda benchmark
            cuda_result, cuda_time = benchmark_tacuda(data, period, iterations=30)

            # Сравнение результатов
            match, max_diff, mean_diff, max_rel_diff = compare_results(ta_result, cuda_result, period)

            speedup = ta_time / cuda_time if cuda_time > 0 else 0
            match_str = "PASS" if match else "FAIL"
            
            # Процент относительной погрешности
            max_rel_pct = max_rel_diff * 100

            print(f"{period:<10} {ta_time:<15.3f} {cuda_time:<15.3f} {speedup:<12.2f}x {match_str:<8} "
                  f"{max_diff:<12.6f} {mean_diff:<12.6f} {max_rel_pct:<12.4f}")

            results.append({
                'period': period,
                'ta_time': ta_time,
                'cuda_time': cuda_time,
                'speedup': speedup,
                'match': match,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'max_rel_diff': max_rel_diff
            })

        print_separator()
        print()

        # Статистика для текущего размера данных
        if results:
            speedups = [r['speedup'] for r in results]
            avg_speedup = np.mean(speedups)
            max_speedup = np.max(speedups)
            min_speedup = np.min(speedups)

            all_match = all(r['match'] for r in results)

            print(f"Summary for {data_desc}:")
            print(f"  Average speedup: {avg_speedup:6.2f}x")
            print(f"  Max speedup:     {max_speedup:6.2f}x")
            print(f"  Min speedup:     {min_speedup:6.2f}x")
            print(f"  All tests match: {'YES' if all_match else 'NO'}")
            
            # Throughput анализ для самого быстрого результата
            best_result = max(results, key=lambda r: r['speedup'])
            data_bytes = data_size * 4 * 2  # input + output (float32)
            time_sec = best_result['cuda_time'] / 1000
            throughput_gbps = (data_bytes / time_sec) / (1024**3)
            
            print(f"  Peak throughput: {throughput_gbps:6.2f} GB/s (period={best_result['period']})")
            print()

    print("\n" + "=" * 120)
    print("RSI Benchmark Complete!".center(120))
    print("=" * 120)
    print()
    print("Notes:")
    print("  * RSI (Relative Strength Index) - one of the most popular momentum indicators")
    print("  * RSI values are in range 0-100")
    print("  * Traditionally: RSI > 70 = overbought, RSI < 30 = oversold")
    print("  * Small differences possible due to floating point accumulation differences")
    print()

if __name__ == '__main__':
    main()

