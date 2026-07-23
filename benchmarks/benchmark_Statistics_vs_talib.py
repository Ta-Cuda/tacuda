#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark Statistics Indicators: tacuda vs TA-Lib
Тестируем статистические функции: Beta, Correl, VAR, AVGDEV, Linear Regression, TSF
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

# Добавляем каталог tacuda.dll
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
    sys.exit(1)

# Определяем функции tacuda
def setup_function(name):
    """Настройка C API функции"""
    try:
        func = getattr(tacuda, f'ct_{name.lower()}')
        return func
    except AttributeError:
        return None

# Функции с одним входом
SINGLE_INPUT_FUNCS = {
    'VAR': setup_function('var'),
    'AVGDEV': setup_function('avgdev'),
    'LINEARREG': setup_function('linearreg'),
    'LINEARREG_SLOPE': setup_function('linearreg_slope'),
    'LINEARREG_ANGLE': setup_function('linearreg_angle'),
    'LINEARREG_INTERCEPT': setup_function('linearreg_intercept'),
    'TSF': setup_function('tsf'),
}

# Функции с двумя входами
DUAL_INPUT_FUNCS = {
    'BETA': setup_function('beta'),
    'CORREL': setup_function('correl'),
}

# Настройка сигнатур для одиночного входа
for name, func in SINGLE_INPUT_FUNCS.items():
    if func:
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.c_int,                    # size
            ctypes.c_int,                    # period
            ctypes.c_void_p                  # stream
        ]
        func.restype = ctypes.c_int

# Настройка сигнатур для двойного входа
for name, func in DUAL_INPUT_FUNCS.items():
    if func:
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input1
            ctypes.POINTER(ctypes.c_float),  # input2
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.c_int,                    # size
            ctypes.c_int,                    # period
            ctypes.c_void_p                  # stream
        ]
        func.restype = ctypes.c_int


def benchmark_talib_single(func_name, data, period, iterations=50):
    """Бенчмарк TA-Lib для функций с одним входом (inputs quantized to float32 for parity)"""
    input_f32 = np.asarray(data, dtype=np.float32)
    input_arr = input_f32.astype(np.float64)
    talib_func = getattr(talib, func_name)
    
    # Прогрев
    for _ in range(5):
        _ = talib_func(input_arr, timeperiod=period)
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = talib_func(input_arr, timeperiod=period)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return result.astype(np.float32), avg_time_ms


def benchmark_talib_dual(func_name, data1, data2, period, iterations=50):
    """Бенчмарк TA-Lib для функций с двумя входами (inputs quantized to float32 for parity)"""
    input1_f32 = np.asarray(data1, dtype=np.float32)
    input2_f32 = np.asarray(data2, dtype=np.float32)
    input1_arr = input1_f32.astype(np.float64)
    input2_arr = input2_f32.astype(np.float64)
    talib_func = getattr(talib, func_name)
    
    # Прогрев
    for _ in range(5):
        _ = talib_func(input1_arr, input2_arr, timeperiod=period)
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = talib_func(input1_arr, input2_arr, timeperiod=period)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return result.astype(np.float32), avg_time_ms


def benchmark_tacuda_single(func, data, period, iterations=50):
    """Бенчмарк tacuda для функций с одним входом"""
    if func is None:
        return None, 0.0
    
    size = len(data)
    input_arr = np.asarray(data, dtype=np.float32)
    output_arr = np.full(size, np.nan, dtype=np.float32)

    input_ptr = input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Прогрев GPU
    for _ in range(5):
        func(input_ptr, output_ptr, size, period, None)

    start = time.perf_counter()
    for _ in range(iterations):
        func(input_ptr, output_ptr, size, period, None)
    end = time.perf_counter()

    avg_time_ms = (end - start) / iterations * 1000
    return output_arr, avg_time_ms


def benchmark_tacuda_dual(func, data1, data2, period, iterations=50):
    """Бенчмарк tacuda для функций с двумя входами"""
    if func is None:
        return None, 0.0
    
    size = len(data1)
    input1_arr = np.asarray(data1, dtype=np.float32)
    input2_arr = np.asarray(data2, dtype=np.float32)
    output_arr = np.full(size, np.nan, dtype=np.float32)

    input1_ptr = input1_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    input2_ptr = input2_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Прогрев GPU
    for _ in range(5):
        func(input1_ptr, input2_ptr, output_ptr, size, period, None)

    start = time.perf_counter()
    for _ in range(iterations):
        func(input1_ptr, input2_ptr, output_ptr, size, period, None)
    end = time.perf_counter()

    avg_time_ms = (end - start) / iterations * 1000
    return output_arr, avg_time_ms


def compare_results(ta_result, cuda_result, period, tolerance=1e-4):
    """Сравнение результатов"""
    if cuda_result is None:
        return False, 0.0, 0.0
    
    # Убираем NaN для сравнения
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


def test_indicator(func_name, func, data1, data2, period, is_dual=False):
    """Тестирование одного индикатора"""
    try:
        if is_dual:
            ta_result, ta_time = benchmark_talib_dual(func_name, data1, data2, period)
            cuda_result, cuda_time = benchmark_tacuda_dual(func, data1, data2, period)
        else:
            ta_result, ta_time = benchmark_talib_single(func_name, data1, period)
            cuda_result, cuda_time = benchmark_tacuda_single(func, data1, period)
        
        match, max_diff, mean_diff = compare_results(ta_result, cuda_result, period)
        speedup = ta_time / cuda_time if cuda_time > 0 else 0
        match_str = "PASS" if match else "x FAIL"
        
        if func is None:
            status = "N/A (not implemented)"
            return None
        
        return {
            'name': func_name,
            'period': period,
            'ta_time': ta_time,
            'cuda_time': cuda_time,
            'speedup': speedup,
            'match': match,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'match_str': match_str
        }
    except Exception as e:
        print(f"  Error testing {func_name}: {e}")
        return None


def main():
    print_separator()
    print("Statistics Indicators Benchmark: tacuda vs TA-Lib".center(110))
    print("Testing: VAR, AVGDEV, Linear Regression, Beta, Correl, TSF".center(110))
    print_separator()
    print()

    data_size = 500000
    print(f"Data size: {data_size:,} elements")
    print()

    # Генерация тестовых данных
    np.random.seed(42)
    
    # Основная серия (имитация цены)
    x = np.linspace(0, 100, data_size)
    data1 = 100 + np.sin(x) * 10 + np.random.randn(data_size) * 2
    
    # Вторая серия (для Beta и Correl) - коррелирующая с первой
    data2 = 100 + np.sin(x + 0.5) * 12 + np.random.randn(data_size) * 2.5

    # Тестовые периоды
    test_periods = [5, 10, 14, 20, 30, 50, 100]

    all_results = []

    # === SINGLE INPUT INDICATORS ===
    print("=" * 110)
    print("SINGLE INPUT STATISTICS".center(110))
    print("=" * 110)
    print()
    
    for func_name, func in SINGLE_INPUT_FUNCS.items():
        if func is None:
            print(f"\n{func_name}: Not implemented in tacuda")
            continue
        
        print(f"\n{func_name}:")
        print("-" * 110)
        print(f"{'Period':<10} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<12} {'Max Diff':<15} {'Mean Diff':<15}")
        print("-" * 110)
        
        for period in test_periods:
            if period >= data_size:
                continue
            
            result = test_indicator(func_name, func, data1, data2, period, is_dual=False)
            if result:
                print(f"{result['period']:<10} {result['ta_time']:<15.3f} {result['cuda_time']:<15.3f} "
                      f"{result['speedup']:<12.2f}x {result['match_str']:<12} "
                      f"{result['max_diff']:<15.6e} {result['mean_diff']:<15.6e}")
                all_results.append(result)

    # === DUAL INPUT INDICATORS ===
    print("\n")
    print("=" * 110)
    print("DUAL INPUT STATISTICS (require 2 series)".center(110))
    print("=" * 110)
    print()
    
    for func_name, func in DUAL_INPUT_FUNCS.items():
        if func is None:
            print(f"\n{func_name}: Not implemented in tacuda")
            continue
        
        print(f"\n{func_name}:")
        print("-" * 110)
        print(f"{'Period':<10} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<12} {'Max Diff':<15} {'Mean Diff':<15}")
        print("-" * 110)
        
        for period in test_periods:
            if period >= data_size:
                continue
            
            result = test_indicator(func_name, func, data1, data2, period, is_dual=True)
            if result:
                print(f"{result['period']:<10} {result['ta_time']:<15.3f} {result['cuda_time']:<15.3f} "
                      f"{result['speedup']:<12.2f}x {result['match_str']:<12} "
                      f"{result['max_diff']:<15.6e} {result['mean_diff']:<15.6e}")
                all_results.append(result)

    # === SUMMARY ===
    print("\n")
    print_separator()
    print("OVERALL SUMMARY".center(110))
    print_separator()
    print()

    if all_results:
        speedups = [r['speedup'] for r in all_results if r['speedup'] > 0]
        matches = [r for r in all_results if r['match']]
        
        print(f"Total tests run:        {len(all_results)}")
        print(f"Tests passed:           {len(matches)} ({len(matches)/len(all_results)*100:.1f}%)")
        print(f"Average speedup:        {np.mean(speedups):.2f}x")
        print(f"Max speedup:            {np.max(speedups):.2f}x")
        print(f"Min speedup:            {np.min(speedups):.2f}x")
        print()
        
        # Группировка по индикаторам
        by_indicator = {}
        for r in all_results:
            name = r['name']
            if name not in by_indicator:
                by_indicator[name] = []
            by_indicator[name].append(r)
        
        print("Performance by Indicator:")
        print("-" * 110)
        print(f"{'Indicator':<25} {'Avg Speedup':<15} {'Tests':<10} {'Passed':<10} {'Status':<15}")
        print("-" * 110)
        
        for name, results in sorted(by_indicator.items()):
            avg_speed = np.mean([r['speedup'] for r in results if r['speedup'] > 0])
            passed = sum(1 for r in results if r['match'])
            total = len(results)
            status = "OK" if passed == total else "x ISSUES"
            print(f"{name:<25} {avg_speed:<15.2f}x {total:<10} {passed:<10} {status:<15}")
        
        print("-" * 110)
    else:
        print("No results collected. Make sure tacuda indicators are implemented.")
    
    print()
    print_separator()
    
    # Throughput для успешных тестов
    if all_results:
        print()
        print("Throughput Analysis (tacuda):")
        print("-" * 110)
        for r in all_results[:10]:  # Показываем первые 10
            # Размер данных: input + output для одиночных, input1 + input2 + output для двойных
            is_dual = r['name'] in DUAL_INPUT_FUNCS
            multiplier = 3 if is_dual else 2
            data_bytes = data_size * 4 * multiplier
            time_sec = r['cuda_time'] / 1000
            throughput_gbps = (data_bytes / time_sec) / (1024**3) if time_sec > 0 else 0
            print(f"  {r['name']:<20} Period {r['period']:4d}: {throughput_gbps:6.2f} GB/s")
        print()


if __name__ == '__main__':
    main()

