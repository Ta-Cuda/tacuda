#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark SMA: tacuda vs TA-Lib
Сравниваем производительность на разных размерах окон
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
    tacuda_sma = tacuda.ct_sma
except AttributeError as exc:
    raise SystemExit("ct_sma symbol not found in tacuda library") from exc

tacuda_sma.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # size
    ctypes.c_int,                    # period
    ctypes.c_void_p                  # stream
]
tacuda_sma.restype = ctypes.c_int

def benchmark_talib(data, period, iterations=100):
    """Бенчмарк TA-Lib SMA"""
    input_arr = np.asarray(data, dtype=np.float64)
    # Прогрев
    for _ in range(5):
        _ = talib.SMA(input_arr, timeperiod=period)
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = talib.SMA(input_arr, timeperiod=period)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return result.astype(np.float32), avg_time_ms

def benchmark_tacuda(data, period, iterations=100):
    """Бенчмарк tacuda SMA"""
    size = len(data)
    input_arr = np.asarray(data, dtype=np.float32)
    output_arr = np.full(size, np.nan, dtype=np.float32)
    
    input_ptr = input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Прогрев GPU
    for _ in range(5):
        tacuda_sma(input_ptr, output_ptr, size, period, None)
    
    start = time.perf_counter()
    for _ in range(iterations):
        tacuda_sma(input_ptr, output_ptr, size, period, None)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return output_arr, avg_time_ms

def compare_results(ta_result, cuda_result, period, tolerance=1e-5):
    """Сравнение результатов"""
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
    print("=" * 100)

def main():
    print("=" * 100)
    print("SMA Benchmark: tacuda vs TA-Lib".center(100))
    print("=" * 100)
    print()
    
    # Проверка env vars
    env_vars = {
        'TACUDA_SMA_BLOCK_SIZE': os.getenv('TACUDA_SMA_BLOCK_SIZE', 'default (256)'),
        'TACUDA_SMA_SMALL_T': os.getenv('TACUDA_SMA_SMALL_T', 'default (32)'),
        'TACUDA_SMA_WARP_T': os.getenv('TACUDA_SMA_WARP_T', 'default (256)'),
        'TACUDA_SMA_PREFER_PREFIX': os.getenv('TACUDA_SMA_PREFER_PREFIX', 'default (1)'),
    }
    
    print("Configuration:")
    for key, val in env_vars.items():
        print(f"  {key}: {val}")
    print()
    
    data_size = 500000
    print(f"Data size: {data_size:,} elements")
    print()
    
    # Генерация тестовых данных (синусоида + шум)
    np.random.seed(42)
    x = np.linspace(0, 100, data_size)
    data = np.sin(x) + np.random.randn(data_size) * 0.1
    
    # Разные размеры окон
    test_windows = [5, 10, 20, 30, 50, 100, 200, 500, 1000]
    
    print_separator()
    print(f"{'Window':<10} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15} {'Mean Diff':<15}")
    print_separator()
    
    results = []
    
    for window in test_windows:
        if window >= data_size:
            continue
        
        # TA-Lib benchmark
        ta_result, ta_time = benchmark_talib(data.astype(np.float32), window, iterations=50)
        
        # tacuda benchmark
        cuda_result, cuda_time = benchmark_tacuda(data, window, iterations=50)
        
        # Сравнение результатов
        match, max_diff, mean_diff = compare_results(ta_result, cuda_result, window)
        
        speedup = ta_time / cuda_time if cuda_time > 0 else 0
        match_str = "PASS" if match else "FAIL"
        
        print(f"{window:<10} {ta_time:<15.3f} {cuda_time:<15.3f} {speedup:<12.2f}x {match_str:<8} {max_diff:<15.6e} {mean_diff:<15.6e}")
        
        results.append({
            'window': window,
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
    
    # Детальный анализ по алгоритмам
    print("Algorithm Analysis:")
    small_t = int(os.getenv('TACUDA_SMA_SMALL_T', '32'))
    warp_t = int(os.getenv('TACUDA_SMA_WARP_T', '256'))
    prefer_prefix = os.getenv('TACUDA_SMA_PREFER_PREFIX', '1') == '1'
    
    for r in results:
        w = r['window']
        if w <= small_t:
            algo = "Shared-memory tiled"
        elif w <= warp_t:
            algo = "Warp-optimized"
        elif prefer_prefix:
            algo = "Double-prefix O(N)"
        else:
            algo = "Grid-stride O(N*W)"
        print(f"  Window {w:4d}: {algo:25s} - {r['speedup']:6.2f}x")
    print()
    
    # Throughput анализ
    print("Throughput Analysis (tacuda):")
    for r in results:
        # Размер данных: input + output (2x для float32)
        data_bytes = data_size * 4 * 2
        time_sec = r['cuda_time'] / 1000
        throughput_gbps = (data_bytes / time_sec) / (1024**3)
        print(f"  Window {r['window']:4d}: {throughput_gbps:6.2f} GB/s")
    print()

if __name__ == '__main__':
    main()

