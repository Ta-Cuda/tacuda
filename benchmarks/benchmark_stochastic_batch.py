#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark Stochastic Oscillator BATCH mode: tacuda vs TA-Lib
Тестируем производительность на батчах из множества серий
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
    sys.exit(1)

try:
    tacuda_stoch = tacuda.ct_stochastic
except AttributeError as exc:
    raise SystemExit("ct_stochastic symbol not found") from exc

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

def benchmark_talib_batch(high_batch, low_batch, close_batch, k_period, d_period, iterations=50):
    """Бенчмарк TA-Lib на батче (последовательно)"""
    num_series, series_len = high_batch.shape
    
    # Прогрев
    for _ in range(5):
        for i in range(num_series):
            _, _ = talib.STOCH(high_batch[i].astype(np.float64), 
                              low_batch[i].astype(np.float64), 
                              close_batch[i].astype(np.float64),
                              fastk_period=k_period, slowk_period=d_period,
                              slowk_matype=0, slowd_period=d_period, slowd_matype=0)
    
    start = time.perf_counter()
    for _ in range(iterations):
        k_results = []
        d_results = []
        for i in range(num_series):
            k, d = talib.STOCH(high_batch[i].astype(np.float64),
                               low_batch[i].astype(np.float64),
                               close_batch[i].astype(np.float64),
                               fastk_period=k_period, slowk_period=d_period,
                               slowk_matype=0, slowd_period=d_period, slowd_matype=0)
            k_results.append(k.astype(np.float32))
            d_results.append(d.astype(np.float32))
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return np.array(k_results), np.array(d_results), avg_time_ms

def benchmark_tacuda_batch(high_batch, low_batch, close_batch, k_period, d_period, iterations=50):
    """Бенчмарк tacuda на батче (параллельно на GPU)"""
    num_series, series_len = high_batch.shape
    
    # Series-major layout: [series][time]
    high_flat = high_batch.flatten().astype(np.float32)
    low_flat = low_batch.flatten().astype(np.float32)
    close_flat = close_batch.flatten().astype(np.float32)
    
    k_out = np.full(num_series * series_len, np.nan, dtype=np.float32)
    d_out = np.full(num_series * series_len, np.nan, dtype=np.float32)
    
    high_ptr = high_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    low_ptr = low_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    close_ptr = close_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    k_ptr = k_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    d_ptr = d_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    total_size = num_series * series_len
    
    # Настройка batch mode через environment
    os.environ['TACUDA_STOCH_ENABLE_BATCH'] = '1'
    os.environ['TACUDA_STOCH_BATCH_COUNT'] = str(num_series)
    os.environ['TACUDA_STOCH_SERIES_LEN'] = str(series_len)
    
    # Прогрев GPU
    for _ in range(5):
        tacuda_stoch(high_ptr, low_ptr, close_ptr, k_ptr, d_ptr,
                     total_size, k_period, d_period, None)
    
    start = time.perf_counter()
    for _ in range(iterations):
        tacuda_stoch(high_ptr, low_ptr, close_ptr, k_ptr, d_ptr,
                     total_size, k_period, d_period, None)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    
    # Reshape обратно в [series][time]
    k_results = k_out.reshape((num_series, series_len))
    d_results = d_out.reshape((num_series, series_len))
    
    return k_results, d_results, avg_time_ms

def compare_batch_results(ta_k_batch, ta_d_batch, cuda_k_batch, cuda_d_batch, 
                         k_period, d_period, tolerance=1e-4):
    """Сравнение batch результатов"""
    num_series = ta_k_batch.shape[0]
    lookback = k_period + 2 * (d_period - 1)
    
    all_match = True
    max_diff_k_all = 0.0
    max_diff_d_all = 0.0
    
    for i in range(num_series):
        ta_k = ta_k_batch[i][lookback:]
        cuda_k = cuda_k_batch[i][lookback:]
        ta_d = ta_d_batch[i][lookback:]
        cuda_d = cuda_d_batch[i][lookback:]
        
        mask_k = ~(np.isnan(ta_k) | np.isnan(cuda_k))
        mask_d = ~(np.isnan(ta_d) | np.isnan(cuda_d))
        
        if np.sum(mask_k) > 0:
            diff_k = np.abs(ta_k[mask_k] - cuda_k[mask_k])
            max_diff_k = np.max(diff_k)
            max_diff_k_all = max(max_diff_k_all, max_diff_k)
            
            if not np.allclose(ta_k[mask_k], cuda_k[mask_k], rtol=tolerance, atol=tolerance):
                all_match = False
        
        if np.sum(mask_d) > 0:
            diff_d = np.abs(ta_d[mask_d] - cuda_d[mask_d])
            max_diff_d = np.max(diff_d)
            max_diff_d_all = max(max_diff_d_all, max_diff_d)
            
            if not np.allclose(ta_d[mask_d], cuda_d[mask_d], rtol=tolerance, atol=tolerance):
                all_match = False
    
    return all_match, max_diff_k_all, max_diff_d_all

def print_separator():
    print("=" * 120)

def main():
    print_separator()
    print("Stochastic Oscillator BATCH Benchmark: tacuda vs TA-Lib".center(120))
    print_separator()
    print()
    
    # Конфигурации тестирования
    test_configs = [
        {"series_len": 10000, "num_series": 10, "k_period": 14, "d_period": 3},
        {"series_len": 10000, "num_series": 50, "k_period": 14, "d_period": 3},
        {"series_len": 10000, "num_series": 100, "k_period": 14, "d_period": 3},
        {"series_len": 50000, "num_series": 10, "k_period": 14, "d_period": 3},
        {"series_len": 50000, "num_series": 50, "k_period": 14, "d_period": 3},
        {"series_len": 1000, "num_series": 500, "k_period": 14, "d_period": 3},
    ]
    
    results = []
    
    for cfg in test_configs:
        series_len = cfg["series_len"]
        num_series = cfg["num_series"]
        k_period = cfg["k_period"]
        d_period = cfg["d_period"]
        
        print(f"Testing: {num_series} series x {series_len} points, K={k_period}, D={d_period}")
        
        # Генерация batch данных
        np.random.seed(42)
        x = np.linspace(0, 100, series_len)
        high_batch = []
        low_batch = []
        close_batch = []
        
        for i in range(num_series):
            base_price = 100 + 0.1 * x + 10 * np.sin(2 * np.pi * x / 50 + i * 0.1)
            volatility = np.random.randn(series_len) * 2
            close = base_price + volatility
            high = close + np.abs(np.random.randn(series_len)) * 1.5
            low = close - np.abs(np.random.randn(series_len)) * 1.5
            close = np.clip(close, low + 0.01, high - 0.01)
            
            high_batch.append(high)
            low_batch.append(low)
            close_batch.append(close)
        
        high_batch = np.array(high_batch, dtype=np.float32)
        low_batch = np.array(low_batch, dtype=np.float32)
        close_batch = np.array(close_batch, dtype=np.float32)
        
        try:
            # TA-Lib benchmark (CPU sequential)
            ta_k, ta_d, ta_time = benchmark_talib_batch(
                high_batch, low_batch, close_batch, k_period, d_period, iterations=10)
            
            # tacuda benchmark (GPU parallel)
            cuda_k, cuda_d, cuda_time = benchmark_tacuda_batch(
                high_batch, low_batch, close_batch, k_period, d_period, iterations=10)
            
            # Сравнение
            match, max_diff_k, max_diff_d = compare_batch_results(
                ta_k, ta_d, cuda_k, cuda_d, k_period, d_period)
            
            speedup = ta_time / cuda_time if cuda_time > 0 else 0
            match_str = "PASS" if match else "FAIL"
            
            print(f"  TA-Lib: {ta_time:8.2f} ms | tacuda: {cuda_time:8.2f} ms | "
                  f"Speedup: {speedup:6.2f}x | Match: {match_str} | "
                  f"MaxDiff K: {max_diff_k:.2e}, D: {max_diff_d:.2e}")
            
            results.append({
                "series_len": series_len,
                "num_series": num_series,
                "k_period": k_period,
                "d_period": d_period,
                "ta_time": ta_time,
                "cuda_time": cuda_time,
                "speedup": speedup,
                "match": match,
                "max_diff_k": max_diff_k,
                "max_diff_d": max_diff_d
            })
        except Exception as e:
            print(f"  ERROR: {e}")
        
        print()
    
    if results:
        print_separator()
        print("Summary".center(120))
        print_separator()
        
        speedups = [r["speedup"] for r in results]
        avg_speedup = np.mean(speedups)
        max_speedup = np.max(speedups)
        min_speedup = np.min(speedups)
        all_match = all(r["match"] for r in results)
        
        print(f"Average speedup: {avg_speedup:6.2f}x")
        print(f"Max speedup:     {max_speedup:6.2f}x")
        print(f"Min speedup:     {min_speedup:6.2f}x")
        print(f"All tests match: {'YES' if all_match else 'NO'}")
        print()
        
        # Throughput analysis
        print("Throughput Analysis (tacuda):")
        for r in results:
            total_elements = r["num_series"] * r["series_len"]
            # 3 inputs (H,L,C) + 2 outputs (K,D) = 5x float32
            data_bytes = total_elements * 4 * 5
            time_sec = r["cuda_time"] / 1000
            throughput_gbps = (data_bytes / time_sec) / (1024**3)
            
            print(f"  {r['num_series']:3d} series x {r['series_len']:5d} points: "
                  f"{throughput_gbps:6.2f} GB/s ({r['speedup']:6.2f}x speedup)")
        print()

if __name__ == '__main__':
    main()

