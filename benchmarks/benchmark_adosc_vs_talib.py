#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark ADOSC (Chaikin A/D Oscillator): tacuda vs TA-Lib
"""

import time
import numpy as np
import talib
import ctypes
import os
import sys

# Load tacuda
tacuda_lib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'Release', 'tacuda.dll')
if not os.path.exists(tacuda_lib_path):
    tacuda_lib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'libtacuda.so')

cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
if os.path.exists(cuda_path):
    os.environ['PATH'] = cuda_path + os.pathsep + os.environ.get('PATH', '')
    try:
        os.add_dll_directory(cuda_path)
    except (AttributeError, FileNotFoundError):
        pass

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
    tacuda_adosc = tacuda.ct_adosc
except AttributeError as exc:
    raise SystemExit("ct_adosc symbol not found in tacuda library") from exc

# ct_adosc(const float* high, const float* low, const float* close, const float* volume,
#          float* output, int size, int fast_period, int slow_period, cudaStream_t stream)
tacuda_adosc.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # high
    ctypes.POINTER(ctypes.c_float),  # low
    ctypes.POINTER(ctypes.c_float),  # close
    ctypes.POINTER(ctypes.c_float),  # volume
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # size
    ctypes.c_int,                    # fast_period
    ctypes.c_int,                    # slow_period
    ctypes.c_void_p                  # stream
]
tacuda_adosc.restype = ctypes.c_int

def benchmark_talib(high, low, close, volume, fast_period, slow_period, iterations=50):
    """Benchmark TA-Lib ADOSC"""
    high_arr = np.asarray(high, dtype=np.float64)
    low_arr = np.asarray(low, dtype=np.float64)
    close_arr = np.asarray(close, dtype=np.float64)
    volume_arr = np.asarray(volume, dtype=np.float64)
    
    # Warmup
    for _ in range(5):
        _ = talib.ADOSC(high_arr, low_arr, close_arr, volume_arr,
                       fastperiod=fast_period, slowperiod=slow_period)
    
    start = time.perf_counter()
    for _ in range(iterations):
        result = talib.ADOSC(high_arr, low_arr, close_arr, volume_arr,
                            fastperiod=fast_period, slowperiod=slow_period)
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / iterations * 1000
    return result.astype(np.float32), avg_time_ms

def benchmark_tacuda(high, low, close, volume, fast_period, slow_period, iterations=50):
    """Benchmark tacuda ADOSC"""
    size = len(high)
    high_arr = np.asarray(high, dtype=np.float32)
    low_arr = np.asarray(low, dtype=np.float32)
    close_arr = np.asarray(close, dtype=np.float32)
    volume_arr = np.asarray(volume, dtype=np.float32)
    
    output = np.full(size, np.nan, dtype=np.float32)

    high_ptr = high_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    low_ptr = low_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    close_ptr = close_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    volume_ptr = volume_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Warmup GPU
    for _ in range(5):
        tacuda_adosc(high_ptr, low_ptr, close_ptr, volume_ptr, output_ptr,
                    size, fast_period, slow_period, None)

    start = time.perf_counter()
    for _ in range(iterations):
        tacuda_adosc(high_ptr, low_ptr, close_ptr, volume_ptr, output_ptr,
                    size, fast_period, slow_period, None)
    end = time.perf_counter()

    avg_time_ms = (end - start) / iterations * 1000
    return output, avg_time_ms

def compare_results(ta_result, cuda_result, fast_period, slow_period, tolerance=1e-4):
    """Compare ADOSC results"""
    # ADOSC lookback = slow_period - 1
    lookback = slow_period - 1
    
    ta_valid = ta_result[lookback:]
    cuda_valid = cuda_result[lookback:]
    
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
    print("=" * 120)

def main():
    print_separator()
    print("ADOSC (Chaikin A/D Oscillator) Benchmark: tacuda vs TA-Lib".center(120))
    print_separator()
    print()

    data_size = 500000
    print(f"Data size: {data_size:,} elements")
    print()

    # Generate realistic OHLCV data
    np.random.seed(42)
    
    x = np.linspace(0, 100, data_size)
    base_price = 100 + 0.1 * x + 10 * np.sin(2 * np.pi * x / 50)
    volatility = np.random.randn(data_size) * 2
    
    close = base_price + volatility
    high = close + np.abs(np.random.randn(data_size)) * 1.5
    low = close - np.abs(np.random.randn(data_size)) * 1.5
    close = np.clip(close, low + 0.01, high - 0.01)
    
    # Volume with trend
    volume = np.abs(np.random.randn(data_size)) * 1000000 + 500000

    # Test configurations (fast_period, slow_period)
    test_configs = [
        (3, 10, "Standard ADOSC (3,10)"),
        (5, 15, "Medium ADOSC (5,15)"),
        (10, 20, "Long ADOSC (10,20)"),
    ]

    print_separator()
    print(f"{'Config':<30} {'Fast Period':<14} {'Slow Period':<14} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} "
          f"{'Speedup':<12} {'Match':<8}")
    print_separator()

    results = []

    for fast_period, slow_period, description in test_configs:
        lookback = slow_period - 1
        if lookback >= data_size:
            continue

        try:
            # TA-Lib benchmark
            ta_result, ta_time = benchmark_talib(high, low, close, volume, fast_period, slow_period, iterations=50)

            # tacuda benchmark
            cuda_result, cuda_time = benchmark_tacuda(high, low, close, volume, fast_period, slow_period, iterations=50)

            # Compare results
            match, max_diff, mean_diff = compare_results(ta_result, cuda_result, fast_period, slow_period)

            speedup = ta_time / cuda_time if cuda_time > 0 else 0
            match_str = "PASS" if match else "FAIL"

            print(f"{description:<30} {fast_period:<14} {slow_period:<14} {ta_time:<15.3f} {cuda_time:<15.3f} "
                  f"{speedup:<12.2f}x {match_str:<8}")

            results.append({
                'description': description,
                'fast_period': fast_period,
                'slow_period': slow_period,
                'ta_time': ta_time,
                'cuda_time': cuda_time,
                'speedup': speedup,
                'match': match,
                'max_diff': max_diff,
                'mean_diff': mean_diff
            })
        
        except Exception as e:
            print(f"{description:<30} ERROR: {e}")

    print_separator()
    print()

    if results:
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

        print("Accuracy Analysis:")
        for r in results:
            print(f"  {r['description']:<30}: Max diff: {r['max_diff']:.6e}, Mean diff: {r['mean_diff']:.6e}")
        print()

        print("Throughput Analysis (tacuda):")
        for r in results:
            # 4 inputs (H,L,C,V) + 1 output = 5x for float32
            data_bytes = data_size * 4 * 5
            time_sec = r['cuda_time'] / 1000
            throughput_gbps = (data_bytes / time_sec) / (1024**3)
            print(f"  {r['description']:<30}: {throughput_gbps:6.2f} GB/s")
        print()

    print_separator()

if __name__ == '__main__':
    main()

