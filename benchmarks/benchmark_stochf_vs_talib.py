# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
import ctypes
import os
import sys
import time
import numpy as np
import talib

# Add dll directory for Python 3.8+
tacuda_dir = os.path.join(os.path.dirname(__file__), '..', 'build', 'Release')
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(os.path.abspath(tacuda_dir))

tacuda = ctypes.CDLL(os.path.join(tacuda_dir, 'tacuda.dll'))

# StochasticFast function: void stochf(const float* high, const float* low, const float* close,
#                                       float* outputK, float* outputD, int size,
#                                       int kPeriod, int dPeriod);
tacuda.stochf.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # high
    ctypes.POINTER(ctypes.c_float),  # low
    ctypes.POINTER(ctypes.c_float),  # close
    ctypes.POINTER(ctypes.c_float),  # output K
    ctypes.POINTER(ctypes.c_float),  # output D
    ctypes.c_int,                    # size
    ctypes.c_int,                    # kPeriod
    ctypes.c_int                     # dPeriod
]

def benchmark_stochf(size=500000, kPeriod=5, dPeriod=3, warmup=3, iterations=10):
    print(f"StochasticFast Benchmark: size={size}, kPeriod={kPeriod}, dPeriod={dPeriod}")
    print("=" * 80)
    
    # Generate test data
    np.random.seed(42)
    high = np.random.rand(size).astype(np.float32) * 100 + 100
    low = high - np.random.rand(size).astype(np.float32) * 10
    close = low + np.random.rand(size).astype(np.float32) * (high - low)
    
    # Ensure valid OHLC relationships
    low = np.minimum(low, close)
    high = np.maximum(high, close)
    
    # TA-Lib benchmark
    print("\nTA-Lib:")
    for i in range(warmup):
        k_talib, d_talib = talib.STOCHF(high, low, close, 
                                         fastk_period=kPeriod, 
                                         fastd_period=dPeriod, 
                                         fastd_matype=0)
    
    talib_times = []
    for i in range(iterations):
        start = time.perf_counter()
        k_talib, d_talib = talib.STOCHF(high, low, close, 
                                         fastk_period=kPeriod, 
                                         fastd_period=dPeriod, 
                                         fastd_matype=0)
        end = time.perf_counter()
        talib_times.append((end - start) * 1000)
    
    talib_avg = np.mean(talib_times)
    talib_std = np.std(talib_times)
    print(f"  Time: {talib_avg:.2f} +/- {talib_std:.2f} ms")
    
    # CUDA benchmark
    print("\nCUDA:")
    k_cuda = np.zeros(size, dtype=np.float32)
    d_cuda = np.zeros(size, dtype=np.float32)
    
    # Warmup
    for i in range(warmup):
        tacuda.stochf(
            high.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            low.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            close.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k_cuda.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            d_cuda.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            size, kPeriod, dPeriod
        )
    
    cuda_times = []
    for i in range(iterations):
        start = time.perf_counter()
        tacuda.stochf(
            high.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            low.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            close.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k_cuda.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            d_cuda.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            size, kPeriod, dPeriod
        )
        end = time.perf_counter()
        cuda_times.append((end - start) * 1000)
    
    cuda_avg = np.mean(cuda_times)
    cuda_std = np.std(cuda_times)
    print(f"  Time: {cuda_avg:.2f} +/- {cuda_std:.2f} ms")
    
    # Speedup
    speedup = talib_avg / cuda_avg
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Accuracy comparison
    print("\nAccuracy:")
    
    # Find first valid index in TA-Lib output
    valid_indices_k = ~np.isnan(k_talib)
    valid_indices_d = ~np.isnan(d_talib)
    
    if np.any(valid_indices_k):
        first_valid_k = np.argmax(valid_indices_k)
        
        # Compare K
        k_talib_valid = k_talib[valid_indices_k]
        k_cuda_valid = k_cuda[valid_indices_k]
        
        k_diff = np.abs(k_talib_valid - k_cuda_valid)
        k_max_diff = np.max(k_diff)
        k_mean_diff = np.mean(k_diff)
        k_match = np.allclose(k_talib_valid, k_cuda_valid, rtol=1e-4, atol=1e-2)
        
        print(f"  K: Match={k_match}, Max diff={k_max_diff:.6f}, Mean diff={k_mean_diff:.6f}")
        print(f"     First valid index: {first_valid_k}")
        print(f"     TA-Lib K[{first_valid_k}]: {k_talib[first_valid_k]:.4f}")
        print(f"     CUDA K[{first_valid_k}]:   {k_cuda[first_valid_k]:.4f}")
    
    if np.any(valid_indices_d):
        first_valid_d = np.argmax(valid_indices_d)
        
        # Compare D
        d_talib_valid = d_talib[valid_indices_d]
        d_cuda_valid = d_cuda[valid_indices_d]
        
        d_diff = np.abs(d_talib_valid - d_cuda_valid)
        d_max_diff = np.max(d_diff)
        d_mean_diff = np.mean(d_diff)
        d_match = np.allclose(d_talib_valid, d_cuda_valid, rtol=1e-4, atol=1e-2)
        
        print(f"  D: Match={d_match}, Max diff={d_max_diff:.6f}, Mean diff={d_mean_diff:.6f}")
        print(f"     First valid index: {first_valid_d}")
        print(f"     TA-Lib D[{first_valid_d}]: {d_talib[first_valid_d]:.4f}")
        print(f"     CUDA D[{first_valid_d}]:   {d_cuda[first_valid_d]:.4f}")
    
    print("\n" + "=" * 80)
    return speedup, k_match and d_match

if __name__ == '__main__':
    # Test with default parameters (TA-Lib defaults: fastk_period=5, fastd_period=3)
    benchmark_stochf(size=500000, kPeriod=5, dPeriod=3)
    
    # Test with other common parameters
    print("\n")
    benchmark_stochf(size=500000, kPeriod=14, dPeriod=3)

