#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Comprehensive Benchmark: Overlap Studies - tacuda vs TA-Lib
Tests SAR, SAREXT, Stochastic, StochasticFast, ULTOSC
"""

import time
import numpy as np
import talib
import ctypes
import os
import sys

# ==================== CUDA Library Loading ====================
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
    print(f"Library path: {tacuda_lib_path}")
    sys.exit(1)

# ==================== Function Signatures ====================

# SAR: ct_sar(high, low, output, size, acceleration, maximum)
try:
    tacuda_sar = tacuda.ct_sar
    tacuda_sar.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # high
        ctypes.POINTER(ctypes.c_float),  # low
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int,                     # size
        ctypes.c_float,                   # acceleration
        ctypes.c_float,                   # maximum
        ctypes.c_void_p                   # stream
    ]
    tacuda_sar.restype = ctypes.c_int
except AttributeError:
    tacuda_sar = None
    print("Warning: ct_sar not found")

# Stochastic: ct_stochastic(high, low, close, outK, outD, size, kPeriod, dPeriod)
try:
    tacuda_stoch = tacuda.ct_stochastic
    tacuda_stoch.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # high
        ctypes.POINTER(ctypes.c_float),  # low
        ctypes.POINTER(ctypes.c_float),  # close
        ctypes.POINTER(ctypes.c_float),  # outK
        ctypes.POINTER(ctypes.c_float),  # outD
        ctypes.c_int,                     # size
        ctypes.c_int,                     # kPeriod
        ctypes.c_int,                     # dPeriod
        ctypes.c_void_p                   # stream
    ]
    tacuda_stoch.restype = ctypes.c_int
except AttributeError:
    tacuda_stoch = None
    print("Warning: ct_stochastic not found")

# StochasticFast
try:
    tacuda_stochf = tacuda.ct_stochf
    tacuda_stochf.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # high
        ctypes.POINTER(ctypes.c_float),  # low
        ctypes.POINTER(ctypes.c_float),  # close
        ctypes.POINTER(ctypes.c_float),  # outK
        ctypes.POINTER(ctypes.c_float),  # outD
        ctypes.c_int,                     # size
        ctypes.c_int,                     # fastK_period
        ctypes.c_int,                     # fastD_period
        ctypes.c_int,                     # fastD_matype
        ctypes.c_void_p                   # stream
    ]
    tacuda_stochf.restype = ctypes.c_int
except AttributeError:
    tacuda_stochf = None
    print("Warning: ct_stochf not found")

# ULTOSC
try:
    tacuda_ultosc = tacuda.ct_ultosc
    tacuda_ultosc.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # high
        ctypes.POINTER(ctypes.c_float),  # low
        ctypes.POINTER(ctypes.c_float),  # close
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int,                     # size
        ctypes.c_int,                     # period1
        ctypes.c_int,                     # period2
        ctypes.c_int,                     # period3
        ctypes.c_void_p                   # stream
    ]
    tacuda_ultosc.restype = ctypes.c_int
except AttributeError:
    tacuda_ultosc = None
    print("Warning: ct_ultosc not found")

# ==================== Data Generation ====================

def generate_ohlc_data(size, seed=42):
    """Generate realistic OHLC data"""
    np.random.seed(seed)
    
    # Generate close prices (trending with noise)
    x = np.linspace(0, 100, size)
    trend = 100 + 20 * np.sin(x / 10)
    noise = np.random.randn(size) * 2
    close = trend + noise
    close = np.maximum(close, 1.0)  # Ensure positive
    
    # Generate high/low around close
    high_offset = np.abs(np.random.randn(size) * 3)
    low_offset = np.abs(np.random.randn(size) * 3)
    
    high = close + high_offset
    low = close - low_offset
    
    # Ensure high >= close >= low
    high = np.maximum(high, close)
    low = np.minimum(low, close)
    
    return high.astype(np.float32), low.astype(np.float32), close.astype(np.float32)

# ==================== SAR Benchmark ====================

def benchmark_sar(high, low, acceleration=0.02, maximum=0.2, iterations=50):
    """Benchmark SAR (Parabolic Stop and Reverse)"""
    if tacuda_sar is None:
        return None, None
    
    size = len(high)
    
    # TA-Lib
    ta_high = high.astype(np.float64)
    ta_low = low.astype(np.float64)
    for _ in range(5):
        _ = talib.SAR(ta_high, ta_low, acceleration=acceleration, maximum=maximum)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        ta_result = talib.SAR(ta_high, ta_low, acceleration=acceleration, maximum=maximum)
    t1 = time.perf_counter()
    ta_time = (t1 - t0) / iterations * 1000
    
    # tacuda
    cu_high = high.copy()
    cu_low = low.copy()
    cu_result = np.full(size, np.nan, dtype=np.float32)
    
    high_ptr = cu_high.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    low_ptr = cu_low.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = cu_result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    for _ in range(5):
        tacuda_sar(high_ptr, low_ptr, out_ptr, size, acceleration, maximum, None)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        tacuda_sar(high_ptr, low_ptr, out_ptr, size, acceleration, maximum, None)
    t1 = time.perf_counter()
    cu_time = (t1 - t0) / iterations * 1000
    
    return (ta_result.astype(np.float32), ta_time), (cu_result, cu_time)

# ==================== Stochastic Benchmark ====================

def benchmark_stochastic(high, low, close, kperiod=5, dperiod=3, iterations=50):
    """Benchmark Stochastic Oscillator"""
    if tacuda_stoch is None:
        return None, None
    
    size = len(high)
    
    # TA-Lib - Note: TA-Lib STOCH has 3 periods: fastk, slowk, slowd
    # Our tacuda Stochastic is simplified with just kPeriod and dPeriod
    # To match: fastk=kperiod, slowk=dperiod, slowd=dperiod
    ta_high = high.astype(np.float64)
    ta_low = low.astype(np.float64)
    ta_close = close.astype(np.float64)
    
    for _ in range(5):
        _, _ = talib.STOCH(ta_high, ta_low, ta_close, 
                          fastk_period=kperiod, slowk_period=dperiod, 
                          slowk_matype=0, slowd_period=dperiod, slowd_matype=0)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        ta_k, ta_d = talib.STOCH(ta_high, ta_low, ta_close,
                                fastk_period=kperiod, slowk_period=dperiod,
                                slowk_matype=0, slowd_period=dperiod, slowd_matype=0)
    t1 = time.perf_counter()
    ta_time = (t1 - t0) / iterations * 1000
    
    # tacuda
    cu_high = high.copy()
    cu_low = low.copy()
    cu_close = close.copy()
    cu_k = np.full(size, np.nan, dtype=np.float32)
    cu_d = np.full(size, np.nan, dtype=np.float32)
    
    high_ptr = cu_high.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    low_ptr = cu_low.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    close_ptr = cu_close.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    k_ptr = cu_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    d_ptr = cu_d.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    for _ in range(5):
        tacuda_stoch(high_ptr, low_ptr, close_ptr, k_ptr, d_ptr,
                    size, kperiod, dperiod, None)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        tacuda_stoch(high_ptr, low_ptr, close_ptr, k_ptr, d_ptr,
                    size, kperiod, dperiod, None)
    t1 = time.perf_counter()
    cu_time = (t1 - t0) / iterations * 1000
    
    ta_result = np.stack([ta_k.astype(np.float32), ta_d.astype(np.float32)])
    cu_result = np.stack([cu_k, cu_d])
    
    return (ta_result, ta_time), (cu_result, cu_time)

# ==================== Stochastic Fast Benchmark ====================

def benchmark_stochastic_fast(high, low, close, fastk=5, fastd=3, iterations=50):
    """Benchmark Fast Stochastic"""
    if tacuda_stochf is None:
        return None, None
    
    size = len(high)
    
    # TA-Lib
    ta_high = high.astype(np.float64)
    ta_low = low.astype(np.float64)
    ta_close = close.astype(np.float64)
    
    for _ in range(5):
        _, _ = talib.STOCHF(ta_high, ta_low, ta_close,
                           fastk_period=fastk, fastd_period=fastd, fastd_matype=0)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        ta_k, ta_d = talib.STOCHF(ta_high, ta_low, ta_close,
                                 fastk_period=fastk, fastd_period=fastd, fastd_matype=0)
    t1 = time.perf_counter()
    ta_time = (t1 - t0) / iterations * 1000
    
    # tacuda
    cu_high = high.copy()
    cu_low = low.copy()
    cu_close = close.copy()
    cu_k = np.full(size, np.nan, dtype=np.float32)
    cu_d = np.full(size, np.nan, dtype=np.float32)
    
    high_ptr = cu_high.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    low_ptr = cu_low.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    close_ptr = cu_close.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    k_ptr = cu_k.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    d_ptr = cu_d.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    for _ in range(5):
        tacuda_stochf(high_ptr, low_ptr, close_ptr, k_ptr, d_ptr,
                     size, fastk, fastd, 0, None)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        tacuda_stochf(high_ptr, low_ptr, close_ptr, k_ptr, d_ptr,
                     size, fastk, fastd, 0, None)
    t1 = time.perf_counter()
    cu_time = (t1 - t0) / iterations * 1000
    
    ta_result = np.stack([ta_k.astype(np.float32), ta_d.astype(np.float32)])
    cu_result = np.stack([cu_k, cu_d])
    
    return (ta_result, ta_time), (cu_result, cu_time)

# ==================== ULTOSC Benchmark ====================

def benchmark_ultosc(high, low, close, p1=7, p2=14, p3=28, iterations=50):
    """Benchmark Ultimate Oscillator"""
    if tacuda_ultosc is None:
        return None, None
    
    size = len(high)
    
    # TA-Lib
    ta_high = high.astype(np.float64)
    ta_low = low.astype(np.float64)
    ta_close = close.astype(np.float64)
    
    for _ in range(5):
        _ = talib.ULTOSC(ta_high, ta_low, ta_close, timeperiod1=p1, timeperiod2=p2, timeperiod3=p3)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        ta_result = talib.ULTOSC(ta_high, ta_low, ta_close, timeperiod1=p1, timeperiod2=p2, timeperiod3=p3)
    t1 = time.perf_counter()
    ta_time = (t1 - t0) / iterations * 1000
    
    # tacuda
    cu_high = high.copy()
    cu_low = low.copy()
    cu_close = close.copy()
    cu_result = np.full(size, np.nan, dtype=np.float32)
    
    high_ptr = cu_high.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    low_ptr = cu_low.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    close_ptr = cu_close.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = cu_result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    for _ in range(5):
        tacuda_ultosc(high_ptr, low_ptr, close_ptr, out_ptr, size, p1, p2, p3, None)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        tacuda_ultosc(high_ptr, low_ptr, close_ptr, out_ptr, size, p1, p2, p3, None)
    t1 = time.perf_counter()
    cu_time = (t1 - t0) / iterations * 1000
    
    return (ta_result.astype(np.float32), ta_time), (cu_result, cu_time)

# ==================== Comparison ====================

def compare_results(ta_result, cu_result, tolerance=1e-4):
    """Compare results between TA-Lib and tacuda"""
    if ta_result.ndim == 1:
        # Single output
        mask = ~(np.isnan(ta_result) | np.isnan(cu_result))
        if not np.any(mask):
            return False, 0.0, 0.0
        
        diff = np.abs(ta_result[mask] - cu_result[mask])
        max_diff = np.max(diff) if len(diff) > 0 else 0.0
        mean_diff = np.mean(diff) if len(diff) > 0 else 0.0
        match = np.allclose(ta_result[mask], cu_result[mask], rtol=tolerance, atol=tolerance)
        
    else:
        # Multiple outputs (e.g., Stochastic K and D)
        all_match = True
        max_diffs = []
        mean_diffs = []
        
        for i in range(ta_result.shape[0]):
            mask = ~(np.isnan(ta_result[i]) | np.isnan(cu_result[i]))
            if not np.any(mask):
                continue
            
            diff = np.abs(ta_result[i][mask] - cu_result[i][mask])
            max_diffs.append(np.max(diff) if len(diff) > 0 else 0.0)
            mean_diffs.append(np.mean(diff) if len(diff) > 0 else 0.0)
            
            if not np.allclose(ta_result[i][mask], cu_result[i][mask], rtol=tolerance, atol=tolerance):
                all_match = False
        
        match = all_match
        max_diff = max(max_diffs) if max_diffs else 0.0
        mean_diff = np.mean(mean_diffs) if mean_diffs else 0.0
    
    return match, max_diff, mean_diff

# ==================== Main ====================

def print_separator():
    print("=" * 110)

def main():
    print_separator()
    print("Overlap Studies Benchmark: tacuda vs TA-Lib".center(110))
    print("Testing: SAR, Stochastic, StochasticFast, ULTOSC".center(110))
    print_separator()
    print()
    
    data_size = 100000
    print(f"Data size: {data_size:,} elements (High, Low, Close arrays)")
    print()
    
    # Generate OHLC data
    print("Generating realistic OHLC data...")
    high, low, close = generate_ohlc_data(data_size)
    print(f"Price range: Low={np.min(low):.2f}, High={np.max(high):.2f}, Close avg={np.mean(close):.2f}")
    print()
    
    all_results = []
    
    # ==================== SAR ====================
    print_separator()
    print("1. SAR (Parabolic Stop and Reverse)".center(110))
    print_separator()
    print(f"{'Params':<25} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15}")
    print_separator()
    
    sar_configs = [
        (0.02, 0.2, "default"),
        (0.01, 0.15, "conservative"),
        (0.05, 0.25, "aggressive"),
    ]
    
    for accel, maxim, label in sar_configs:
        ta_res, cu_res = benchmark_sar(high, low, accel, maxim)
        if ta_res is None:
            print(f"{label:<25} SKIPPED (not implemented)")
            continue
        
        ta_result, ta_time = ta_res
        cu_result, cu_time = cu_res
        
        match, max_diff, mean_diff = compare_results(ta_result, cu_result)
        speedup = ta_time / cu_time if cu_time > 0 else 0
        
        params_str = f"accel={accel}, max={maxim}"
        match_str = "PASS" if match else "FAIL"
        print(f"{params_str:<25} {ta_time:<15.3f} {cu_time:<15.3f} {speedup:<12.2f}x {match_str:<8} {max_diff:<15.6e}")
        
        all_results.append({
            'indicator': 'SAR',
            'params': label,
            'speedup': speedup,
            'match': match
        })
    
    print()
    
    # ==================== Stochastic ====================
    print_separator()
    print("2. Stochastic Oscillator".center(110))
    print_separator()
    print(f"{'Params':<25} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15}")
    print_separator()
    
    stoch_configs = [
        (5, 3, "default (5,3)"),
        (14, 3, "standard (14,3)"),
        (21, 5, "smooth (21,5)"),
    ]
    
    for kperiod, dperiod, label in stoch_configs:
        ta_res, cu_res = benchmark_stochastic(high, low, close, kperiod, dperiod)
        if ta_res is None:
            print(f"{label:<25} SKIPPED (not implemented)")
            continue
        
        ta_result, ta_time = ta_res
        cu_result, cu_time = cu_res
        
        match, max_diff, mean_diff = compare_results(ta_result, cu_result)
        speedup = ta_time / cu_time if cu_time > 0 else 0
        
        match_str = "PASS" if match else "FAIL"
        print(f"{label:<25} {ta_time:<15.3f} {cu_time:<15.3f} {speedup:<12.2f}x {match_str:<8} {max_diff:<15.6e}")
        
        all_results.append({
            'indicator': 'Stochastic',
            'params': label,
            'speedup': speedup,
            'match': match
        })
    
    print()
    
    # ==================== Stochastic Fast ====================
    print_separator()
    print("3. Stochastic Fast".center(110))
    print_separator()
    print(f"{'Params':<25} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15}")
    print_separator()
    
    stochf_configs = [
        (5, 3, "default (5,3)"),
        (14, 3, "standard (14,3)"),
        (21, 5, "smooth (21,5)"),
    ]
    
    for fastk, fastd, label in stochf_configs:
        ta_res, cu_res = benchmark_stochastic_fast(high, low, close, fastk, fastd)
        if ta_res is None:
            print(f"{label:<25} SKIPPED (not implemented)")
            continue
        
        ta_result, ta_time = ta_res
        cu_result, cu_time = cu_res
        
        match, max_diff, mean_diff = compare_results(ta_result, cu_result)
        speedup = ta_time / cu_time if cu_time > 0 else 0
        
        match_str = "PASS" if match else "FAIL"
        print(f"{label:<25} {ta_time:<15.3f} {cu_time:<15.3f} {speedup:<12.2f}x {match_str:<8} {max_diff:<15.6e}")
        
        all_results.append({
            'indicator': 'StochasticFast',
            'params': label,
            'speedup': speedup,
            'match': match
        })
    
    print()
    
    # ==================== ULTOSC ====================
    print_separator()
    print("4. Ultimate Oscillator (ULTOSC)".center(110))
    print_separator()
    print(f"{'Params':<25} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15}")
    print_separator()
    
    ultosc_configs = [
        (7, 14, 28, "default (7,14,28)"),
        (5, 10, 20, "fast (5,10,20)"),
        (10, 20, 40, "slow (10,20,40)"),
    ]
    
    for p1, p2, p3, label in ultosc_configs:
        ta_res, cu_res = benchmark_ultosc(high, low, close, p1, p2, p3)
        if ta_res is None:
            print(f"{label:<25} SKIPPED (not implemented)")
            continue
        
        ta_result, ta_time = ta_res
        cu_result, cu_time = cu_res
        
        match, max_diff, mean_diff = compare_results(ta_result, cu_result)
        speedup = ta_time / cu_time if cu_time > 0 else 0
        
        match_str = "PASS" if match else "FAIL"
        print(f"{label:<25} {ta_time:<15.3f} {cu_time:<15.3f} {speedup:<12.2f}x {match_str:<8} {max_diff:<15.6e}")
        
        all_results.append({
            'indicator': 'ULTOSC',
            'params': label,
            'speedup': speedup,
            'match': match
        })
    
    print()
    
    # ==================== Summary ====================
    print_separator()
    print("OVERALL SUMMARY".center(110))
    print_separator()
    
    if all_results:
        speedups = [r['speedup'] for r in all_results if r['speedup'] > 0]
        matches = [r['match'] for r in all_results]
        
        print(f"Total tests run:       {len(all_results)}")
        print(f"Tests passed:          {sum(matches)}")
        print(f"Tests failed:          {len(matches) - sum(matches)}")
        print()
        print(f"Average speedup:       {np.mean(speedups):.2f}x")
        print(f"Max speedup:           {np.max(speedups):.2f}x")
        print(f"Min speedup:           {np.min(speedups):.2f}x")
        print()
        
        # By indicator
        print("Performance by Indicator:")
        for indicator_name in ['SAR', 'Stochastic', 'StochasticFast', 'ULTOSC']:
            ind_results = [r for r in all_results if r['indicator'] == indicator_name]
            if ind_results:
                ind_speedups = [r['speedup'] for r in ind_results if r['speedup'] > 0]
                avg_speedup = np.mean(ind_speedups) if ind_speedups else 0
                print(f"  {indicator_name:<20} {avg_speedup:6.2f}x")
    else:
        print("No tests were run (indicators not implemented)")
    
    print_separator()

if __name__ == '__main__':
    main()

