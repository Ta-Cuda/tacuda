#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark Hilbert Transform indicators: tacuda vs TA-Lib
Tests: HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE, HT_TRENDLINE
"""

import numpy as np
import time
import sys
import os
import ctypes

import talib

# Load tacuda library
tacuda_lib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'Release', 'tacuda.dll')
if not os.path.exists(tacuda_lib_path):
    tacuda_lib_path = os.path.join(os.path.dirname(__file__), '..', 'build', 'libtacuda.so')

# Add CUDA DLL path
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
if os.path.exists(cuda_path):
    os.environ['PATH'] = cuda_path + os.pathsep + os.environ.get('PATH', '')
    try:
        os.add_dll_directory(cuda_path)
    except (AttributeError, FileNotFoundError):
        pass

# Add tacuda.dll directory
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

# Load Hilbert Transform functions
# Standard HT function signature: input, output, size, stream
ht_single_argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # size
    ctypes.c_void_p                  # stream
]

# Dual output signature: input, output1, output2, size, stream
ht_dual_argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # output1
    ctypes.POINTER(ctypes.c_float),  # output2
    ctypes.c_int,                    # size
    ctypes.c_void_p                  # stream
]

def load_ht_function(name, argtypes):
    try:
        func = getattr(tacuda, name)
        func.argtypes = argtypes
        func.restype = ctypes.c_int
        return func
    except AttributeError:
        print(f"Warning: {name} not found")
        return None

tacuda_ht_dcperiod = load_ht_function('ct_ht_dcperiod', ht_single_argtypes)
tacuda_ht_dcphase = load_ht_function('ct_ht_dcphase', ht_single_argtypes)
tacuda_ht_phasor = load_ht_function('ct_ht_phasor', ht_dual_argtypes)
tacuda_ht_sine = load_ht_function('ct_ht_sine', ht_dual_argtypes)
tacuda_ht_trendmode = load_ht_function('ct_ht_trendmode', ht_single_argtypes)
tacuda_ht_trendline = load_ht_function('ct_ht_trendline', ht_single_argtypes)


def print_header(title, width=110):
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)

def print_row(*args, widths=None):
    if widths is None:
        widths = [16] * len(args)
    row = ""
    for arg, w in zip(args, widths):
        if isinstance(arg, float):
            row += f"{arg:<{w}.3f}"
        else:
            row += f"{str(arg):<{w}}"
    print(row)


def benchmark_single_output(name, ta_func, cuda_func, data, warmup=3, iterations=10):
    """Benchmark single-output HT indicator"""
    size = len(data)
    output = np.zeros(size, dtype=np.float32)
    data64 = data.astype(np.float64)  # TA-Lib requires float64
    
    in_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Warmup
    for _ in range(warmup):
        _ = ta_func(data64)
        cuda_func(in_ptr, out_ptr, size, None)
    
    # Benchmark TA-Lib
    ta_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        ta_result = ta_func(data64)
        ta_times.append((time.perf_counter() - start) * 1000)
    ta_time = np.median(ta_times)
    
    # Benchmark tacuda
    cuda_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        cuda_func(in_ptr, out_ptr, size, None)
        cuda_times.append((time.perf_counter() - start) * 1000)
    cuda_time = np.median(cuda_times)
    
    cuda_result = output.copy()
    speedup = ta_time / cuda_time if cuda_time > 0 else 0
    
    # Compare results
    mask = ~np.isnan(ta_result) & ~np.isnan(cuda_result)
    if np.sum(mask) > 0:
        diff = np.abs(ta_result[mask] - cuda_result[mask])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        all_match = max_diff <= 1e-2  # Allow some tolerance for sequential accumulation
    else:
        max_diff = 0.0
        mean_diff = 0.0
        all_match = True
    
    status = "x PASS" if all_match else "FAIL"
    
    return {
        'name': name,
        'ta_time': ta_time,
        'cuda_time': cuda_time,
        'speedup': speedup,
        'match': all_match,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'status': status
    }


def benchmark_dual_output(name, ta_func, cuda_func, data, warmup=3, iterations=10):
    """Benchmark dual-output HT indicator (PHASOR, SINE)"""
    size = len(data)
    output1 = np.zeros(size, dtype=np.float32)
    output2 = np.zeros(size, dtype=np.float32)
    data64 = data.astype(np.float64)  # TA-Lib requires float64
    
    in_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out1_ptr = output1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out2_ptr = output2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Warmup
    for _ in range(warmup):
        _ = ta_func(data64)
        cuda_func(in_ptr, out1_ptr, out2_ptr, size, None)
    
    # Benchmark TA-Lib
    ta_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        ta_result = ta_func(data64)  # Returns tuple (out1, out2)
        ta_times.append((time.perf_counter() - start) * 1000)
    ta_time = np.median(ta_times)
    
    # Benchmark tacuda
    cuda_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        cuda_func(in_ptr, out1_ptr, out2_ptr, size, None)
        cuda_times.append((time.perf_counter() - start) * 1000)
    cuda_time = np.median(cuda_times)
    
    cuda_result1 = output1.copy()
    cuda_result2 = output2.copy()
    speedup = ta_time / cuda_time if cuda_time > 0 else 0
    
    # Compare results
    max_diff = 0.0
    mean_diff = 0.0
    all_match = True
    
    for ta_r, cuda_r in [(ta_result[0], cuda_result1), (ta_result[1], cuda_result2)]:
        mask = ~np.isnan(ta_r) & ~np.isnan(cuda_r)
        if np.sum(mask) > 0:
            diff = np.abs(ta_r[mask] - cuda_r[mask])
            max_diff = max(max_diff, np.max(diff))
            mean_diff = max(mean_diff, np.mean(diff))
            if np.max(diff) > 1e-2:
                all_match = False
    
    status = "x PASS" if all_match else "FAIL"
    
    return {
        'name': name,
        'ta_time': ta_time,
        'cuda_time': cuda_time,
        'speedup': speedup,
        'match': all_match,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'status': status
    }


def benchmark_trendmode(data, warmup=3, iterations=10):
    """Benchmark HT_TRENDMODE (integer output)"""
    size = len(data)
    output = np.zeros(size, dtype=np.float32)
    data64 = data.astype(np.float64)  # TA-Lib requires float64
    
    in_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Warmup
    for _ in range(warmup):
        _ = talib.HT_TRENDMODE(data64)
        tacuda_ht_trendmode(in_ptr, out_ptr, size, None)
    
    # Benchmark TA-Lib
    ta_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        ta_result = talib.HT_TRENDMODE(data64)
        ta_times.append((time.perf_counter() - start) * 1000)
    ta_time = np.median(ta_times)
    
    # Benchmark tacuda
    cuda_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        tacuda_ht_trendmode(in_ptr, out_ptr, size, None)
        cuda_times.append((time.perf_counter() - start) * 1000)
    cuda_time = np.median(cuda_times)
    
    cuda_result = output.copy()
    speedup = ta_time / cuda_time if cuda_time > 0 else 0
    
    # Compare (integer output)
    ta_int = ta_result.astype(np.int32)
    cuda_int = cuda_result.astype(np.int32)
    
    mask = ~np.isnan(ta_result)
    if np.sum(mask) > 0:
        matches = np.sum(ta_int[mask] == cuda_int[mask])
        total = np.sum(mask)
        match_pct = matches / total * 100
        all_match = match_pct > 99.0
    else:
        match_pct = 100.0
        all_match = True
    
    status = "x PASS" if all_match else "FAIL"
    
    return {
        'name': 'HT_TRENDMODE',
        'ta_time': ta_time,
        'cuda_time': cuda_time,
        'speedup': speedup,
        'match': all_match,
        'match_pct': match_pct,
        'max_diff': 0,
        'mean_diff': 0,
        'status': status
    }


def main():
    print_header("Hilbert Transform Benchmark: tacuda vs TA-Lib")
    print()
    
    # Generate test data
    size = 500000
    print(f"Data size: {size:,} elements")
    print()
    
    np.random.seed(42)
    # Generate realistic price data with trend and cycles
    t = np.linspace(0, 100 * np.pi, size)
    trend = 100 + 0.001 * np.arange(size)
    cycle1 = 5 * np.sin(t / 10)
    cycle2 = 3 * np.sin(t / 20)
    cycle3 = 2 * np.sin(t / 35)
    noise = np.random.randn(size) * 0.5
    
    data = (trend + cycle1 + cycle2 + cycle3 + noise).astype(np.float32)
    
    print(f"Price range: {data.min():.2f} - {data.max():.2f}")
    print()
    
    results = []
    
    # 1. HT_DCPERIOD
    if tacuda_ht_dcperiod:
        print_header("1. HT_DCPERIOD (Dominant Cycle Period)")
        result = benchmark_single_output("HT_DCPERIOD", talib.HT_DCPERIOD, tacuda_ht_dcperiod, data)
        results.append(result)
        print_row("TA-Lib (ms)", "tacuda (ms)", "Speedup", "Match", "Max Diff", "Mean Diff")
        print("=" * 110)
        print_row(result['ta_time'], result['cuda_time'], f"{result['speedup']:.2f}x", 
                  result['status'], f"{result['max_diff']:.2e}", f"{result['mean_diff']:.2e}")
        print()
    
    # 2. HT_DCPHASE
    if tacuda_ht_dcphase:
        print_header("2. HT_DCPHASE (Dominant Cycle Phase)")
        result = benchmark_single_output("HT_DCPHASE", talib.HT_DCPHASE, tacuda_ht_dcphase, data)
        results.append(result)
        print_row("TA-Lib (ms)", "tacuda (ms)", "Speedup", "Match", "Max Diff", "Mean Diff")
        print("=" * 110)
        print_row(result['ta_time'], result['cuda_time'], f"{result['speedup']:.2f}x", 
                  result['status'], f"{result['max_diff']:.2e}", f"{result['mean_diff']:.2e}")
        print()
    
    # 3. HT_PHASOR
    if tacuda_ht_phasor:
        print_header("3. HT_PHASOR (Hilbert Transform Phasor)")
        result = benchmark_dual_output("HT_PHASOR", talib.HT_PHASOR, tacuda_ht_phasor, data)
        results.append(result)
        print_row("TA-Lib (ms)", "tacuda (ms)", "Speedup", "Match", "Max Diff", "Mean Diff")
        print("=" * 110)
        print_row(result['ta_time'], result['cuda_time'], f"{result['speedup']:.2f}x", 
                  result['status'], f"{result['max_diff']:.2e}", f"{result['mean_diff']:.2e}")
        print()
    
    # 4. HT_SINE
    if tacuda_ht_sine:
        print_header("4. HT_SINE (Hilbert Transform Sine)")
        result = benchmark_dual_output("HT_SINE", talib.HT_SINE, tacuda_ht_sine, data)
        results.append(result)
        print_row("TA-Lib (ms)", "tacuda (ms)", "Speedup", "Match", "Max Diff", "Mean Diff")
        print("=" * 110)
        print_row(result['ta_time'], result['cuda_time'], f"{result['speedup']:.2f}x", 
                  result['status'], f"{result['max_diff']:.2e}", f"{result['mean_diff']:.2e}")
        print()
    
    # 5. HT_TRENDMODE
    if tacuda_ht_trendmode:
        print_header("5. HT_TRENDMODE (Hilbert Transform Trend vs Cycle Mode)")
        result = benchmark_trendmode(data)
        results.append(result)
        print_row("TA-Lib (ms)", "tacuda (ms)", "Speedup", "Match", "Match %")
        print("=" * 110)
        print_row(result['ta_time'], result['cuda_time'], f"{result['speedup']:.2f}x", 
                  result['status'], f"{result.get('match_pct', 100):.1f}%")
        print()
    
    # 6. HT_TRENDLINE
    if tacuda_ht_trendline:
        print_header("6. HT_TRENDLINE (Hilbert Transform Instantaneous Trendline)")
        result = benchmark_single_output("HT_TRENDLINE", talib.HT_TRENDLINE, tacuda_ht_trendline, data)
        results.append(result)
        print_row("TA-Lib (ms)", "tacuda (ms)", "Speedup", "Match", "Max Diff", "Mean Diff")
        print("=" * 110)
        print_row(result['ta_time'], result['cuda_time'], f"{result['speedup']:.2f}x", 
                  result['status'], f"{result['max_diff']:.2e}", f"{result['mean_diff']:.2e}")
        print()
    
    # Summary
    print_header("SUMMARY")
    print()
    print(f"{'Indicator':<20} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Status':<12}")
    print("-" * 74)
    
    for r in results:
        print(f"{r['name']:<20} {r['ta_time']:<15.3f} {r['cuda_time']:<15.3f} {r['speedup']:<12.2f}x {r['status']:<12}")
    
    print()
    if results:
        avg_speedup = np.mean([r['speedup'] for r in results])
        passed = sum(1 for r in results if r['match'])
        total = len(results)
        
        print(f"Average Speedup: {avg_speedup:.2f}x")
        print(f"Tests Passed: {passed}/{total}")
    print()
    
    print("Notes:")
    print("  - Hilbert Transform is inherently sequential (adaptive period estimation)")
    print("  - GPU optimization removes CPU<->GPU data transfer overhead")
    print("  - Single-threaded GPU kernel keeps data on device")
    print()


if __name__ == "__main__":
    main()
