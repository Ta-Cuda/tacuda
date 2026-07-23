#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark T3: tacuda vs TA-Lib
"""
import time
import numpy as np
import talib
import ctypes
import os
import sys

# Load tacuda
this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, '..', 'build', 'Release', 'tacuda.dll')
if not os.path.exists(lib_path):
    lib_path = os.path.join(this_dir, '..', 'build', 'libtacuda.so')

cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
if os.path.exists(cuda_path):
    os.environ['PATH'] = cuda_path + os.pathsep + os.environ.get('PATH', '')
    try:
        os.add_dll_directory(cuda_path)
    except (AttributeError, FileNotFoundError):
        pass

try:
    tacuda = ctypes.CDLL(lib_path)
except Exception as e:
    print(f"Failed to load tacuda library: {e}")
    print(f"Library path: {lib_path}")
    print("Make sure CUDA runtime DLLs are in PATH")
    sys.exit(1)

try:
    ct_t3 = tacuda.ct_t3
except AttributeError as exc:
    raise SystemExit("ct_t3 symbol not found in tacuda library") from exc

ct_t3.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # size
    ctypes.c_int,                    # period
    ctypes.c_float,                  # vFactor
    ctypes.c_void_p                  # stream
]
ct_t3.restype = ctypes.c_int


def benchmark_talib(data, period, vfactor=0.7, iterations=50):
    arr = np.asarray(data, dtype=np.float64)
    for _ in range(5):
        _ = talib.T3(arr, timeperiod=period, vfactor=vfactor)
    t0 = time.perf_counter()
    for _ in range(iterations):
        out = talib.T3(arr, timeperiod=period, vfactor=vfactor)
    t1 = time.perf_counter()
    return out.astype(np.float32), (t1 - t0) / iterations * 1000


def benchmark_tacuda(data, period, vfactor=0.7, iterations=50):
    size = len(data)
    inp = np.asarray(data, dtype=np.float32)
    out = np.full(size, np.nan, dtype=np.float32)
    in_ptr = inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    for _ in range(5):
        ct_t3(in_ptr, out_ptr, size, period, ctypes.c_float(vfactor), None)
    t0 = time.perf_counter()
    for _ in range(iterations):
        ct_t3(in_ptr, out_ptr, size, period, ctypes.c_float(vfactor), None)
    t1 = time.perf_counter()
    return out, (t1 - t0) / iterations * 1000


def compare_results(ta_out, cu_out, period, tol=1e-5):
    start = 6 * (period - 1)
    ta_v = ta_out[start:]
    cu_v = cu_out[start:]
    mask = ~(np.isnan(ta_v) | np.isnan(cu_v))
    ta_v = ta_v[mask]
    cu_v = cu_v[mask]
    if ta_v.size == 0:
        return False, 0.0, 0.0
    diff = np.abs(ta_v - cu_v)
    return bool(np.allclose(ta_v, cu_v, rtol=tol, atol=tol)), float(np.max(diff)), float(np.mean(diff))


def print_sep():
    print("=" * 100)


def main():
    print_sep()
    print("T3 Benchmark: tacuda vs TA-Lib".center(100))
    print_sep()
    print()

    data_size = 500000
    print(f"Data size: {data_size:,} elements\n")
    np.random.seed(42)
    x = np.linspace(0, 100, data_size)
    data = (np.sin(x) + np.random.randn(data_size) * 0.05).astype(np.float32)

    period = 5
    vfactors = [0.0, 0.5, 0.7, 1.0]

    print_sep()
    print(f"{'vFactor':<10} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15} {'Mean Diff':<15}")
    print_sep()

    results = []
    for vf in vfactors:
        ta_out, ta_ms = benchmark_talib(data, period, vfactor=vf)
        cu_out, cu_ms = benchmark_tacuda(data, period, vfactor=vf)
        match, dmax, dmean = compare_results(ta_out, cu_out, period)
        speedup = ta_ms / cu_ms if cu_ms > 0 else 0.0
        print(f"{vf:<10.2f} {ta_ms:<15.3f} {cu_ms:<15.3f} {speedup:<12.2f}x {'PASS' if match else 'FAIL':<8} {dmax:<15.6e} {dmean:<15.6e}")
        results.append((vf, ta_ms, cu_ms, speedup, match, dmax, dmean))

    print_sep()
    print()
    sp = np.array([r[3] for r in results], dtype=np.float64)
    print("Summary:")
    print(f"  Average speedup: {sp.mean():.2f}x")
    print(f"  Max speedup:     {sp.max():.2f}x")
    print(f"  Min speedup:     {sp.min():.2f}x")
    print(f"  All tests match: {'YES' if all(r[4] for r in results) else 'NO'}")
    print()

if __name__ == '__main__':
    main()
