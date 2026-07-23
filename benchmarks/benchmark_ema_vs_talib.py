#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark EMA: tacuda vs TA-Lib
Сравниваем производительность и точность для разных окон.
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
    tacuda_ema = tacuda.ct_ema
except AttributeError as exc:
    raise SystemExit("ct_ema symbol not found in tacuda library") from exc

tacuda_ema.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # size
    ctypes.c_int,                    # period
    ctypes.c_void_p                  # stream
]
tacuda_ema.restype = ctypes.c_int

def benchmark_talib(data, period, iterations=50):
    arr = np.asarray(data, dtype=np.float64)
    for _ in range(5):
        _ = talib.EMA(arr, timeperiod=period)
    t0 = time.perf_counter()
    for _ in range(iterations):
        out = talib.EMA(arr, timeperiod=period)
    t1 = time.perf_counter()
    return out.astype(np.float32), (t1 - t0) / iterations * 1000

def benchmark_tacuda(data, period, iterations=50):
    size = len(data)
    inp = np.asarray(data, dtype=np.float32)
    out = np.full(size, np.nan, dtype=np.float32)
    in_ptr = inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # Конфигурация EMA (по умолчанию: последовательный корретный путь)
    os.environ.setdefault('TACUDA_EMA_ALGO', '0')
    for _ in range(5):
        tacuda_ema(in_ptr, out_ptr, size, period, None)
    t0 = time.perf_counter()
    for _ in range(iterations):
        tacuda_ema(in_ptr, out_ptr, size, period, None)
    t1 = time.perf_counter()
    return out, (t1 - t0) / iterations * 1000

def compare_results(ta_out, cu_out, period, tol=1e-5):
    start = period - 1
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
    print("EMA Benchmark: tacuda vs TA-Lib".center(100))
    print_sep()
    print()

    # Конфигурация EMA
    cfg = {
        'TACUDA_EMA_BLOCK_SIZE': os.getenv('TACUDA_EMA_BLOCK_SIZE', 'default (256)'),
        'TACUDA_EMA_SEQ_T': os.getenv('TACUDA_EMA_SEQ_T', 'default (10000)'),
        'TACUDA_EMA_WARP_T': os.getenv('TACUDA_EMA_WARP_T', 'default (100000)'),
        'TACUDA_EMA_ALGO': os.getenv('TACUDA_EMA_ALGO', 'Auto (0)')
    }
    print('Configuration:')
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print()

    data_size = 500000
    print(f"Data size: {data_size:,} elements\n")
    np.random.seed(42)
    x = np.linspace(0, 100, data_size)
    data = (np.sin(x) + np.random.randn(data_size) * 0.05).astype(np.float32)

    windows = [5, 10, 12, 20, 26, 50, 100, 200, 500, 1000]

    print_sep()
    print(f"{'Window':<10} {'TA-Lib (ms)':<15} {'tacuda (ms)':<15} {'Speedup':<12} {'Match':<8} {'Max Diff':<15} {'Mean Diff':<15}")
    print_sep()

    results = []
    for w in windows:
        if w >= data_size:
            continue
        ta_out, ta_ms = benchmark_talib(data, w)
        cu_out, cu_ms = benchmark_tacuda(data, w)
        match, dmax, dmean = compare_results(ta_out, cu_out, w)
        speedup = ta_ms / cu_ms if cu_ms > 0 else 0.0
        print(f"{w:<10} {ta_ms:<15.3f} {cu_ms:<15.3f} {speedup:<12.2f}x {'PASS' if match else 'FAIL':<8} {dmax:<15.6e} {dmean:<15.6e}")
        results.append((w, ta_ms, cu_ms, speedup, match, dmax, dmean))

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


