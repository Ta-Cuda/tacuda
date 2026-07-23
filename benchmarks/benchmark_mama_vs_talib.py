#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark MAMA: tacuda vs TA-Lib
E2E timing (host<->device + kernel) for tacuda, CPU-only for TA-Lib.
"""

import time
import numpy as np
import talib
import ctypes
import ctypes.util
import os
import sys
import platform

# ------------------------------------------------------------
# Locate and load tacuda shared library cross-platform
# ------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))

def find_tacuda_lib():
    sysname = platform.system()
    if sysname == "Windows":
        candidates = [
            os.path.join(ROOT, "..", "build", "Release", "tacuda.dll"),
            os.path.join(ROOT, "..", "build", "tacuda.dll"),
        ]
    elif sysname == "Darwin":
        candidates = [os.path.join(ROOT, "..", "build", "libtacuda.dylib")]
    else:  # Linux/Unix
        candidates = [os.path.join(ROOT, "..", "build", "libtacuda.so")]
    for p in candidates:
        if os.path.exists(p):
            return os.path.normpath(p)
    return None

tacuda_lib_path = find_tacuda_lib()
if not tacuda_lib_path:
    print("Failed to locate tacuda shared library")
    sys.exit(1)

# On Windows ensure CUDA bin/build dir are in DLL search path (Py3.8+)
dll_ctx_handles = []
if os.name == "nt":
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
    if os.path.isdir(cuda_path) and hasattr(os, "add_dll_directory"):
        try: dll_ctx_handles.append(os.add_dll_directory(cuda_path))
        except OSError: pass
    build_dir = os.path.dirname(tacuda_lib_path)
    if hasattr(os, "add_dll_directory"):
        try: dll_ctx_handles.append(os.add_dll_directory(build_dir))
        except OSError: pass

# Load the library
try:
    tacuda = ctypes.CDLL(tacuda_lib_path)
except OSError as e:
    if os.name == "nt":
        try:
            tacuda = ctypes.WinDLL(tacuda_lib_path)
        except OSError:
            print(f"Failed to load tacuda library: {e}\nPath: {tacuda_lib_path}")
            sys.exit(1)
    else:
        print(f"Failed to load tacuda library: {e}\nPath: {tacuda_lib_path}")
        sys.exit(1)

# Resolve symbol
try:
    tacuda_mama = tacuda.ct_mama
except AttributeError as exc:
    raise SystemExit("ct_mama symbol not found in tacuda library") from exc

# ct_mama(const float* host_input, float* host_mama, float* host_fama,
#         int size, float fastLimit, float slowLimit, cudaStream_t stream)
tacuda_mama.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # input
    ctypes.POINTER(ctypes.c_float),  # output mama
    ctypes.POINTER(ctypes.c_float),  # output fama
    ctypes.c_int,                    # size
    ctypes.c_float,                  # fastLimit
    ctypes.c_float,                  # slowLimit
    ctypes.c_void_p                  # stream (nullptr ok)
]
tacuda_mama.restype = ctypes.c_int  # 0 == OK

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def set_batch_env(batch_count: int, series_len: int):
    """Fix batch layout for the backend via env (read by the shared lib)."""
    os.environ["TACUDA_MAMA_BATCH_COUNT"] = str(int(batch_count))
    os.environ["TACUDA_MAMA_SERIES_LEN"]  = str(int(series_len))

def benchmark_talib(data, fast=0.5, slow=0.05, iterations=50, batch_count=1):
    """
    TA-Lib reference (CPU).
    data: series-major flattened if batch_count > 1.
    """
    if batch_count == 1:
        arr64 = np.asarray(data, dtype=np.float64)  # one-time convert
        for _ in range(5):
            _m, _f = talib.MAMA(arr64, fastlimit=fast, slowlimit=slow)
        t0 = time.perf_counter()
        for _ in range(iterations):
            m, f = talib.MAMA(arr64, fastlimit=fast, slowlimit=slow)
        t1 = time.perf_counter()
        out = np.stack([m.astype(np.float32, copy=False),
                        f.astype(np.float32, copy=False)], axis=0)
        return out, (t1 - t0) / iterations * 1000.0
    else:
        data_2d = np.asarray(data, dtype=np.float32).reshape(batch_count, -1)
        series_len = data_2d.shape[1]
        # ONE-TIME convert to float64 for all series
        data_2d64 = data_2d.astype(np.float64, copy=False)

        out_m_all = np.full((batch_count, series_len), np.nan, dtype=np.float32)
        out_f_all = np.full((batch_count, series_len), np.nan, dtype=np.float32)

        # warmup
        for _ in range(5):
            for b in range(batch_count):
                _m, _f = talib.MAMA(data_2d64[b], fastlimit=fast, slowlimit=slow)

        t0 = time.perf_counter()
        for _ in range(iterations):
            for b in range(batch_count):
                m, f = talib.MAMA(data_2d64[b], fastlimit=fast, slowlimit=slow)
                out_m_all[b] = m.astype(np.float32, copy=False)
                out_f_all[b] = f.astype(np.float32, copy=False)
        t1 = time.perf_counter()

        out = np.stack([out_m_all.ravel(), out_f_all.ravel()], axis=0)
        return out, (t1 - t0) / iterations * 1000.0

def benchmark_tacuda(data, fast=0.5, slow=0.05, iterations=50, batch_count=1):
    """
    tacuda E2E timing: includes H2D/D2H and any implicit syncs.
    data: series-major flattened of length = batch_count * series_len.
    """
    inp = np.ascontiguousarray(data, dtype=np.float32)
    size = inp.size
    out_m = np.full(size, np.nan, dtype=np.float32)
    out_f = np.full(size, np.nan, dtype=np.float32)

    in_ptr    = inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_m_ptr = out_m.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_f_ptr = out_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    null_stream = ctypes.c_void_p(0)

    # warmup
    for _ in range(5):
        rc = tacuda_mama(in_ptr, out_m_ptr, out_f_ptr, size,
                         ctypes.c_float(fast), ctypes.c_float(slow), null_stream)
        if rc != 0:
            raise RuntimeError(f"ct_mama failed in warmup with code {rc}")

    t0 = time.perf_counter()
    for _ in range(iterations):
        rc = tacuda_mama(in_ptr, out_m_ptr, out_f_ptr, size,
                         ctypes.c_float(fast), ctypes.c_float(slow), null_stream)
        if rc != 0:
            raise RuntimeError(f"ct_mama failed with code {rc}")
    t1 = time.perf_counter()

    out = np.stack([out_m, out_f], axis=0)
    return out, (t1 - t0) / iterations * 1000.0

def compare_results(ta_out, cu_out, rtol=1e-3, atol=1e-5):
    """
    Compare only finite values; treat NaNs (lookback) as equal.
    """
    m_mask = np.isfinite(ta_out[0]) & np.isfinite(cu_out[0])
    f_mask = np.isfinite(ta_out[1]) & np.isfinite(cu_out[1])

    m_ok = np.allclose(ta_out[0], cu_out[0], rtol=rtol, atol=atol, equal_nan=True)
    f_ok = np.allclose(ta_out[1], cu_out[1], rtol=rtol, atol=atol, equal_nan=True)

    dmax_m = float(np.max(np.abs(ta_out[0][m_mask] - cu_out[0][m_mask]))) if m_mask.any() else 0.0
    dmax_f = float(np.max(np.abs(ta_out[1][f_mask] - cu_out[1][f_mask]))) if f_mask.any() else 0.0
    return (m_ok and f_ok), max(dmax_m, dmax_f)

def print_sep():
    print("=" * 100)

def main():
    print_sep()
    print("MAMA Benchmark: tacuda (E2E) vs TA-Lib".center(100))
    print_sep()
    print()

    # Batch config from env or defaults
    batch_count = int(os.environ.get('TACUDA_MAMA_BATCH_COUNT', '1'))
    series_len  = int(os.environ.get('TACUDA_MAMA_SERIES_LEN', '500000'))

    if batch_count > 1 and series_len > 0:
        data_size = batch_count * series_len
        print(f"Batch mode: {batch_count} series x {series_len:,} elements = {data_size:,} total\n")
        set_batch_env(batch_count, series_len)
    else:
        batch_count = 1
        data_size = series_len if series_len > 0 else 500000
        print(f"Single series: {data_size:,} elements\n")
        set_batch_env(1, data_size)

    # Synthetic data (series-major)
    np.random.seed(42)
    if batch_count > 1:
        xs = np.linspace(0, 100, series_len, dtype=np.float32)
        data = np.concatenate([
            (np.sin(xs + b * 0.1) + np.random.randn(series_len).astype(np.float32) * 0.05)
            for b in range(batch_count)
        ]).astype(np.float32, copy=False)
    else:
        xs = np.linspace(0, 100, data_size, dtype=np.float32)
        data = (np.sin(xs) + np.random.randn(data_size).astype(np.float32) * 0.05).astype(np.float32, copy=False)

    fast, slow = 0.5, 0.05

    print_sep()
    print(f"{'Config':<10} {'Case':<12} {'TA-Lib (ms)':<15} {'tacuda E2E (ms)':<18} {'Speedup':<12} {'Match':<8} {'Max Diff':<15}")
    print_sep()

    # Two modes kept for compatibility (if the lib reads them)
    configs = [('seq', '1'), ('opt', '2')]
    for label, algo in configs:
        os.environ['TACUDA_MAMA_ALGO'] = algo
        ta_out, ta_ms = benchmark_talib(data, fast=fast, slow=slow, batch_count=batch_count)
        cu_out, cu_ms = benchmark_tacuda(data, fast=fast, slow=slow, batch_count=batch_count)
        ok, dmax = compare_results(ta_out, cu_out, rtol=1e-3, atol=1e-5)
        speedup = (ta_ms / cu_ms) if cu_ms > 0 else 0.0
        print(f"{label:<10} {'default':<12} {ta_ms:<15.3f} {cu_ms:<18.3f} {speedup:<12.2f}x {'PASS' if ok else 'FAIL':<8} {dmax:<15.6e}")

    print_sep()

if __name__ == '__main__':
    main()
