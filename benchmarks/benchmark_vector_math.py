#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Large-array benchmark for tacuda vector math vs TA-Lib and NumPy.

Element-wise math has no lookback, so it is the best-case demonstration of
GPU throughput on large inputs. Three measurement modes per function:

* ``tacuda host``   - includes H2D/D2H PCIe transfers (worst case for GPU);
* ``tacuda device`` - data already resident on the GPU (pipeline case);
* ``talib`` / ``numpy`` - CPU references (double / float32).

Usage:
    TACUDA_LIBRARY=/path/libtacuda.so python benchmarks/benchmark_vector_math.py --size 10000000
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bindings" / "python"))

import tacuda  # noqa: E402

try:
    import talib
except ImportError:  # pragma: no cover
    talib = None


def bench(fn, repeats: int = 5) -> float:
    fn()  # warmup
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    n = args.size
    rng = np.random.default_rng(42)
    x32 = (rng.random(n, dtype=np.float32) * 2.0 - 1.0)  # [-1, 1]
    y32 = (rng.random(n, dtype=np.float32) + 0.5)        # [0.5, 1.5]
    x64 = x32.astype(np.float64)
    y64 = y32.astype(np.float64)
    pos32 = np.abs(x32) + 0.01
    pos64 = pos32.astype(np.float64)

    unary_cases = [
        ("SIN", tacuda.sin, "sin_d", np.sin, "SIN", x32, x64),
        ("COS", tacuda.cos, "cos_d", np.cos, "COS", x32, x64),
        ("TAN", tacuda.tan, "tan_d", np.tan, "TAN", x32, x64),
        ("ASIN", tacuda.asin, "asin_d", np.arcsin, "ASIN", x32, x64),
        ("ACOS", tacuda.acos, "acos_d", np.arccos, "ACOS", x32, x64),
        ("ATAN", tacuda.atan, "atan_d", np.arctan, "ATAN", x32, x64),
        ("SINH", tacuda.sinh, "sinh_d", np.sinh, "SINH", x32, x64),
        ("COSH", tacuda.cosh, "cosh_d", np.cosh, "COSH", x32, x64),
        ("TANH", tacuda.tanh, "tanh_d", np.tanh, "TANH", x32, x64),
        ("EXP", tacuda.exp, "exp_d", np.exp, "EXP", x32, x64),
        ("LN", tacuda.ln, "ln_d", np.log, "LN", pos32, pos64),
        ("LOG10", tacuda.log10, "log10_d", np.log10, "LOG10", pos32, pos64),
        ("SQRT", tacuda.sqrt, "sqrt_d", np.sqrt, "SQRT", pos32, pos64),
        ("CEIL", tacuda.ceil, "ceil_d", np.ceil, "CEIL", x32, x64),
        ("FLOOR", tacuda.floor, "floor_d", np.floor, "FLOOR", x32, x64),
    ]
    binary_cases = [
        ("ADD", tacuda.add, "add_d", np.add, "ADD"),
        ("SUB", tacuda.sub, "sub_d", np.subtract, "SUB"),
        ("MULT", tacuda.mult, "mult_d", np.multiply, "MULT"),
        ("DIV", tacuda.div, "div_d", np.divide, "DIV"),
    ]

    # Device-resident buffers for the pipeline scenario.
    d_x = tacuda.device_alloc(n)
    d_y = tacuda.device_alloc(n)
    d_out = tacuda.device_alloc(n)
    if not (d_x and d_y and d_out):
        raise RuntimeError("device_alloc failed")
    tacuda.device_upload(x32, d_x, n)
    tacuda.device_upload(y32, d_y, n)
    tacuda.device_sync()

    lib = tacuda._lib

    def device_call(c_name: str, nargs: int):
        cfunc = getattr(lib, c_name)
        if nargs == 1:
            def run():
                cfunc(ctypes.c_void_p(d_x), ctypes.c_void_p(d_out),
                      ctypes.c_int(n), ctypes.c_void_p(0))
                tacuda.device_sync()
        else:
            def run():
                cfunc(ctypes.c_void_p(d_x), ctypes.c_void_p(d_y),
                      ctypes.c_void_p(d_out), ctypes.c_int(n), ctypes.c_void_p(0))
                tacuda.device_sync()
        return run

    print(f"# Vector math benchmark: {n:,} elements, best of {args.repeats}")
    header = f"{'func':8} {'tacuda host':>12} {'tacuda dev':>12} {'numpy f32':>12} {'talib f64':>12} {'dev speedup':>12}"
    print(header)
    print("-" * len(header))

    rows = []
    for name, host_fn, dev_name, np_fn, talib_name, a32, a64 in unary_cases:
        t_host = bench(lambda: host_fn(a32), args.repeats)
        t_dev = bench(device_call(f"ct_{dev_name}", 1), args.repeats)
        t_np = bench(lambda: np_fn(a32), args.repeats)
        t_talib = bench(lambda: getattr(talib, talib_name)(a64), args.repeats) if talib else float("nan")
        base = min(t_np, t_talib) if talib else t_np
        rows.append((name, t_host, t_dev, t_np, t_talib, base / t_dev))

    for name, host_fn, dev_name, np_fn, talib_name in binary_cases:
        t_host = bench(lambda: host_fn(x32, y32), args.repeats)
        t_dev = bench(device_call(f"ct_{dev_name}", 2), args.repeats)
        t_np = bench(lambda: np_fn(x32, y32), args.repeats)
        t_talib = bench(lambda: getattr(talib, talib_name)(x64, y64), args.repeats) if talib else float("nan")
        base = min(t_np, t_talib) if talib else t_np
        rows.append((name, t_host, t_dev, t_np, t_talib, base / t_dev))

    for name, t_host, t_dev, t_np, t_talib, speedup in rows:
        print(f"{name:8} {t_host:10.2f}ms {t_dev:10.3f}ms {t_np:10.2f}ms {t_talib:10.2f}ms {speedup:11.1f}x")

    tacuda.device_free(d_x)
    tacuda.device_free(d_y)
    tacuda.device_free(d_out)


if __name__ == "__main__":
    main()
