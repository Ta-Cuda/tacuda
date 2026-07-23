#!/usr/bin/env python
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
import os
import sys
import time
from typing import Any, List, Tuple, Callable

import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
tacuda_bindings_path = os.path.join(project_root, "bindings", "python")
sys.path.insert(0, project_root)
sys.path.insert(0, tacuda_bindings_path)

if sys.platform.startswith("win"):
    cuda_paths = [
        r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\bin",
        r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1\\bin",
        r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0\\bin",
        r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin",
    ]
    for p in cuda_paths:
        if os.path.exists(p):
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(p)
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            break

dll_path = os.path.join(project_root, "build", "Release", "tacuda.dll")
if not os.path.exists(dll_path):
    dll_path = os.path.join(project_root, "build", "Debug", "tacuda.dll")
if os.path.exists(dll_path):
    os.environ["TACUDA_LIBRARY"] = dll_path
else:
    print(f"ERROR: Cannot find TACUDA library: {dll_path}")
    sys.exit(1)

import tacuda
import talib
from benchmarks.talib_compare_helpers import build_specs, compare_arrays, generate_inputs


def bench(f: Callable, *args, repeats: int = 3) -> float:
    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = f(*args)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> int:
    dtypes = [d.strip() for d in os.getenv("TACUDA_BENCH_DTYPES", "float32").split(",") if d.strip()]
    exit_code = 0
    for dtype in dtypes:
        print(f"\n=== DTYPE PROFILE: {dtype} ===")
        data = generate_inputs(100_000, seed=42)
        specs = build_specs(tacuda, talib, dtype=dtype)
        report: List[Tuple[str, bool, float, float]] = []

        for spec in specs:
            try:
                tt = bench(spec.tacuda_fn, data)
                ta = bench(spec.talib_fn, data)
                t_out = spec.tacuda_fn(data)
                a_out = spec.talib_fn(data)
                if isinstance(t_out, tuple):
                    ok = True
                    md = 0.0
                    for ti, ai in zip(t_out, a_out):
                        ok_i, md_i, _, _ = compare_arrays(np.asarray(ti), np.asarray(ai), spec.tolerance)
                        ok = ok and ok_i
                        md = max(md, md_i)
                else:
                    ok, md, _, _ = compare_arrays(np.asarray(t_out), np.asarray(a_out), spec.tolerance)
                speedup = ta / max(tt, 1e-9)
                report.append((spec.name, ok, md, speedup))
            except Exception as e:
                report.append((spec.name + f" [ERROR: {e}]", False, float("inf"), 0.0))

        total = len(report)
        passed = sum(1 for _, ok, _, _ in report if ok)
        print(f"\nTotal: {total}, Passed: {passed}, Failed: {total-passed}")
        worst = sorted(report, key=lambda x: (not x[1], -x[2]))[:10]
        print("\nWorst (by correctness / max_diff):")
        for name, ok, md, sp in worst:
            print(f"  {name:18} ok={ok} max_diff={md:.3e} speedup={sp:.2f}x")
        failures = [(n, md, sp) for (n, ok, md, sp) in report if not ok]
        if failures:
            print("\nFailures:")
            for n, md, sp in failures:
                print(f"  {n:18} max_diff={md:.3e} speedup={sp:.2f}x")
            exit_code = 1
        top = sorted(report, key=lambda x: x[3], reverse=True)[:10]
        print("\nTop speedups:")
        for name, ok, md, sp in top:
            print(f"  {name:18} ok={ok} max_diff={md:.3e} speedup={sp:.2f}x")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())


