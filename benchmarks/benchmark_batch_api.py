#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""
Benchmark + stress test for the explicit batch API (ct_*_batch, v0.2.0).

Compares three execution strategies over a portfolio of series:
  1. loop      — per-series single calls (baseline, one upload/kernel/download each)
  2. batch(c)  — ct_*_batch with several maxConcurrency values (stream overlap)

Also runs a stress phase: repeated large-portfolio dispatches checking result
stability and buffer-pool growth.

Usage:
    python benchmarks/benchmark_batch_api.py [--quick]
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bindings", "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tacuda  # noqa: E402
from datasets import portfolio  # noqa: E402  (reproducible benchmark data)


def make_portfolio(num_series: int, series_len: int, seed: int = 42) -> np.ndarray:
    return portfolio("trending", num_series, series_len, seed=seed)


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters * 1000.0


def rows_match(a: np.ndarray, b: np.ndarray, tol: float = 1e-4) -> bool:
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return False
    mask = ~nan_a
    return bool(np.allclose(a[mask], b[mask], rtol=tol, atol=tol))


def run_indicator(name, matrix, loop_fn, batch_fn, concurrency_levels, warmup, iters):
    num_series, series_len = matrix.shape
    print(f"\n--- {name}: {num_series} series x {series_len} bars ---")

    reference = loop_fn(matrix)
    loop_ms = bench(lambda: loop_fn(matrix), warmup, iters)
    print(f"  {'loop (per-series)':<28} {loop_ms:10.2f} ms   1.00x")

    results = {}
    for conc in concurrency_levels:
        out = batch_fn(matrix, conc)
        ok = rows_match(out, reference)
        ms = bench(lambda c=conc: batch_fn(matrix, c), warmup, iters)
        label = "auto" if conc == 0 else str(conc)
        speedup = loop_ms / ms if ms > 0 else 0.0
        flag = "" if ok else "   MISMATCH!"
        print(f"  {'batch(maxConcurrency=' + label + ')':<28} {ms:10.2f} ms {speedup:6.2f}x{flag}")
        results[conc] = (ms, ok)
    return loop_ms, results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smaller shapes / fewer iterations")
    args = parser.parse_args()

    print("=" * 78)
    print(f"TACUDA batch API benchmark  (native {tacuda.version()})")
    print("=" * 78)

    if args.quick:
        shapes = [(16, 4096), (64, 1024)]
        warmup, iters = 2, 5
        stress_rounds = 3
    else:
        shapes = [(10, 1000), (100, 1000), (500, 1000), (100, 16384), (16, 262144)]
        warmup, iters = 3, 20
        stress_rounds = 25

    concurrency_levels = [0, 1, 2, 4, 8]
    failures = 0

    for num_series, series_len in shapes:
        matrix = make_portfolio(num_series, series_len)

        def rsi_loop(m):
            return np.stack([tacuda.rsi(m[i], 14) for i in range(m.shape[0])])

        def rsi_batch(m, conc):
            return tacuda.rsi_batch(m, period=14, max_concurrency=conc)

        _, res = run_indicator(f"RSI(14)", matrix, rsi_loop, rsi_batch,
                               concurrency_levels, warmup, iters)
        failures += sum(0 if ok else 1 for _, ok in res.values())

        def sma_loop(m):
            return np.stack([tacuda.sma(m[i], 20) for i in range(m.shape[0])])

        def sma_batch(m, conc):
            return tacuda.sma_batch(m, period=20, max_concurrency=conc)

        _, res = run_indicator(f"SMA(20)", matrix, sma_loop, sma_batch,
                               concurrency_levels, warmup, iters)
        failures += sum(0 if ok else 1 for _, ok in res.values())

        def macd_loop(m):
            return np.stack([tacuda.macd(m[i], 12, 26, 9, 1)[0] for i in range(m.shape[0])])

        def macd_batch(m, conc):
            return tacuda.macd_batch(m, max_concurrency=conc)[0]

        _, res = run_indicator(f"MACD(12,26,9)", matrix, macd_loop, macd_batch,
                               concurrency_levels, warmup, iters)
        failures += sum(0 if ok else 1 for _, ok in res.values())

    # ------------------------------------------------------------------
    # Stress phase: repeated large dispatches, result & memory stability
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("STRESS PHASE")
    print("=" * 78)
    num_series, series_len = (64, 8192) if args.quick else (256, 16384)
    matrix = make_portfolio(num_series, series_len, seed=7)
    baseline = tacuda.rsi_batch(matrix, period=14, max_concurrency=4)
    alloc_before = tacuda.device_buffer_pool_allocation_count()

    stress_ok = True
    start = time.perf_counter()
    for round_idx in range(stress_rounds):
        out = tacuda.rsi_batch(matrix, period=14, max_concurrency=4)
        if not rows_match(out, baseline):
            print(f"  round {round_idx}: RESULT DRIFT detected")
            stress_ok = False
    elapsed = time.perf_counter() - start
    alloc_after = tacuda.device_buffer_pool_allocation_count()

    bars = num_series * series_len * stress_rounds
    print(f"  {stress_rounds} rounds x {num_series}x{series_len} bars: "
          f"{elapsed:.2f} s total, {bars / elapsed / 1e6:.1f} Mbar/s")
    print(f"  buffer pool allocations: {alloc_before} -> {alloc_after} "
          f"(+{alloc_after - alloc_before})")
    print(f"  results stable: {'YES' if stress_ok else 'NO'}")
    if not stress_ok:
        failures += 1

    # Invalid inputs must raise, not crash.
    try:
        tacuda.rsi_batch(np.zeros((0, 8), dtype=np.float32), period=14)
        print("  invalid input (0 series): NOT rejected")
        failures += 1
    except (ValueError, RuntimeError):
        print("  invalid input (0 series): rejected as expected")

    print("\n" + "=" * 78)
    print(f"RESULT: {'PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    print("=" * 78)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
