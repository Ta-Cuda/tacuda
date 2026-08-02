#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Host-API speedup vs TA-Lib, per indicator.

This is the measurement behind docs/SLOW_INDICATORS_ANALYSIS.md. It answers
one question honestly: for a single series through the plain host API (the
simplest thing a user can call), is TaCuda faster than TA-Lib, and by how
much?

The host API is the least favourable case for a GPU library — one PCIe
round trip per call, no reuse of resident data — so these numbers are a
floor, not a headline. Device and batch pipelines amortize the transfer and
are measured separately (benchmarks/benchmark_pipeline_vs_talib.py,
docs/EXPERIMENTS.md).

Usage (GPU box with tacuda importable and TA-Lib installed):
    python benchmarks/bench_host_vs_talib.py [--size 500000] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

try:
    import tacuda
except ImportError as exc:  # pragma: no cover
    print(f"tacuda import failed: {exc}", file=sys.stderr)
    sys.exit(2)

try:
    import talib
except ImportError as exc:  # pragma: no cover
    print(f"TA-Lib import failed: {exc}", file=sys.stderr)
    sys.exit(2)


def make_data(n: int) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(20260802)
    close = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    close = np.maximum(close, 1.0)
    high = close + rng.rand(n) * 2.0
    low = close - rng.rand(n) * 2.0
    volume = rng.rand(n) * 1_000_000 + 1_000
    return {
        "close32": close.astype(np.float32),
        "high32": high.astype(np.float32),
        "low32": low.astype(np.float32),
        "volume32": volume.astype(np.float32),
        "close64": close.astype(np.float64),
        "high64": high.astype(np.float64),
        "low64": low.astype(np.float64),
        "volume64": volume.astype(np.float64),
    }


def timeit(fn: Callable[[], object], repeats: int) -> float:
    """Median wall time in milliseconds, after a warm-up call.

    The first call pays CUDA context creation and buffer-pool population; a
    live process pays that once, so charging it to every measurement would
    overstate the cost of the API rather than measure it.
    """
    fn()
    samples: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def build_cases(d: Dict[str, np.ndarray]) -> List[Tuple[str, str, Callable, Callable]]:
    """(name, group, tacuda_call, talib_call) — the indicators named in
    SLOW_INDICATORS_ANALYSIS.md plus the sequential-by-design group."""
    c32, h32, l32, v32 = d["close32"], d["high32"], d["low32"], d["volume32"]
    c64, h64, l64, v64 = d["close64"], d["high64"], d["low64"], d["volume64"]

    return [
        # Previously "critical" (<0.1x) per the old analysis
        ("VAR(14)", "was-critical", lambda: tacuda.var(c32, 14), lambda: talib.VAR(c64, 14)),
        ("KAMA(14)", "was-critical", lambda: tacuda.kama(c32, 14, 2, 30), lambda: talib.KAMA(c64, 14)),
        ("CMO(14)", "was-critical", lambda: tacuda.cmo(c32, 14), lambda: talib.CMO(c64, 14)),
        ("DX(14)", "was-critical", lambda: tacuda.dx(h32, l32, c32, 14), lambda: talib.DX(h64, l64, c64, 14)),
        ("MINUS_DI(14)", "was-critical", lambda: tacuda.minus_di(h32, l32, c32, 14), lambda: talib.MINUS_DI(h64, l64, c64, 14)),
        ("PLUS_DI(14)", "was-critical", lambda: tacuda.plus_di(h32, l32, c32, 14), lambda: talib.PLUS_DI(h64, l64, c64, 14)),
        ("OBV", "was-critical", lambda: tacuda.obv(c32, v32), lambda: talib.OBV(c64, v64)),
        # Previously "high" / "medium"
        ("MACD(12,26,9)", "was-slow", lambda: tacuda.macd(c32, 12, 26, 9), lambda: talib.MACD(c64, 12, 26, 9)),
        ("SUM(14)", "was-slow", lambda: tacuda.sum(c32, 14), lambda: talib.SUM(c64, 14)),
        ("ROCR100(10)", "was-slow", lambda: tacuda.rocr100(c32, 10), lambda: talib.ROCR100(c64, 10)),
        ("SMA(14)", "was-slow", lambda: tacuda.sma(c32, 14), lambda: talib.SMA(c64, 14)),
        ("WMA(10)", "was-slow", lambda: tacuda.wma(c32, 10), lambda: talib.WMA(c64, 10)),
        ("EMA(12)", "was-slow", lambda: tacuda.ema(c32, 12), lambda: talib.EMA(c64, 12)),
        ("TRIX(14)", "was-slow", lambda: tacuda.trix(c32, 14), lambda: talib.TRIX(c64, 14)),
        ("TYPPRICE", "was-slow", lambda: tacuda.typprice(h32, l32, c32), lambda: talib.TYPPRICE(h64, l64, c64)),
        # Reference points that were already fast
        ("RSI(14)", "reference", lambda: tacuda.rsi(c32, 14), lambda: talib.RSI(c64, 14)),
        ("ATR(14)", "reference", lambda: tacuda.atr(h32, l32, c32, 14), lambda: talib.ATR(h64, l64, c64, 14)),
        ("ADX(14)", "reference", lambda: tacuda.adx(h32, l32, c32, 14), lambda: talib.ADX(h64, l64, c64, 14)),
        ("BBANDS(20)", "reference", lambda: tacuda.bbands(c32, 20, 2.0, 2.0), lambda: talib.BBANDS(c64, 20, 2, 2)),
        ("STDDEV(20)", "reference", lambda: tacuda.stddev(c32, 20), lambda: talib.STDDEV(c64, 20, 1)),
        # Sequential by design — documented, not a regression
        ("SAR", "sequential", lambda: tacuda.sar(h32, l32, 0.02, 0.2), lambda: talib.SAR(h64, l64, 0.02, 0.2)),
        ("HT_TRENDLINE", "sequential", lambda: tacuda.ht_trendline(c32), lambda: talib.HT_TRENDLINE(c64)),
        ("HT_DCPERIOD", "sequential", lambda: tacuda.ht_dcperiod(c32), lambda: talib.HT_DCPERIOD(c64)),
        ("MAXINDEX(30)", "sequential", lambda: tacuda.maxindex(c32, 30), lambda: talib.MAXINDEX(c64, 30)),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=500_000)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    data = make_data(args.size)
    cases = build_cases(data)

    print(f"tacuda {tacuda.version()} host API vs TA-Lib, {args.size:,} bars, "
          f"median of {args.repeats} runs\n")
    print(f"{'indicator':<16}{'group':<14}{'tacuda ms':>11}{'talib ms':>11}{'speedup':>10}")
    print("-" * 62)

    rows = []
    slower = []
    for name, group, ct_fn, tl_fn in cases:
        try:
            ct_ms = timeit(ct_fn, args.repeats)
            tl_ms = timeit(tl_fn, args.repeats)
        except Exception as exc:  # noqa: BLE001 - report and keep sweeping
            print(f"{name:<16}{group:<14}{'ERROR':>11}  {type(exc).__name__}: {exc}")
            rows.append({"indicator": name, "group": group, "error": str(exc)})
            continue

        speedup = tl_ms / ct_ms if ct_ms > 0 else float("inf")
        flag = "" if speedup >= 1.0 else "  <- slower"
        print(f"{name:<16}{group:<14}{ct_ms:>11.3f}{tl_ms:>11.3f}{speedup:>9.2f}x{flag}")
        rows.append({"indicator": name, "group": group, "tacuda_ms": ct_ms,
                     "talib_ms": tl_ms, "speedup": speedup})
        if speedup < 1.0:
            slower.append((name, group, speedup))

    print()
    if slower:
        print(f"{len(slower)} of {len(cases)} slower than TA-Lib through the host API:")
        for name, group, sp in sorted(slower, key=lambda r: r[2]):
            print(f"  {name:<16}{sp:.2f}x   ({group})")
    else:
        print("every measured indicator is at least as fast as TA-Lib "
              "through the host API")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"version": tacuda.version(), "bars": args.size,
                       "repeats": args.repeats, "rows": rows}, fh, indent=2)
        print(f"\nreport written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
