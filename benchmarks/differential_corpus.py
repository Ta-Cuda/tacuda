#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Differential corpus: TaCuda vs the installed TA-Lib, strict oracle.

Every spec from talib_compare_helpers.build_specs() runs over a corpus of
datasets and must pass compare_arrays() — which since 0.5.0 fails on any
NaN-mask, Inf-mask or shape difference, not only on numeric drift. This is
the source of truth behind any compatibility claim.

Datasets:
  rw-<seed>   random walks (three seeds; the classic case)
  trend       monotonic uptrend with noise
  flat        a fully constant series (flat-window semantics: RSI=0, ...)
  zeros       series crossing exact zeros (ROC/ROCP/ROCR zero-denominator)
  short       length 32 — around or below many lookbacks

Usage (on a GPU box with tacuda importable and TA-Lib installed):
    python benchmarks/differential_corpus.py [--json out.json] [--groups g1,g2]

Exit code 0 only when every (function, dataset) cell passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict

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

from talib_compare_helpers import (build_specs, compare_arrays,
                                   compare_discrete, generate_inputs)


def _bars_from_close(close: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    n = close.size
    spread_hi = rng.rand(n) * 2.0
    spread_lo = rng.rand(n) * 2.0
    high = close + spread_hi
    low = close - spread_lo
    open_ = close + (rng.rand(n) - 0.5)
    volume = rng.rand(n) * 1_000_000 + 1_000
    periods = rng.randint(2, 31, size=n).astype(np.float32)
    return {
        "close": close.astype(np.float64),
        "high": high.astype(np.float64),
        "low": low.astype(np.float64),
        "open": open_.astype(np.float64),
        "volume": volume.astype(np.float64),
        "close2": high.astype(np.float64),
        "periods": periods,
    }


def _candle_fixture() -> Dict[str, np.ndarray]:
    """Hand-crafted OHLC bars that ACTIVATE the rare candlestick patterns
    which never fire on random walks (3WHITESOLDIERS, 3BLACKCROWS,
    IDENTICAL3CROWS, KICKING(BYLENGTH), 3STARSINSOUTH, CONCEALBABYSWALL,
    MATHOLD, STALLEDPATTERN, 3LINESTRIKE). Without them the exact candle
    oracle would pass vacuously for those specs. Between injections the
    series is calm (small bodies/ranges) so the TA-Lib candle-setting
    averages stay low and the injected bars register as 'long'."""
    n = 512
    o = np.empty(n); h = np.empty(n); l = np.empty(n); c = np.empty(n)
    level = 100.0
    for i in range(n):
        d = 0.1 if i % 2 == 0 else -0.1
        o[i], c[i] = level - d, level + d
        h[i] = max(o[i], c[i]) + 0.15
        l[i] = min(o[i], c[i]) - 0.15

    def put(i0, bars):
        for k, (bo, bh, bl, bc) in enumerate(bars):
            o[i0 + k], h[i0 + k], l[i0 + k], c[i0 + k] = bo, bh, bl, bc

    # three white soldiers
    put(40, [(100.00, 103.02, 99.90, 103.00),
             (101.00, 105.02, 100.90, 105.00),
             (103.00, 108.02, 102.90, 108.00)])
    # three black crows (preceded by a white candle)
    put(80, [(100.00, 104.05, 99.95, 104.00),
             (103.50, 103.55, 100.49, 100.50),
             (102.50, 102.55, 98.99, 99.00),
             (100.00, 100.05, 96.99, 97.00)])
    # identical three crows (each opens at the prior close)
    put(120, [(104.00, 104.02, 99.99, 100.00),
              (100.00, 100.02, 95.99, 96.00),
              (96.00, 96.02, 91.99, 92.00)])
    # kicking / kicking by length (black marubozu, gap up, white marubozu)
    put(160, [(100.000, 100.005, 95.995, 96.000),
              (101.000, 105.005, 100.995, 105.000)])
    # three stars in the south
    put(200, [(100.00, 100.02, 91.00, 96.00),
              (98.00, 98.02, 93.00, 95.50),
              (95.40, 95.42, 94.78, 94.80)])
    # concealing baby swallow
    put(240, [(100.000, 100.005, 95.995, 96.000),
              (95.500, 95.505, 91.495, 91.500),
              (90.50, 92.50, 88.99, 89.00),
              (93.00, 93.01, 88.49, 88.50)])
    # mat hold: long white; small black gapping up; two small black reaction
    # days penetrating the first body (but < 50%); white close above all highs
    put(280, [(100.00, 105.01, 99.99, 105.00),
              (105.90, 106.00, 105.75, 105.80),
              (105.05, 105.15, 104.90, 104.95),
              (104.80, 104.90, 104.60, 104.70),
              (105.00, 108.01, 104.99, 108.00)])
    # stalled pattern
    put(320, [(100.00, 103.01, 99.99, 103.00),
              (101.50, 106.01, 101.40, 106.00),
              (105.90, 106.45, 105.85, 106.40)])
    # three-line strike (three soldiers + engulfing black strike)
    put(360, [(100.00, 103.02, 99.90, 103.00),
              (101.00, 105.02, 100.90, 105.00),
              (103.00, 108.02, 102.90, 108.00),
              (108.50, 108.60, 99.40, 99.50)])

    rng = np.random.RandomState(3)
    return {
        "open": o, "high": h, "low": l, "close": c,
        "volume": rng.rand(n) * 1_000_000 + 1_000,
        "close2": h.copy(),
        "periods": rng.randint(2, 31, size=n).astype(np.float32),
    }


def build_corpus(n: int) -> Dict[str, Dict[str, np.ndarray]]:
    corpus: Dict[str, Dict[str, np.ndarray]] = {}
    for seed in (42, 1337, 2026):
        corpus[f"rw-{seed}"] = generate_inputs(n, seed=seed)

    rng = np.random.RandomState(7)
    trend = 100.0 + np.arange(n) * 0.05 + rng.randn(n) * 0.3
    corpus["trend"] = _bars_from_close(trend, seed=7)

    flat_close = np.full(n, 123.45)
    flat = {
        "close": flat_close,
        "high": flat_close.copy(),
        "low": flat_close.copy(),
        "open": flat_close.copy(),
        "volume": np.full(n, 5_000.0),
        "close2": flat_close.copy(),
        "periods": np.full(n, 14.0, dtype=np.float32),
    }
    corpus["flat"] = flat

    zero_close = 50.0 * np.sin(np.arange(n) * 0.1)  # crosses exact zero
    zero_close[::37] = 0.0
    corpus["zeros"] = _bars_from_close(zero_close + 100.0, seed=11)
    corpus["zeros"]["close"] = zero_close  # keep zeros in the primary input

    corpus["short"] = generate_inputs(32, seed=5)
    corpus["candles"] = _candle_fixture()
    return corpus


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--json", default="")
    ap.add_argument("--groups", default="", help="comma-separated group filter")
    ap.add_argument("--datasets", default="", help="comma-separated dataset filter")
    args = ap.parse_args()

    specs = build_specs(tacuda, talib)

    # Coverage: every installed TA-Lib function must have at least one spec.
    # Exact match on the talib_name recorded by _maybe_add — no fuzzy
    # substring logic. Missing coverage fails the run (exit 1): otherwise the
    # compatibility claim is narrower than the corpus suggests.
    covered = {s.talib_name for s in specs if s.talib_name}
    missing = sorted(set(talib.get_functions()) - covered)
    if missing:
        print(f"COVERAGE FAIL: {len(missing)} TA-Lib functions without corpus "
              "specs: " + ", ".join(missing))

    if args.groups:
        wanted = {g.strip() for g in args.groups.split(",")}
        specs = [s for s in specs if s.group in wanted]
    corpus = build_corpus(args.size)
    if args.datasets:
        wanted = {d.strip() for d in args.datasets.split(",")}
        corpus = {k: v for k, v in corpus.items() if k in wanted}

    results = []
    failures = 0
    skipped = 0
    for spec in specs:
        for ds_name, data in corpus.items():
            if ds_name in getattr(spec, "skip_datasets", ()):
                skipped += 1
                continue
            cell = {"function": spec.name, "group": spec.group, "dataset": ds_name}
            try:
                got = spec.tacuda_fn(data)
                ref = spec.talib_fn(data)
            except Exception as exc:  # noqa: BLE001 - report, don't crash the sweep
                cell.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                results.append(cell)
                failures += 1
                continue

            got_t = got if isinstance(got, tuple) else (got,)
            ref_t = ref if isinstance(ref, tuple) else (ref,)
            if len(got_t) != len(ref_t):
                cell.update(ok=False, error=f"output arity {len(got_t)} != {len(ref_t)}")
                results.append(cell)
                failures += 1
                continue

            ok = True
            max_diff = 0.0
            nan_mm = 0
            discrete = getattr(spec, "discrete", False)
            cmp = compare_discrete if discrete else compare_arrays
            for g, r in zip(got_t, ref_t):
                ok_i, md, _, nm = cmp(
                    np.asarray(g, dtype=np.float64),
                    np.asarray(r, dtype=np.float64), spec.tolerance)
                ok &= ok_i
                max_diff = max(max_diff, md if np.isfinite(md) else float("inf"))
                nan_mm += max(nm, 0)
            cell.update(ok=bool(ok), maxDiff=max_diff, nanMismatch=nan_mm)
            if discrete:
                # An exact-equality pass over an all-zero reference proves
                # nothing; record how many signals the oracle actually had.
                r0 = np.asarray(ref_t[0], dtype=np.float64)
                finite = r0[~np.isnan(r0)]
                cell["referenceSignalCount"] = int(np.sum(finite != 0.0))
                cell["referencePositive"] = int(np.sum(finite > 0.0))
                cell["referenceNegative"] = int(np.sum(finite < 0.0))
            results.append(cell)
            if not ok:
                failures += 1

    total = len(results)
    passed = total - failures
    by_spec: Dict[str, bool] = {}
    for cell in results:
        by_spec[cell["function"]] = by_spec.get(cell["function"], True) and cell["ok"]
    specs_ok = sum(1 for v in by_spec.values() if v)

    # Candle activation: a pattern that never fires on any dataset passes the
    # exact oracle vacuously — surface those so fixtures can be added.
    signals_by_spec: Dict[str, int] = {}
    for cell in results:
        if "referenceSignalCount" in cell:
            signals_by_spec[cell["function"]] = (
                signals_by_spec.get(cell["function"], 0)
                + cell["referenceSignalCount"])
    inactive = sorted(n for n, c in signals_by_spec.items() if c == 0)
    if inactive:
        print(f"WARNING: {len(inactive)} candle specs never fire on this "
              "corpus (exact-equality pass is vacuous): " + ", ".join(inactive))

    skip_note = f", {skipped} cells skipped (documented deviations)" if skipped else ""
    print(f"differential corpus: {passed}/{total} cells pass, "
          f"{specs_ok}/{len(by_spec)} specs fully green "
          f"({len(corpus)} datasets x {len(specs)} specs{skip_note})")
    for cell in results:
        if not cell["ok"]:
            detail = cell.get("error") or (
                f"maxDiff={cell.get('maxDiff'):.3g} nanMismatch={cell.get('nanMismatch')}")
            print(f"  FAIL {cell['function']:24s} {cell['dataset']:8s} {detail}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"cells": results,
                       "summary": {"passed": passed, "total": total,
                                    "specsGreen": specs_ok,
                                    "specs": len(by_spec),
                                    "candleSignalsBySpec": signals_by_spec,
                                    "inactiveCandleSpecs": inactive,
                                    "coverageMissing": missing}}, fh, indent=1)
        print(f"wrote {args.json}")

    if missing:
        return 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
