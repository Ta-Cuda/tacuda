#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Streaming corpus: incremental appends vs full recompute vs TA-Lib.

The streaming API's whole promise is that appending bars reproduces what a
full-series call would have produced. This sweep checks that twice over,
under the same strict oracle the differential corpus uses (identical shapes,
identical NaN masks, identical Inf masks/signs, mixed abs/rel tolerance):

  stream vs tacuda   appending in chunks of 1 / 3 / 16 / all must equal the
                     full-series tacuda call on the same data
  stream vs talib    and it must equal the installed TA-Lib as well, which
                     is the claim users actually care about

Batch streams are checked too: a portfolio of N series must equal N separate
single-series streams, on both execution paths (CPU and GPU).

Usage (GPU box with tacuda importable and TA-Lib installed):
    python benchmarks/streaming_corpus.py [--json out.json]

Exit code 0 only when every cell passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Callable, Dict, List, Sequence, Tuple

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

from differential_corpus import build_corpus
from talib_compare_helpers import compare_arrays, to_talib_f64

CHUNKS = (1, 3, 16, 0)  # 0 == one bulk append


class Spec:
    """One streaming family: how to build it, feed it and reference it."""

    def __init__(self, name: str, factory: Callable[[], object],
                 batch_factory: Callable[[int], object],
                 inputs: Sequence[str], outputs: int,
                 full: Callable[[Dict[str, np.ndarray]], object],
                 reference: Callable[[Dict[str, np.ndarray]], object] | None,
                 tolerance: float,
                 skip_reference_on: Sequence[str] = ()) -> None:
        self.name = name
        self.factory = factory
        self.batch_factory = batch_factory
        self.inputs = tuple(inputs)
        self.outputs = outputs
        self.full = full
        self.reference = reference
        self.tolerance = tolerance
        # Datasets where TA-Lib itself is the odd one out; the stream-vs-tacuda
        # check still runs, only the external reference is skipped.
        self.skip_reference_on = tuple(skip_reference_on)


def _f32(data: Dict[str, np.ndarray], key: str) -> np.ndarray:
    return np.ascontiguousarray(data[key], dtype=np.float32)


def _ref_in(data: Dict[str, np.ndarray], key: str) -> np.ndarray:
    """Input for the TA-Lib reference, quantized to float32 first.

    TA-Lib computes in double; the streams see float32 prices. Feeding TA-Lib
    the un-quantized float64 series compares two different inputs, and for a
    cumulative line like AD the divergence compounds over thousands of bars —
    that is a property of the data, not of the implementation.
    """
    return to_talib_f64(data[key])


def _as_tuple(value) -> Tuple[np.ndarray, ...]:
    return value if isinstance(value, tuple) else (value,)


def build_specs() -> List[Spec]:
    st = tacuda.stream
    specs: List[Spec] = []

    def add(name, factory, batch_factory, inputs, outputs, full, reference,
            tol=2e-5, skip_reference_on=()):
        specs.append(Spec(name, factory, batch_factory, inputs, outputs, full,
                          reference, tol, skip_reference_on))

    add("SMA", lambda: st.Sma(20), lambda n: st.SmaBatch(20, num_series=n),
        ("close",), 1,
        lambda d: tacuda.sma(_f32(d, "close"), 20),
        lambda d: talib.SMA(_ref_in(d, "close"), timeperiod=20))
    add("EMA", lambda: st.Ema(21), lambda n: st.EmaBatch(21, num_series=n),
        ("close",), 1,
        lambda d: tacuda.ema(_f32(d, "close"), 21),
        lambda d: talib.EMA(_ref_in(d, "close"), timeperiod=21))
    add("WMA", lambda: st.Wma(18), lambda n: st.WmaBatch(18, num_series=n),
        ("close",), 1,
        lambda d: tacuda.wma(_f32(d, "close"), 18),
        lambda d: talib.WMA(_ref_in(d, "close"), timeperiod=18))
    add("RSI", lambda: st.Rsi(14), lambda n: st.RsiBatch(14, num_series=n),
        ("close",), 1,
        lambda d: tacuda.rsi(_f32(d, "close"), 14),
        lambda d: talib.RSI(_ref_in(d, "close"), timeperiod=14))
    add("ATR", lambda: st.Atr(14), lambda n: st.AtrBatch(14, num_series=n),
        ("high", "low", "close"), 1,
        lambda d: tacuda.atr(_f32(d, "high"), _f32(d, "low"), _f32(d, "close"), 14),
        lambda d: talib.ATR(_ref_in(d, "high"), _ref_in(d, "low"), _ref_in(d, "close"), timeperiod=14))
    add("NATR", lambda: st.Natr(14), lambda n: st.NatrBatch(14, num_series=n),
        ("high", "low", "close"), 1,
        lambda d: tacuda.natr(_f32(d, "high"), _f32(d, "low"), _f32(d, "close"), 14),
        lambda d: talib.NATR(_ref_in(d, "high"), _ref_in(d, "low"), _ref_in(d, "close"), timeperiod=14),
        # Same exclusion the differential corpus makes: TA-Lib's ta_NATR.c
        # writes the zero-close guard value into outReal[0] instead of
        # outReal[outIdx], corrupting its own FIRST output bar. TaCuda emits 0
        # at the zero-close bar itself, so on the zero-crossing dataset the
        # reference is the wrong one, not the stream.
        skip_reference_on=("zeros",))
    add("OBV", lambda: st.Obv(), lambda n: st.ObvBatch(num_series=n),
        ("close", "volume"), 1,
        lambda d: tacuda.obv(_f32(d, "close"), _f32(d, "volume")),
        lambda d: talib.OBV(_ref_in(d, "close"), _ref_in(d, "volume")),
        tol=1e-4)
    add("AD", lambda: st.Ad(), lambda n: st.AdBatch(num_series=n),
        ("high", "low", "close", "volume"), 1,
        lambda d: tacuda.ad(_f32(d, "high"), _f32(d, "low"), _f32(d, "close"),
                            _f32(d, "volume")),
        lambda d: talib.AD(_ref_in(d, "high"), _ref_in(d, "low"), _ref_in(d, "close"), _ref_in(d, "volume")),
        tol=1e-4)
    add("MACD", lambda: st.Macd(12, 26, 9),
        lambda n: st.MacdBatch(12, 26, 9, num_series=n),
        ("close",), 3,
        lambda d: tacuda.macd(_f32(d, "close"), 12, 26, 9),
        lambda d: talib.MACD(_ref_in(d, "close"), fastperiod=12, slowperiod=26,
                             signalperiod=9))
    add("BBANDS", lambda: st.Bbands(20, 2.0, 2.0),
        lambda n: st.BbandsBatch(20, 2.0, 2.0, num_series=n),
        ("close",), 3,
        lambda d: tacuda.bbands(_f32(d, "close"), 20, 2.0, 2.0),
        lambda d: talib.BBANDS(_ref_in(d, "close"), timeperiod=20, nbdevup=2.0, nbdevdn=2.0))
    return specs


def run_stream(spec: Spec, data: Dict[str, np.ndarray],
               chunk: int) -> Tuple[np.ndarray, ...]:
    """Feed the dataset through a single-series stream in fixed-size chunks."""
    cols = [_f32(data, name) for name in spec.inputs]
    n = cols[0].size
    step = n if chunk == 0 else chunk
    outs = [np.empty(n, dtype=np.float32) for _ in range(spec.outputs)]

    with spec.factory() as s:
        off = 0
        while off < n:
            take = min(step, n - off)
            got = _as_tuple(s.append(*[c[off:off + take] for c in cols]))
            for k, line in enumerate(got):
                outs[k][off:off + take] = line
            off += take
    return tuple(outs)


def run_batch(spec: Spec, data: Dict[str, np.ndarray], series: int,
              exec_policy: int) -> Tuple[np.ndarray, ...]:
    """Same data replicated across `series` rows; every row must agree."""
    cols = [_f32(data, name) for name in spec.inputs]
    n = cols[0].size
    mats = [np.repeat(c[None, :], series, axis=0) for c in cols]
    # Shift each row slightly so the rows are not trivially identical work.
    for m in mats:
        for r in range(series):
            m[r] = np.roll(m[r], r)

    with spec.batch_factory(series) as b:
        b.set_exec(exec_policy)
        got = _as_tuple(b.append(*mats))
    return got, mats


def compare(label: str, got: Sequence[np.ndarray], ref: Sequence[np.ndarray],
            tolerance: float) -> Tuple[bool, dict]:
    if len(got) != len(ref):
        return False, {"error": f"arity {len(got)} != {len(ref)}"}
    ok = True
    worst = 0.0
    nan_mm = 0
    for g, r in zip(got, ref):
        passed, diff, _, mism = compare_arrays(np.asarray(g, dtype=np.float64),
                                               np.asarray(r, dtype=np.float64),
                                               tolerance)
        ok = ok and passed
        worst = max(worst, diff if np.isfinite(diff) else float("inf"))
        nan_mm += max(0, mism)
    return ok, {"max_diff": worst, "nan_mismatch": nan_mm}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=4096)
    ap.add_argument("--json", default="")
    ap.add_argument("--series", type=int, default=17,
                    help="rows in the batch-stream check")
    args = ap.parse_args()

    specs = build_specs()
    corpus = build_corpus(args.size)
    # The candle fixture has no bearing on these families.
    corpus.pop("candles", None)

    results: List[dict] = []
    failures = 0

    for spec in specs:
        for ds_name, data in corpus.items():
            # --- single-series: every chunking must agree with a full recompute
            full = _as_tuple(spec.full(data))
            baseline = None
            for chunk in CHUNKS:
                cell = {"function": spec.name, "dataset": ds_name,
                        "check": f"stream(chunk={chunk or 'all'}) vs tacuda"}
                try:
                    got = run_stream(spec, data, chunk)
                except Exception as exc:  # noqa: BLE001
                    cell.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                    results.append(cell)
                    failures += 1
                    continue
                if baseline is None:
                    baseline = got
                ok, info = compare(cell["check"], got, full, spec.tolerance)
                cell.update(ok=ok, **info)
                results.append(cell)
                failures += 0 if ok else 1

                # Chunking must not change a single bit.
                if chunk != CHUNKS[0]:
                    same = all(np.array_equal(a, b, equal_nan=True)
                               for a, b in zip(got, baseline))
                    cell2 = {"function": spec.name, "dataset": ds_name,
                             "check": f"chunk={chunk or 'all'} bit-equals chunk=1",
                             "ok": same}
                    results.append(cell2)
                    failures += 0 if same else 1

            # --- and against TA-Lib itself
            if (spec.reference is not None and baseline is not None
                    and ds_name not in spec.skip_reference_on):
                cell = {"function": spec.name, "dataset": ds_name,
                        "check": "stream vs talib"}
                try:
                    ref = _as_tuple(spec.reference(data))
                    ok, info = compare(cell["check"], baseline, ref, spec.tolerance)
                    cell.update(ok=ok, **info)
                except Exception as exc:  # noqa: BLE001
                    cell.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                    ok = False
                results.append(cell)
                failures += 0 if cell["ok"] else 1

            # --- batch streams: every row equals its own single-series stream
            for policy, pname in ((tacuda.stream.EXEC_CPU, "cpu"),
                                  (tacuda.stream.EXEC_GPU, "gpu")):
                cell = {"function": spec.name, "dataset": ds_name,
                        "check": f"batch({args.series}, {pname}) vs single streams"}
                try:
                    got, mats = run_batch(spec, data, args.series, policy)
                    ok = True
                    worst = 0.0
                    for r in range(args.series):
                        row_data = {name: mats[i][r]
                                    for i, name in enumerate(spec.inputs)}
                        single = run_stream(spec, row_data, 0)
                        row_ok, info = compare("row", [g[r] for g in got], single,
                                               0.0)  # must be exact
                        ok = ok and row_ok
                        worst = max(worst, info.get("max_diff", 0.0))
                    cell.update(ok=ok, max_diff=worst)
                except Exception as exc:  # noqa: BLE001
                    cell.update(ok=False, error=f"{type(exc).__name__}: {exc}")
                results.append(cell)
                failures += 0 if cell["ok"] else 1

    total = len(results)
    passed = sum(1 for r in results if r.get("ok"))
    print(f"\nstreaming corpus: {passed}/{total} cells passed "
          f"({len(specs)} families x {len(corpus)} datasets)")
    if failures:
        print("\nFAILURES:")
        for r in results:
            if not r.get("ok"):
                print(f"  {r['function']:8s} {r['dataset']:8s} {r['check']}: "
                      f"{r.get('error') or r}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"version": tacuda.version(), "cells": results}, fh, indent=2)
        print(f"report written to {args.json}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
