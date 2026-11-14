#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""GPU regression for issue #6: one kernel, independent rows, TA-Lib parity.

Uses the same ctypes CUDA loader as the all-family graph sweep. Run with
--graph-only against an old build to reproduce the launch-count defect.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import json
import os
from pathlib import Path

import numpy as np
import talib

from batch_graph_capture_sweep import Cuda, Tacuda, find_cudart, load_specs, SCALAR_DEFAULTS

NAMES = ("mfi", "atr", "rsi", "adosc", "stddev", "bbands", "sum", "obv",
         "ad", "stochastic", "sma", "min", "max", "ema", "wma", "natr", "macd")
PARAMS = dict(SCALAR_DEFAULTS, period=14, fastPeriod=12, slowPeriod=26,
              signalPeriod=9, maType=1)
PARAMS_BY_NAME = {"adosc": dict(fastPeriod=3, slowPeriod=10)}
COLS = {"d_inputs": "close", "d_highs": "high", "d_lows": "low",
        "d_closes": "close", "d_volumes": "volume"}


def data(s, t, seed=6):
    rng = np.random.default_rng(seed)
    c = (100 + rng.normal(0, 0.4, (s, t)).cumsum(axis=1)).astype("f4")
    h = (c + rng.uniform(0.1, 1.5, c.shape)).astype("f4")
    l = (c - rng.uniform(0.1, 1.5, c.shape)).astype("f4")
    v = rng.uniform(10, 2000, c.shape).astype("f4")
    return dict(close=c, high=h, low=l, volume=v)


def reference(name, xs, p):
    outputs = []
    for row in range(xs["close"].shape[0]):
        c, h, l, v = (xs[k][row].astype("f8") for k in ("close", "high", "low", "volume"))
        n = p["period"]
        funcs = {
            "mfi": lambda: talib.MFI(h, l, c, v, n),
            "atr": lambda: talib.ATR(h, l, c, n),
            "rsi": lambda: talib.RSI(c, n),
            "adosc": lambda: talib.ADOSC(h, l, c, v, p["fastPeriod"], p["slowPeriod"]),
            "stddev": lambda: talib.STDDEV(c, n),
            "bbands": lambda: talib.BBANDS(c, n, float(p["upperMul"]), float(p["lowerMul"]), 0),
            "sum": lambda: talib.SUM(c, n), "obv": lambda: talib.OBV(c, v),
            "ad": lambda: talib.AD(h, l, c, v),
            "stochastic": lambda: talib.STOCH(h, l, c, p["kPeriod"], p["dPeriod"], 0, p["dPeriod"], 0),
            "sma": lambda: talib.SMA(c, n), "min": lambda: talib.MIN(c, n),
            "max": lambda: talib.MAX(c, n), "ema": lambda: talib.EMA(c, n),
            "wma": lambda: talib.WMA(c, n), "natr": lambda: talib.NATR(h, l, c, n),
            "macd": lambda: talib.MACD(c, p["fastPeriod"], p["slowPeriod"], p["signalPeriod"]),
        }
        result = funcs[name]()
        if not isinstance(result, tuple):
            result = (result,)
        outputs.append(result)
    return [np.array([r[k] for r in outputs], dtype="f4") for k in range(len(outputs[0]))]


def same(a, b):
    for x, y in zip(a, b):
        assert np.array_equal(np.isnan(x), np.isnan(y)), "NaN masks differ"
        good = ~np.isnan(x)
        assert np.array_equal(x[good].view("u4"), y[good].view("u4")), "finite bits differ"


class Case:
    def __init__(self, cuda, lib, spec, xs, stream, overrides=None):
        self.cuda, self.lib, self.spec, self.stream = cuda, lib, spec, stream
        self.s, self.t = xs["close"].shape
        self.p = dict(PARAMS, **PARAMS_BY_NAME.get(spec.c_name[3:-8], {}))
        self.p.update(overrides or {})
        self.ptrs = []
        self.ins = [self.alloc(xs[COLS[n]]) for n in spec.inputs]
        self.outs = [self.alloc() for _ in spec.outputs]

    def alloc(self, values=None):
        ptr = ct.c_void_p()
        self.cuda.check(self.cuda.cudaMalloc(ct.byref(ptr), self.s * self.t * 4), "alloc")
        self.ptrs.append(ptr)
        if values is not None:
            values = np.ascontiguousarray(values, dtype="f4")
            self.cuda.check(self.cuda.cudaMemcpy(ptr, values.ctypes.data, values.nbytes, 1), "upload")
        return ptr

    def args(self, host_ins=None, host_outs=None, concurrency=0):
        scalars = dict(self.p, seriesLen=self.t, numSeries=self.s,
                       maxConcurrency=concurrency, stream=self.stream)
        return list(host_ins or self.ins) + list(host_outs or self.outs) + [scalars[x.name] for x in self.spec.scalars]

    def launch(self, concurrency=0):
        st = self.lib.fns[self.spec.c_name](*self.args(concurrency=concurrency))
        assert st == 0, (self.spec.c_name, st, self.lib.last_error())

    def result(self):
        self.cuda.check(self.cuda.cudaStreamSynchronize(self.stream), "sync")
        result = []
        for ptr in self.outs:
            arr = np.empty((self.s, self.t), dtype="f4")
            self.cuda.check(self.cuda.cudaMemcpy(arr.ctypes.data, ptr, arr.nbytes, 2), "download")
            result.append(arr)
        return result

    def graph(self, expected=1):
        graph, executable = ct.c_void_p(), ct.c_void_p()
        self.cuda.check(self.cuda.cudaStreamBeginCapture(self.stream, 0), "begin capture")
        self.launch()
        self.cuda.check(self.cuda.cudaStreamEndCapture(self.stream, ct.byref(graph)), "end capture")
        try:
            count = ct.c_size_t()
            self.cuda.check(self.cuda.cudaGraphGetNodes(graph, None, ct.byref(count)), "nodes")
            assert count.value == expected, f"{self.spec.c_name}: graph has {count.value} nodes, expected {expected}"
            if expected == 1:
                nodes = (ct.c_void_p * 1)()
                self.cuda.check(self.cuda.cudaGraphGetNodes(graph, nodes, ct.byref(count)), "nodes")
                kind = ct.c_int()
                self.cuda.check(self.cuda.lib.cudaGraphNodeGetType(nodes[0], ct.byref(kind)), "node type")
                assert kind.value == 0, "the only graph node must be a kernel"
            self.cuda.check(self.cuda.cudaGraphInstantiateWithFlags(ct.byref(executable), graph, 0), "instantiate")
            for _ in range(3):
                for ptr in self.outs:
                    self.cuda.check(self.cuda.cudaMemsetAsync(ptr, 0x5A, self.s * self.t * 4, self.stream), "poison")
                self.cuda.check(self.cuda.cudaGraphLaunch(executable, self.stream), "replay")
                yield self.result()
        finally:
            if executable.value:
                self.cuda.cudaGraphExecDestroy(executable)
            self.cuda.cudaGraphDestroy(graph)

    def close(self):
        for ptr in reversed(self.ptrs):
            self.cuda.check(self.cuda.cudaFree(ptr), "free")


def check_case(cuda, lib, spec, xs, stream, **kwargs):
    c = Case(cuda, lib, spec, xs, stream, kwargs)
    try:
        c.launch()
        result = c.result()
        return result
    finally:
        c.close()


def legacy_rows(c):
    """The off switch must delegate to the existing single-series API."""
    name = c.spec.c_name.replace("_batch", "")
    fn = getattr(c.lib.lib, name)
    params = [x for x in c.spec.scalars if x.name not in ("seriesLen", "numSeries", "maxConcurrency", "stream")]
    fn.argtypes = ([ct.c_void_p] * (len(c.ins) + len(c.outs)) + [ct.c_int]
                   + [{"int": ct.c_int, "float": ct.c_float}[x.ctype] for x in params] + [ct.c_void_p])
    fn.restype = ct.c_int
    for s in range(c.s):
        offset = s * c.t * 4
        args = [v.value + offset for v in c.ins + c.outs] + [c.t]
        args += [c.p[x.name] for x in params] + [c.stream]
        assert fn(*args) == 0, (name, c.lib.last_error())
    return c.result()


def graph_count(c):
    graph = ct.c_void_p()
    c.cuda.check(c.cuda.cudaStreamBeginCapture(c.stream, 0), "begin capture")
    c.launch()
    c.cuda.check(c.cuda.cudaStreamEndCapture(c.stream, ct.byref(graph)), "end capture")
    try:
        count = ct.c_size_t()
        c.cuda.check(c.cuda.cudaGraphGetNodes(graph, None, ct.byref(count)), "nodes")
        return count.value
    finally:
        c.cuda.cudaGraphDestroy(graph)


def dispatch_checks(cuda, lib, spec, stream):
    recurrent = spec.c_name[3:-8] in ("mfi", "atr", "rsi", "ema", "natr", "obv", "ad", "adosc", "macd")
    for t in (32, 33):  # exact boundary length == 4 * series
        c = Case(cuda, lib, spec, data(8, t), stream)
        try:
            os.environ["TACUDA_BATCH_FUSED"] = "auto"
            n = graph_count(c)
            assert (n == 1) == (not recurrent or t == 32), (spec.c_name, t, n)
            os.environ["TACUDA_BATCH_FUSED"] = "1"
            assert graph_count(c) == 1
            os.environ["TACUDA_BATCH_FUSED"] = "0"
            assert graph_count(c) > 1
            c.launch()
            same(c.result(), legacy_rows(c))
        finally:
            c.close()
    # Unsupported windows and MACD types retain their original implementation.
    name = spec.c_name[3:-8]
    variants = []
    if name in ("mfi", "stddev", "bbands", "sum", "sma", "min", "max", "wma"):
        variants.append(dict(period=65))
    if name == "stochastic":
        variants += [dict(kPeriod=65), dict(dPeriod=33)]
    if name == "macd":
        variants.append(dict(maType=0))
    for v in variants:
        c = Case(cuda, lib, spec, data(5, 171), stream, v)
        try:
            os.environ["TACUDA_BATCH_FUSED"] = "1"
            assert graph_count(c) > 1
            c.launch()
            same(c.result(), legacy_rows(c))
        finally:
            c.close()
    os.environ["TACUDA_BATCH_FUSED"] = "1"


def invalid_checks(cuda, lib, spec, stream):
    c = Case(cuda, lib, spec, data(3, 50), stream)
    try:
        # Compare error classification with the existing dispatcher; none of
        # these calls may enqueue a kernel or write the output matrices.
        base = c.args()
        offset = len(c.ins) + len(c.outs)
        indices = {x.name: offset + i for i, x in enumerate(spec.scalars)}
        changes = [(0, None), (len(c.ins), None),
                   (indices["seriesLen"], 0), (indices["numSeries"], -1)]
        for scalar in ("period", "kPeriod", "dPeriod", "fastPeriod", "slowPeriod", "signalPeriod"):
            if scalar in indices:
                changes.append((indices[scalar], 0))
        for index, value in changes:
            statuses = []
            for mode in ("0", "1"):
                os.environ["TACUDA_BATCH_FUSED"] = mode
                args = base.copy()
                args[index] = value
                statuses.append(lib.fns[spec.c_name](*args))
            assert statuses[0] != 0 and statuses[0] == statuses[1], (spec.c_name, index, statuses)
    finally:
        c.close()
        os.environ["TACUDA_BATCH_FUSED"] = "1"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--cudart-dir")
    ap.add_argument("--graph-only", action="store_true")
    ap.add_argument("--only", choices=NAMES)
    ap.add_argument("--json")
    opts = ap.parse_args()
    cuda = Cuda(find_cudart(opts.cudart_dir))
    cuda.lib.cudaGraphNodeGetType.argtypes = [ct.c_void_p, ct.POINTER(ct.c_int)]
    specs = {f.c_name[3:-8]: f for f in load_specs() if f.c_name[3:-8] in NAMES}
    lib = Tacuda(str(Path(opts.lib).resolve()), list(specs.values()))
    stream = ct.c_void_p()
    cuda.check(cuda.cudaStreamCreateWithFlags(ct.byref(stream), 1), "create stream")
    os.environ["TACUDA_BATCH_FUSED"] = "1"
    metrics, errors = {}, []
    names = [opts.only] if opts.only else NAMES
    try:
        for name in names:
            spec = specs[name]
            c = Case(cuda, lib, spec, data(67, 234), stream)
            try:
                c.launch()
                eager = c.result()
                for replay in c.graph():
                    same(replay, eager)
                if opts.graph_only:
                    print(f"PASS {name}: one kernel + replay", flush=True)
                    continue
                # Host and default-stream contracts use the same arithmetic.
                fn = getattr(lib.lib, spec.c_name[:-2])
                fn.argtypes, fn.restype = lib.fns[spec.c_name].argtypes, ct.c_int
                xs = data(67, 234)
                houts = [np.empty_like(xs["close"]) for _ in spec.outputs]
                assert fn(*c.args([xs[COLS[n]].ctypes.data for n in spec.inputs],
                                  [v.ctypes.data for v in houts], 3)) == 0
                same(eager, houts)
                c.stream = None
                c.launch(7)
                same(eager, c.result())
                bit_diff = 0
                for shape in ((1, 1), (2, 16), (17, 19), (67, 234), (3, 513)):
                    xs = data(*shape)
                    out = check_case(cuda, lib, spec, xs, stream)
                    refs = reference(name, xs, c.p)
                    for a, b in zip(out, refs):
                        np.testing.assert_allclose(a, b, rtol=3e-6, atol=3e-5, equal_nan=True, err_msg=f"{name} {shape}")
                        finite = np.isfinite(b)
                        bit_diff += int(np.count_nonzero(a[finite].view("u4") != b[finite].view("u4")))
                    # Rows, their order, batch size and appended future bars cannot affect a row's prefix.
                    if shape == (67, 234):
                        perm = np.random.default_rng(7).permutation(shape[0])
                        shuffled = check_case(cuda, lib, spec, {k: v[perm] for k, v in xs.items()}, stream)
                        same([a[perm] for a in out], shuffled)
                        single = check_case(cuda, lib, spec, {k: v[31:32] for k, v in xs.items()}, stream)
                        same([a[31:32] for a in out], single)
                        prefix = check_case(cuda, lib, spec, {k: v[:, :117] for k, v in xs.items()}, stream)
                        same([a[:, :117] for a in out], prefix)
                        for at in (0, 110, 233):
                            bad = {k: v.copy() for k, v in xs.items()}
                            for v in bad.values():
                                v[0, at] = np.nan
                            poisoned = check_case(cuda, lib, spec, bad, stream)
                            same([a[1:] for a in out], [a[1:] for a in poisoned])
                # Corpus of valid periods, reversed period roles, zero ranges
                # and subnormal-sized changes. Every output is initialized.
                periods = (1, 2, 3, 63, 64) if name in ("atr", "natr") else (2, 3, 63, 64)
                variants = [dict(period=p) for p in periods]
                if name == "adosc":
                    variants = [dict(fastPeriod=10, slowPeriod=3), dict(fastPeriod=2, slowPeriod=2)]
                if name == "macd":
                    variants = [dict(fastPeriod=26, slowPeriod=12), dict(fastPeriod=2, slowPeriod=3, signalPeriod=2)]
                if name == "stochastic":
                    variants = [dict(kPeriod=2, dPeriod=1), dict(kPeriod=5, dPeriod=5)]
                for v in variants:
                    xs = data(7, 257, seed=17)
                    params = dict(c.p, **v)
                    result = check_case(cuda, lib, spec, xs, stream, **v)
                    for a, b in zip(result, reference(name, xs, params)):
                        np.testing.assert_allclose(a, b, rtol=3e-6, atol=3e-5, equal_nan=True, err_msg=f"{name} {v}")
                flat = {k: np.full((3, 129), 100.0, dtype="f4") for k in ("close", "high", "low", "volume")}
                result = check_case(cuda, lib, spec, flat, stream)
                for a, b in zip(result, reference(name, flat, c.p)):
                    np.testing.assert_array_equal(a, b)
                if name == "ema":
                    unit = {"close": np.array([[1e30, 1.0, -1e30, 1.0, np.nan, 2.0]], dtype="f4")}
                    out = check_case(cuda, lib, spec, unit, stream, period=1)
                    same(out, [unit["close"]])
                if name == "macd":
                    unit = {"close": np.arange(1.0, 40.0, dtype="f4")[None, :]}
                    out = check_case(cuda, lib, spec, unit, stream, fastPeriod=2, slowPeriod=3, signalPeriod=1)
                    for line, value in zip(out, (0.5, 0.5, 0.0)):
                        assert np.isnan(line[:, :2]).all()
                        np.testing.assert_array_equal(line[:, 2:], value)
                if name != "stochastic":
                    assert bit_diff == 0, f"{name}: {bit_diff} finite bits differ from the primary TA-Lib fixtures"
                dispatch_checks(cuda, lib, spec, stream)
                invalid_checks(cuda, lib, spec, stream)
                metrics[name] = {"talib_bit_differences": bit_diff}
                print(f"PASS {name}: graph, TA-Lib, host/default stream, permutation, batch size, prefix, NaN isolation; bitDiff={bit_diff}", flush=True)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                print(f"FAIL {name}: {exc}", flush=True)
            finally:
                c.close()
    finally:
        cuda.cudaStreamDestroy(stream)
    report = dict(metrics=metrics, errors=errors, talib=talib.__version__)
    if opts.json:
        Path(opts.json).write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        raise SystemExit(f"{len(errors)} families failed")


if __name__ == "__main__":
    main()
