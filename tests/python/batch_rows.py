#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Row-batched legacy families: whole-matrix execution, bit-identical rows.

Every family with a row plan (RowsPlan, scripts/generate_batch_api.py) must
return exactly the bits of the per-series dispatcher (TACUDA_BATCH_FUSED=0),
NaN payloads included, across shapes, parameters and data with non-finite
values and flat stretches. Independently of the dispatcher, a direct
single-series call of the family's ct_*_d on each row must give the same
bits. The batched path must be taken whenever the rule below expects it,
issue a number of operations that does not grow with the number of rows,
capture into a CUDA graph whose replay repeats the eager result, and give
the same result through the host API, the legacy default stream and a
caller's stream. Invalid parameters keep the original status and message.
With --ref-lib every case must also match another build bit for bit, e.g.
the release before row batching.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import json
import os
import sys
from pathlib import Path

import numpy as np

from batch_graph_capture_sweep import Cuda, Tacuda, find_cudart, load_specs, SCALAR_DEFAULTS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import generate_batch_api as gen  # noqa: E402  (the plans are defined there)

COLS = {"d_inputs": "close", "d_opens": "open", "d_highs": "high", "d_lows": "low",
        "d_closes": "close", "d_volumes": "volume", "d_xs": "close", "d_ys": "high"}
SHAPES = ((2, 16), (5, 63), (17, 64), (67, 234), (3, 513))
CTRL = ("seriesLen", "numSeries", "maxConcurrency", "stream")
POISON = 0x5A5A5A5A
FLAT_ROW_LIMIT = (2 ** 31 - 1) // 4
FLAT_BLOCK = 1 << 22  # elements per flattened call
# Fewer rows than this, each at least LONG_ROW long: the per-series path.
LONG_ROW_SERIES = 64
LONG_ROW = {"accbands": 1 << 14, "trima": 1 << 14}  # default 1 << 16
# Shape spanning two flat blocks (1677 + 323 rows) for a sample of families.
BLOCK_SHAPE = (2000, 2500)
BLOCK_FAMILIES = ("typprice", "trange", "willr", "aroon", "stochf", "trima", "cdl_doji",
                  "cdl_hikkake_mod", "pivot_low", "sar", "ht_sine")
# Shapes on both sides of the long-row rule.
LONG_ROW_CASES = {"willr": ((8, 70000), (64, 70000)), "cdl_doji": ((2, 65536), (2, 65535)),
                  "accbands": ((32, 20000), (64, 20000), (32, 16383))}
# Upper bound of graph nodes on the batched path (ACCBANDS period 1: three
# SMA calls with copies plus scratch); it must not depend on the row count.
MAX_BATCHED_NODES = 24

# Stacked single-series wrappers allocate scratch, so their graphs need
# CUDA memory pools: on the per-series path, and on the flat batched path
# that calls the same wrapper once.
LEGACY_SCRATCH = {"aroon", "minmax", "stochf", "trima", "accbands", "mama", "macdfix",
                  "minmaxindex", "ht_phasor", "ht_sine"}
BATCHED_SCRATCH = {"aroon", "minmax", "stochf", "trima", "accbands"}
PRICE = ("typprice", "wclprice", "medprice", "avgprice", "bop")
LAG = ("momentum", "roc", "rocp", "rocr", "rocr100", "change", "aroon", "aroonosc",
       "market_strength")
WINDOW = ("willr", "midprice", "var", "linearreg", "linearreg_slope",
          "linearreg_intercept", "linearreg_angle", "tsf", "minmax", "cci", "avgdev",
          "skew", "kurtosis", "rolling_mean", "rolling_max", "rolling_min", "rolling_std")


def sma_direct(period):
    """tacuda::SMA strategy with the default configuration: direct windows
    below the prefix-sum threshold (96)."""
    return 1 <= period < 96


def expected_batched(name, params, s, t):
    """Independent statement of when the batched path must run."""
    if s < 2:
        return False
    p = dict(SCALAR_DEFAULTS, **params)
    if gen.ROWS[name].kind == "rows":
        if name in ("maxindex", "minindex", "minmaxindex", "midpoint"):
            return p["period"] > 0
        return name != "macdfix" or p["signalPeriod"] > 0
    if name.startswith("cdl_"):
        warmup, ok = int(gen.ROWS[name].warmup), True
    elif name in PRICE:
        warmup, ok = 0, True
    elif name == "trange":
        warmup, ok = 1, True
    elif name in LAG:
        warmup, ok = p["period"], p["period"] > 0
    elif name in WINDOW:
        warmup, ok = p["period"] - 1, p["period"] > 0
    elif name == "ultosc":
        periods = (p["period1"], p["period2"], p["period3"])
        warmup, ok = max(periods), min(periods) > 0
    elif name == "stochf":
        warmup, ok = p["kPeriod"] + p["dPeriod"] - 2, p["kPeriod"] > 0 and p["dPeriod"] > 0
    elif name == "trima":
        n = p["period"]
        half = (n + 1) // 2
        warmup = n - 1
        ok = n > 0 and sma_direct(half) and sma_direct(half + 1 if n % 2 == 0 else half)
    elif name == "accbands":
        warmup, ok = p["period"] - 1, p["period"] > 0 and sma_direct(p["period"])
    elif name == "trendscore":
        warmup, ok = p["lookback"] + 10, p["lookback"] >= 0
    elif name == "refdev":
        warmup, ok = p["smaPeriod"] + p["sumPeriod"] - 2, p["smaPeriod"] > 0 and p["sumPeriod"] > 0
    elif name in ("pivot_high", "pivot_low"):
        warmup, ok = 2 * p["strength"], p["strength"] > 0
    else:
        raise KeyError(f"no expectation for {name}")
    if s < LONG_ROW_SERIES and t >= LONG_ROW.get(name, 1 << 16):
        return False
    return ok and 0 <= warmup < t <= FLAT_ROW_LIMIT


def data(s, t, kind, seed=11):
    rng = np.random.default_rng(seed + 7 * s + t)
    close = 100 + rng.normal(0, 0.6, (s, t)).cumsum(axis=1)
    open_ = np.concatenate([close[:, :1], close[:, :-1]], axis=1) + rng.normal(0, 0.2, (s, t))
    high = np.maximum(open_, close) + rng.uniform(0.0, 1.2, (s, t))
    low = np.minimum(open_, close) - rng.uniform(0.0, 1.2, (s, t))
    volume = rng.uniform(10, 5000, (s, t))
    cols = dict(open=open_, high=high, low=low, close=close, volume=volume)
    if kind == "flat":
        # Zero ranges and repeated values hit the division and tie branches.
        for k in cols:
            cols[k][:, t // 3: 2 * t // 3] = cols["close"][:, t // 3: t // 3 + 1]
        cols["volume"][:, ::5] = 0.0
    cols = {k: v.astype("f4") for k, v in cols.items()}
    if kind == "nonfinite":
        spots = [(0, 0, np.nan), (s - 1, t // 2, np.nan), (s // 2, t - 1, np.nan),
                 (min(1, s - 1), min(5, t - 1), np.inf), (s - 1, min(9, t - 1), -np.inf)]
        for row, col, value in spots:
            for k in ("close", "high", "low", "open"):
                cols[k][row, col] = value
        cols["volume"][s // 2, min(3, t - 1)] = np.nan
    return cols


def variants(name):
    if name.startswith(("cdl_", "ht_")) or name in PRICE or name == "trange":
        return [{}]
    table = {
        "ultosc": [dict(period1=7, period2=14, period3=28), dict(period1=2, period2=3, period3=4),
                   dict(period1=28, period2=14, period3=7), dict(period1=1, period2=1, period3=1)],
        "stochf": [dict(kPeriod=14, dPeriod=3), dict(kPeriod=5, dPeriod=1),
                   dict(kPeriod=1, dPeriod=1), dict(kPeriod=20, dPeriod=10)],
        "refdev": [dict(smaPeriod=10, sumPeriod=10), dict(smaPeriod=2, sumPeriod=3),
                   dict(smaPeriod=1, sumPeriod=1), dict(smaPeriod=30, sumPeriod=20)],
        "trendscore": [dict(lookback=0), dict(lookback=1), dict(lookback=10), dict(lookback=40)],
        "pivot_high": [dict(strength=1), dict(strength=3), dict(strength=10)],
        "pivot_low": [dict(strength=1), dict(strength=3), dict(strength=10)],
        "sar": [{}, dict(step=0.05, maxAcceleration=0.5)],
        "sarext": [{}, dict(startValue=-5.0, offsetOnReverse=0.01, accInitLong=0.03,
                            accLong=0.02, accMaxLong=0.3, accInitShort=0.01,
                            accShort=0.03, accMaxShort=0.25)],
        "sar_el": [{}, dict(accFact=0.1, maxAccel=0.3)],
        "mama": [{}, dict(fastLimit=0.9, slowLimit=0.1)],
        "macdfix": [dict(signalPeriod=9), dict(signalPeriod=1), dict(signalPeriod=20)],
    }
    if name in table:
        return table[name]
    periods = [1, 2, 3, 5, 14, 30, 64]
    if name in ("trima", "accbands"):
        periods += [96, 200]  # past the SMA direct-window limit: per-series path
    return [dict(period=p) for p in periods]


class Run:
    """One family on one matrix: device inputs, outputs and the batch call."""

    def __init__(self, cuda, lib, spec, xs):
        self.cuda, self.lib, self.spec = cuda, lib, spec
        self.s, self.t = xs["close"].shape
        self.bytes = self.s * self.t * 4
        self.ptrs = []
        self.ins = [self.alloc(xs[COLS[n]]) for n in spec.inputs]
        self.outs = [self.alloc() for _ in spec.outputs]

    def alloc(self, values=None):
        ptr = ct.c_void_p()
        self.cuda.check(self.cuda.cudaMalloc(ct.byref(ptr), self.bytes), "alloc")
        self.ptrs.append(ptr)
        if values is not None:
            values = np.ascontiguousarray(values, dtype="f4")
            self.cuda.check(self.cuda.cudaMemcpy(ptr, values.ctypes.data, values.nbytes, 1), "upload")
        return ptr

    def args(self, params, stream, ins=None, outs=None):
        scalars = dict(SCALAR_DEFAULTS, **params)
        scalars.update(seriesLen=self.t, numSeries=self.s, maxConcurrency=0, stream=stream)
        return list(ins or self.ins) + list(outs or self.outs) + [scalars[x.name] for x in self.spec.scalars]

    def poison(self, stream):
        for ptr in self.outs:
            self.cuda.check(self.cuda.cudaMemsetAsync(ptr, 0x5A, self.bytes, stream), "poison")

    def call(self, params, stream, mode, lib=None):
        lib = lib or self.lib
        os.environ["TACUDA_BATCH_FUSED"] = mode
        self.poison(stream)
        st = lib.fns[self.spec.c_name](*self.args(params, stream))
        return st, lib.last_error()

    def read(self, stream):
        self.cuda.check(self.cuda.cudaStreamSynchronize(stream), "sync")
        result = []
        for ptr in self.outs:
            arr = np.empty((self.s, self.t), dtype="u4")
            self.cuda.check(self.cuda.cudaMemcpy(arr.ctypes.data, ptr, self.bytes, 2), "download")
            result.append(arr)
        return result

    def close(self):
        for ptr in reversed(self.ptrs):
            self.cuda.check(self.cuda.cudaFree(ptr), "free")


def same_bits(a, b, what):
    for k, (x, y) in enumerate(zip(a, b)):
        if not np.array_equal(x, y):
            bad = np.argwhere(x != y)
            r, c = bad[0]
            raise AssertionError(f"{what}: output {k} differs in {len(bad)} elements, first at "
                                 f"row {r} bar {c}: {x[r, c]:#010x} vs {y[r, c]:#010x}")


def direct_rows(lib, family, run, params, stream):
    """Each row through the family's own single-series entry point."""
    fn = getattr(lib.lib, family.d_func)
    scalars = [x for x in run.spec.scalars if x.name not in CTRL]
    fn.argtypes = ([ct.c_void_p] * (len(run.ins) + len(run.outs)) + [ct.c_int]
                   + [{"int": ct.c_int, "float": ct.c_float}[x.ctype] for x in scalars] + [ct.c_void_p])
    fn.restype = ct.c_int
    values = dict(SCALAR_DEFAULTS, **params)
    run.poison(stream)
    for row in range(run.s):
        offset = row * run.t * 4
        args = [p.value + offset for p in run.ins + run.outs] + [run.t]
        args += [values[x.name] for x in scalars] + [stream]
        st = fn(*args)
        assert st == 0, (family.d_func, st, lib.last_error())
    return run.read(stream)


def capture(cuda, run, params, stream, mode):
    """Graph of one call and its node count."""
    os.environ["TACUDA_BATCH_FUSED"] = mode
    graph = ct.c_void_p()
    cuda.check(cuda.cudaStreamBeginCapture(stream, 0), "begin capture")
    st = run.lib.fns[run.spec.c_name](*run.args(params, stream))
    end = cuda.cudaStreamEndCapture(stream, ct.byref(graph))
    assert st == 0 and end == 0, (run.spec.c_name, mode, st, end, run.lib.last_error())
    count = ct.c_size_t()
    cuda.check(cuda.cudaGraphGetNodes(graph, None, ct.byref(count)), "nodes")
    return graph, count.value


def node_count(cuda, run, params, stream, mode):
    graph, count = capture(cuda, run, params, stream, mode)
    cuda.cudaGraphDestroy(graph)
    return count


def replay(cuda, run, graph, stream):
    executable = ct.c_void_p()
    try:
        cuda.check(cuda.cudaGraphInstantiateWithFlags(ct.byref(executable), graph, 0), "instantiate")
        results = []
        for _ in range(2):
            run.poison(stream)
            cuda.check(cuda.cudaGraphLaunch(executable, stream), "replay")
            results.append(run.read(stream))
        return results
    finally:
        if executable.value:
            cuda.cudaGraphExecDestroy(executable)
        cuda.cudaGraphDestroy(graph)


def check_family(cuda, lib, spec, family, stream, pools, quick, ref=None):
    name = family.name
    capture_auto = pools or name not in BATCHED_SCRATCH
    capture_legacy = pools or name not in LEGACY_SCRATCH
    cases = batched_cases = ref_cases = 0
    shapes = SHAPES[2:4] if quick else SHAPES
    params_list = variants(name)
    for shape in shapes:
        for kind in ("walk", "nonfinite", "flat"):
            run = Run(cuda, lib, spec, data(*shape, kind))
            try:
                for params in params_list:
                    st0, err0 = run.call(params, stream, "0")
                    legacy = run.read(stream) if st0 == 0 else None
                    st1, err1 = run.call(params, stream, "auto")
                    batched = run.read(stream) if st1 == 0 else None
                    what = f"{name} {shape} {kind} {params}"
                    assert (st0, err0) == (st1, err1), f"{what}: {st0} {err0!r} vs {st1} {err1!r}"
                    if st0 == 0:
                        same_bits(batched, legacy, what)
                    if ref is not None:
                        # Another build (the release before row batching):
                        # its per-series bits are the compatibility contract.
                        st2, err2 = run.call(params, stream, "auto", ref)
                        assert st2 == st0, f"{what}: reference build status {st2} ({err2}) vs {st0}"
                        if st0 == 0:
                            same_bits(batched, run.read(stream), f"{what} vs reference build")
                        ref_cases += 1
                    cases += 1
                    # The dispatcher must pick the path the rule expects:
                    # fewer graph nodes than the per-series path, or the
                    # per-series path itself.
                    if kind == "walk" and st0 == 0 and shape[0] >= 17 and capture_auto and capture_legacy:
                        expect = expected_batched(name, params, *shape)
                        nodes = node_count(cuda, run, params, stream, "auto")
                        legacy_nodes = node_count(cuda, run, params, stream, "0")
                        if expect:
                            assert nodes < legacy_nodes, f"{what}: per-series path taken ({nodes} nodes)"
                            batched_cases += 1
                        else:
                            assert nodes == legacy_nodes, \
                                f"{what}: batched path taken ({nodes} vs {legacy_nodes} nodes)"
            finally:
                run.close()

    params = params_list[min(1, len(params_list) - 1)]
    run = Run(cuda, lib, spec, data(17, 300, "nonfinite"))
    metrics = {}
    try:
        assert expected_batched(name, params, 17, 300), f"{name}: reference case is not batched"
        # Dispatcher-independent oracle: the family's single-series entry point.
        st, err = run.call(params, stream, "auto")
        assert st == 0, (name, st, err)
        batched = run.read(stream)
        same_bits(batched, direct_rows(lib, family, run, params, stream), f"{name} vs single-series calls")

        # Same bits on the legacy default stream and through the host API.
        st, err = run.call(params, None, "auto")
        assert st == 0, (name, st, err)
        same_bits(run.read(None), batched, f"{name} stream 0")
        host = getattr(lib.lib, spec.c_name[:-2])
        host.argtypes, host.restype = lib.fns[spec.c_name].argtypes, ct.c_int
        xs = data(17, 300, "nonfinite")
        houts = [np.full((17, 300), POISON, dtype="u4") for _ in spec.outputs]
        hins = [np.ascontiguousarray(xs[COLS[n]]) for n in spec.inputs]
        os.environ["TACUDA_BATCH_FUSED"] = "auto"
        st = host(*run.args(params, stream, [a.ctypes.data for a in hins], [a.ctypes.data for a in houts]))
        assert st == 0, (name, st, lib.last_error())
        same_bits(houts, batched, f"{name} host API")

        # A fixed number of operations, captured and replayed bit for bit.
        if capture_auto:
            graph, nodes = capture(cuda, run, params, stream, "auto")
            for again in replay(cuda, run, graph, stream):
                same_bits(again, batched, f"{name} graph replay")
            wide = Run(cuda, lib, spec, data(68, 300, "walk"))
            try:
                wide_nodes = node_count(cuda, wide, params, stream, "auto")
                legacy_nodes = node_count(cuda, wide, params, stream, "0") if capture_legacy else None
            finally:
                wide.close()
            assert wide_nodes == nodes <= MAX_BATCHED_NODES, \
                f"{name}: {nodes} graph nodes at 17 rows, {wide_nodes} at 68"
            assert legacy_nodes is None or legacy_nodes >= 68, (name, legacy_nodes)
            metrics.update(graph_nodes=nodes, per_series_nodes_68_rows=legacy_nodes)

        # More rows than one flat call covers: blocks of whole rows.
        if name in BLOCK_FAMILIES:
            big = Run(cuda, lib, spec, data(*BLOCK_SHAPE, "nonfinite"))
            try:
                st0, err0 = big.call(params, stream, "0")
                legacy = big.read(stream)
                st1, err1 = big.call(params, stream, "auto")
                assert st0 == st1 == 0, (name, st0, err0, st1, err1)
                same_bits(big.read(stream), legacy, f"{name} {BLOCK_SHAPE}")
                if ref is not None:
                    st2, err2 = big.call(params, stream, "auto", ref)
                    assert st2 == 0, (name, st2, err2)
                    same_bits(big.read(stream), legacy, f"{name} {BLOCK_SHAPE} vs reference build")
                if capture_auto:
                    blocks = -(-BLOCK_SHAPE[0] // max(1, FLAT_BLOCK // BLOCK_SHAPE[1]))
                    expected = blocks * nodes if gen.ROWS[name].kind == "flat" else nodes
                    big_nodes = node_count(cuda, big, params, stream, "auto")
                    assert big_nodes == expected, f"{name}: {big_nodes} nodes for {blocks} blocks"
                    metrics["block_nodes"] = big_nodes
            finally:
                big.close()

        # Few long rows keep the per-series path; 64 rows take the flat one.
        if name in LONG_ROW_CASES and capture_auto and capture_legacy:
            for shape in LONG_ROW_CASES[name]:
                big = Run(cuda, lib, spec, data(*shape, "walk"))
                try:
                    st0, err0 = big.call(params, stream, "0")
                    legacy = big.read(stream)
                    st1, err1 = big.call(params, stream, "auto")
                    assert st0 == st1 == 0, (name, shape, st0, err0, st1, err1)
                    same_bits(big.read(stream), legacy, f"{name} {shape}")
                    expect = expected_batched(name, params, *shape)
                    nodes = node_count(cuda, big, params, stream, "auto")
                    legacy_nodes = node_count(cuda, big, params, stream, "0")
                    assert (nodes < legacy_nodes) == expect, \
                        f"{name} {shape}: {nodes} vs {legacy_nodes} nodes, batched expected: {expect}"
                finally:
                    big.close()

        # Invalid parameters: the original status and message.
        for scalar in ("period", "kPeriod", "dPeriod", "period1", "smaPeriod", "strength",
                       "signalPeriod", "lookback"):
            if scalar not in {x.name for x in spec.scalars}:
                continue
            for value in (0, -1):
                bad = dict(params, **{scalar: value})
                st0, err0 = run.call(bad, stream, "0")
                run.read(stream)
                st1, err1 = run.call(bad, stream, "auto")
                run.read(stream)
                assert (st0, err0) == (st1, err1), (name, scalar, value, st0, err0, st1, err1)
    finally:
        run.close()
    metrics.update(cases=cases, batched_path_checks=batched_cases, reference_cases=ref_cases,
                   kind=gen.ROWS[name].kind)
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--ref-lib", help="also require the bits of another build (e.g. the previous release)")
    ap.add_argument("--cudart-dir")
    ap.add_argument("--only", default="", help="comma-separated family names")
    ap.add_argument("--quick", action="store_true", help="two shapes instead of five")
    ap.add_argument("--json")
    opts = ap.parse_args()
    cuda = Cuda(find_cudart(opts.cudart_dir))
    families = {f.name: f for f in gen.FAMILIES}
    names = [n for n in opts.only.split(",") if n] or sorted(gen.ROWS)
    unknown = [n for n in names if n not in gen.ROWS]
    if unknown:
        raise SystemExit(f"no row plan: {unknown}")
    specs = {f.c_name[3:-8]: f for f in load_specs() if f.c_name[3:-8] in names}
    lib = Tacuda(str(Path(opts.lib).resolve()), list(specs.values()))
    ref = Tacuda(str(Path(opts.ref_lib).resolve()), list(specs.values())) if opts.ref_lib else None
    stream = ct.c_void_p()
    cuda.check(cuda.cudaStreamCreateWithFlags(ct.byref(stream), 1), "create stream")
    pools = cuda.graph_memory_supported()
    metrics, errors = {}, []
    try:
        for name in names:
            try:
                metrics[name] = check_family(cuda, lib, specs[name], families[name], stream, pools,
                                             opts.quick, ref)
                print(f"PASS {name}: {metrics[name]}", flush=True)
            except Exception as exc:  # report every family, then fail
                errors.append(f"{name}: {exc}")
                print(f"FAIL {name}: {exc}", flush=True)
    finally:
        os.environ.pop("TACUDA_BATCH_FUSED", None)
        cuda.cudaStreamDestroy(stream)
    report = dict(families=len(names), metrics=metrics, errors=errors, memory_pools=pools)
    if opts.json:
        Path(opts.json).write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(names) - len(errors)}/{len(names)} row-batched families bit-identical")
    if errors:
        raise SystemExit(f"{len(errors)} families failed")


if __name__ == "__main__":
    main()
