#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Every ct_*_batch_d on a caller's stream: strict CUDA graph capture, replay
equal to the eager result bit for bit, and (optionally) the same output as a
reference build of the library.

Signatures come from bindings/python/tacuda/_generated.py, so new families
are covered without touching this file (only an unknown scalar name needs a
value in SCALAR_DEFAULTS).

    python tests/python/batch_graph_capture_sweep.py --lib build/libtacuda.so \
        [--ref-lib old/build/libtacuda.so] [--cudart-dir $CUDA_HOME/lib64]

Capture uses cudaStreamCaptureModeGlobal: besides synchronization it rejects
cudaMalloc/cudaFree and other calls that are unsafe during capture.
Exit status 0 when every family passes.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import importlib.util
import math
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "bindings" / "python" / "tacuda" / "_generated.py"

H2D, D2H = 1, 2
STREAM_NON_BLOCKING = 1
CAPTURE_MODE_GLOBAL = 0

# One valid value per scalar name used by any *_batch_d family.
SCALAR_DEFAULTS = {
    "period": 14, "period1": 7, "period2": 14, "period3": 28,
    "rsiPeriod": 14, "smaPeriod": 10, "sumPeriod": 10, "lookback": 10,
    "strength": 3, "fastPeriod": 12, "slowPeriod": 26, "signalPeriod": 9,
    "kPeriod": 14, "dPeriod": 3, "maType": 0, "dMaType": 0,
    "fastType": 1, "slowType": 1, "signalType": 1,
    "minPeriod": 2, "maxPeriod": 30,
    "initial": 0.0, "upperMul": 2.0, "lowerMul": 2.0, "step": 0.02,
    "maxAcceleration": 0.2, "maxAccel": 0.2, "accFact": 0.02, "factor": 1.0,
    "vFactor": 0.7, "fastLimit": 0.5, "slowLimit": 0.05, "startValue": 0.0,
    "offsetOnReverse": 0.0, "accInitLong": 0.02, "accLong": 0.02,
    "accMaxLong": 0.2, "accInitShort": 0.02, "accShort": 0.02,
    "accMaxShort": 0.2,
}

# Extra parameter sets for families whose internal path depends on them.
VARIANTS = {
    "ct_ma_batch_d": [{"maType": t} for t in range(9)],
    "ct_mavp_batch_d": [{"maType": 0}, {"maType": 1}],
    "ct_macdext_batch_d": [
        {"fastType": 1, "slowType": 1, "signalType": 1},   # all-EMA, device seeds
        {"fastType": 0, "slowType": 0, "signalType": 0},
        {"fastType": 2, "slowType": 1, "signalType": 0},
    ],
    "ct_kama_batch_d": [{"period": 10}, {"period": 100}],   # multi-block / single-block tail
    "ct_sma_batch_d": [{"period": 5}, {"period": 20}, {"period": 120}],
}


def load_specs():
    spec = importlib.util.spec_from_file_location("tacuda_generated_specs", GENERATED)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module
    spec.loader.exec_module(module)
    return [f for f in module.FUNCTIONS.values() if f.c_name.endswith("_batch_d")]


def find_cudart(cudart_dir: str | None):
    patterns = ["libcudart.so.12", "libcudart.so", "cudart64_12.dll", "cudart64_*.dll"]
    dirs = [d for d in (cudart_dir, os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")) if d]
    candidates = []
    for d in dirs:
        for sub in ("", "lib64", "bin", os.path.join("bin", "x64")):
            for pat in patterns:
                candidates += sorted(glob.glob(os.path.join(d, sub, pat)))
    candidates += patterns[:3]  # rely on the loader's search path
    for c in candidates:
        try:
            return ctypes.CDLL(c)
        except OSError:
            continue
    raise SystemExit("libcudart not found; pass --cudart-dir")


class Cuda:
    def __init__(self, lib):
        self.lib = lib
        vp, sz, i = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        sig = {
            "cudaMalloc": [ctypes.POINTER(vp), sz], "cudaFree": [vp],
            "cudaMemcpy": [vp, vp, sz, i], "cudaMemsetAsync": [vp, i, sz, vp],
            "cudaStreamCreateWithFlags": [ctypes.POINTER(vp), ctypes.c_uint],
            "cudaStreamDestroy": [vp], "cudaStreamSynchronize": [vp],
            "cudaStreamBeginCapture": [vp, i],
            "cudaStreamEndCapture": [vp, ctypes.POINTER(vp)],
            "cudaGraphInstantiateWithFlags": [ctypes.POINTER(vp), vp, ctypes.c_ulonglong],
            "cudaGraphLaunch": [vp, vp], "cudaGraphGetNodes": [vp, vp, ctypes.POINTER(sz)],
            "cudaGraphExecDestroy": [vp], "cudaGraphDestroy": [vp],
            "cudaGetLastError": [], "cudaGetErrorName": [i],
        }
        for name, args in sig.items():
            fn = getattr(lib, name)
            fn.argtypes = args
            fn.restype = ctypes.c_char_p if name == "cudaGetErrorName" else ctypes.c_int

    def __getattr__(self, name):
        return getattr(self.lib, name)

    def check(self, err, what):
        if err != 0:
            raise RuntimeError(f"{what}: {self.lib.cudaGetErrorName(err).decode()}")

    def name(self, err):
        return self.lib.cudaGetErrorName(err).decode()


class Tacuda:
    def __init__(self, path, specs):
        self.lib = ctypes.CDLL(path)  # RTLD_LOCAL: two builds coexist
        self.lib.ct_get_last_error.restype = ctypes.c_char_p
        self.fns = {}
        for f in specs:
            fn = getattr(self.lib, f.c_name, None)
            if fn is None:
                continue
            types = [ctypes.c_void_p] * (len(f.inputs) + len(f.outputs))
            for s in f.scalars:
                types.append({"int": ctypes.c_int, "float": ctypes.c_float}.get(s.ctype, ctypes.c_void_p))
            fn.argtypes = types
            fn.restype = ctypes.c_int
            self.fns[f.c_name] = fn

    def last_error(self):
        e = self.lib.ct_get_last_error()
        return e.decode(errors="replace") if e else ""


class Workspace:
    """Device OHLCV matrices + output matrices shared by every family."""

    def __init__(self, cuda, series, length, max_outputs):
        self.cuda, self.S, self.T = cuda, series, length
        self.count = series * length
        self.bytes = self.count * 4
        cols = {k: [] for k in ("open", "high", "low", "close", "close2", "volume", "periods")}
        for s in range(series):
            price, price2 = 50.0 + 10.0 * s, 80.0 - 3.0 * s
            for t in range(length):
                step = 0.8 * math.sin(0.31 * t + s) + 0.3 * math.cos(1.7 * t * (s + 1))
                o = price
                price += step
                price2 += 0.6 * math.cos(0.23 * t + 2 * s) + 0.25 * step
                cols["open"].append(o)
                cols["close"].append(price)
                cols["close2"].append(price2)
                cols["high"].append(max(o, price) + 0.4 + 0.2 * math.sin(0.9 * t))
                cols["low"].append(min(o, price) - 0.4 - 0.2 * math.cos(1.1 * t))
                cols["volume"].append(1000.0 + 500.0 * (1.0 + math.sin(0.13 * t + 2.0 * s)))
                cols["periods"].append(float(2 + (t * 7 + s) % 29))
        self.inputs = {k: self._upload(v) for k, v in cols.items()}
        self.outputs = [self._alloc() for _ in range(max_outputs)]

    def _alloc(self):
        p = ctypes.c_void_p()
        self.cuda.check(self.cuda.cudaMalloc(ctypes.byref(p), self.bytes), "cudaMalloc")
        return p

    def _upload(self, values):
        p = self._alloc()
        buf = struct.pack(f"{len(values)}f", *values)
        self.cuda.check(self.cuda.cudaMemcpy(p, buf, self.bytes, H2D), "cudaMemcpy H2D")
        return p

    def input_for(self, name):
        key = {
            "d_opens": "open", "d_highs": "high", "d_lows": "low", "d_closes": "close",
            "d_inputs": "close", "d_values": "close", "d_xs": "close", "d_ys": "close2",
            "d_volumes": "volume", "d_periods": "periods",
        }.get(name)
        if key is None:
            raise KeyError(f"no data for input {name}")
        return self.inputs[key]

    def poison(self, stream, n):
        for p in self.outputs[:n]:
            self.cuda.check(self.cuda.cudaMemsetAsync(p, 0x5A, self.bytes, stream), "cudaMemsetAsync")

    def download(self, n):
        out = []
        for p in self.outputs[:n]:
            buf = ctypes.create_string_buffer(self.bytes)
            self.cuda.check(self.cuda.cudaMemcpy(buf, p, self.bytes, D2H), "cudaMemcpy D2H")
            out.append(buf.raw)
        return out


def make_args(ws, f, overrides, stream):
    args = [ws.input_for(n) for n in f.inputs] + ws.outputs[: len(f.outputs)]
    for s in f.scalars:
        if s.name == "seriesLen":
            args.append(ws.T)
        elif s.name == "numSeries":
            args.append(ws.S)
        elif s.name == "maxConcurrency":
            args.append(0)
        elif s.name == "stream":
            args.append(stream)
        elif s.name in overrides:
            args.append(overrides[s.name])
        elif s.name in SCALAR_DEFAULTS:
            args.append(SCALAR_DEFAULTS[s.name])
        else:
            raise KeyError(f"{f.c_name}: no value for scalar '{s.name}'")
    return args


def eager(cuda, lib, ws, f, overrides, stream):
    ws.poison(stream, len(f.outputs))
    st = lib.fns[f.c_name](*make_args(ws, f, overrides, stream))
    err = lib.last_error()
    cuda.check(cuda.cudaStreamSynchronize(stream), f"{f.c_name} sync")
    return st, err, ws.download(len(f.outputs))


def compare(a, b):
    """(identical, differing elements, NaN-mask mismatches, max ULP distance)."""
    if a == b:
        return True, 0, 0, 0
    diff = nan_mismatch = max_ulp = 0
    for x, y in zip(a, b):
        if x == y:
            continue
        xi = struct.unpack(f"{len(x) // 4}i", x)
        yi = struct.unpack(f"{len(y) // 4}i", y)
        xf = struct.unpack(f"{len(x) // 4}f", x)
        yf = struct.unpack(f"{len(y) // 4}f", y)
        for u, v, fu, fv in zip(xi, yi, xf, yf):
            if u == v:
                continue
            if math.isnan(fu) and math.isnan(fv):
                continue  # both NaN: payload differences are not a result change
            diff += 1
            if math.isnan(fu) != math.isnan(fv):
                nan_mismatch += 1
                continue
            ou = u if u >= 0 else -(u & 0x7FFFFFFF)  # ordered integer view
            ov = v if v >= 0 else -(v & 0x7FFFFFFF)
            max_ulp = max(max_ulp, abs(ou - ov))
    return diff == 0, diff, nan_mismatch, max_ulp


def capture(cuda, lib, ws, f, overrides, stream, eager_out):
    ws.poison(stream, len(f.outputs))
    cuda.check(cuda.cudaStreamSynchronize(stream), "sync before capture")
    graph = ctypes.c_void_p()
    cuda.check(cuda.cudaStreamBeginCapture(stream, CAPTURE_MODE_GLOBAL), "begin capture")
    st = lib.fns[f.c_name](*make_args(ws, f, overrides, stream))
    err = lib.last_error()
    end = cuda.cudaStreamEndCapture(stream, ctypes.byref(graph))
    cuda.cudaGetLastError()
    if st != 0 or end != 0 or not graph.value:
        if graph.value:
            cuda.cudaGraphDestroy(graph)
        return f"capture failed: status={st} endCapture={cuda.name(end)} {err}".strip()
    nodes = ctypes.c_size_t(0)
    cuda.cudaGraphGetNodes(graph, None, ctypes.byref(nodes))
    exe = ctypes.c_void_p()
    inst = cuda.cudaGraphInstantiateWithFlags(ctypes.byref(exe), graph, 0)
    if inst != 0:
        cuda.cudaGraphDestroy(graph)
        return f"instantiate failed: {cuda.name(inst)}"
    problem = None
    for replay in range(2):
        ws.poison(stream, len(f.outputs))
        cuda.check(cuda.cudaGraphLaunch(exe, stream), "graph launch")
        cuda.check(cuda.cudaStreamSynchronize(stream), "graph sync")
        same, diff, _, ulp = compare(eager_out, ws.download(len(f.outputs)))
        if not same:
            problem = f"replay {replay} differs from eager: {diff} elements, max {ulp} ulp"
            break
    cuda.cudaGraphExecDestroy(exe)
    cuda.cudaGraphDestroy(graph)
    return problem or f"ok ({nodes.value} nodes)"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lib", required=True, help="libtacuda under test")
    ap.add_argument("--ref-lib", help="reference build for a bit-for-bit regression check")
    ap.add_argument("--cudart-dir", help="directory holding libcudart / cudart64_*.dll")
    ap.add_argument("--series", type=int, default=6)
    ap.add_argument("--len", type=int, default=300)
    ap.add_argument("--max-ulp", type=int, default=0,
                    help="allowed ULP distance from --ref-lib (default: identical)")
    ap.add_argument("--only", help="comma-separated c_name filter")
    args = ap.parse_args()

    specs = load_specs()
    if args.only:
        wanted = set(args.only.split(","))
        specs = [f for f in specs if f.c_name in wanted]
    cuda = Cuda(find_cudart(args.cudart_dir))
    lib = Tacuda(args.lib, specs)
    ref = Tacuda(args.ref_lib, specs) if args.ref_lib else None

    stream = ctypes.c_void_p()
    cuda.check(cuda.cudaStreamCreateWithFlags(ctypes.byref(stream), STREAM_NON_BLOCKING), "stream")
    ws = Workspace(cuda, args.series, args.len, max(len(f.outputs) for f in specs))

    failures, ref_diffs, cases = [], [], 0
    for f in specs:
        if f.c_name not in lib.fns:
            failures.append(f"{f.c_name}: missing from {args.lib}")
            continue
        for overrides in VARIANTS.get(f.c_name, [{}]):
            cases += 1
            label = f.c_name + (f" {overrides}" if overrides else "")
            st, err, out = eager(cuda, lib, ws, f, overrides, stream)
            if st != 0:
                failures.append(f"{label}: eager status {st} {err}")
                print(f"FAIL {label}: eager status {st} {err}")
                continue
            ref_note = ""
            if ref is not None and f.c_name in ref.fns:
                rst, rerr, rout = eager(cuda, ref, ws, f, overrides, stream)
                if rst != 0:
                    ref_note = f" | ref status {rst}"
                else:
                    same, diff, nanmm, ulp = compare(rout, out)
                    if same:
                        ref_note = " | ref identical"
                    else:
                        ref_note = f" | ref: {diff} elements differ, NaN-mask {nanmm}, max {ulp} ulp"
                        ref_diffs.append((label, diff, nanmm, ulp))
                        if nanmm or ulp > args.max_ulp:
                            failures.append(f"{label}: regression vs reference{ref_note}")
            result = capture(cuda, lib, ws, f, overrides, stream, out)
            if not result.startswith("ok"):
                failures.append(f"{label}: {result}")
            print(f"{'ok  ' if result.startswith('ok') else 'FAIL'} {label}: {result}{ref_note}")

    print(f"\n{cases} cases over {len(specs)} families; {len(failures)} failures")
    if ref is not None:
        print(f"differences from reference: {len(ref_diffs)} cases")
        for label, diff, nanmm, ulp in ref_diffs:
            print(f"  {label}: {diff} elements, NaN-mask {nanmm}, max {ulp} ulp")
    for msg in failures:
        print("  FAIL", msg)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
