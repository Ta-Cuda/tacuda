#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Long single-series scans: repeated eager and CUDA Graph results must agree.

The input exceeds the device-wide scan tile count that exposes scheduling-
dependent floating point addition. Run with --lib pointing to the old build
to reproduce MFI/ADOSC/STDDEV jitter. No fused-batch dispatch is involved.
"""
import argparse
import ctypes as ct
from pathlib import Path

import numpy as np

from batch_graph_capture_sweep import Cuda, Tacuda, find_cudart, load_specs
from batch_fused import Case, data, same


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--cudart-dir")
    ap.add_argument("--length", type=int, default=1048577)
    ap.add_argument("--reps", type=int, default=12)
    opts = ap.parse_args()
    cuda = Cuda(find_cudart(opts.cudart_dir))
    names = ("mfi", "adosc", "stddev", "ad")
    specs = {f.c_name[3:-8]: f for f in load_specs() if f.c_name[3:-8] in names}
    lib = Tacuda(str(Path(opts.lib).resolve()), list(specs.values()))
    stream = ct.c_void_p()
    cuda.check(cuda.cudaStreamCreateWithFlags(ct.byref(stream), 1), "create stream")
    errors = []
    try:
        xs = data(1, opts.length)
        # Fractional volumes expose inexact AD terms and MFI additions.
        for name, spec in specs.items():
            c = Case(cuda, lib, spec, xs, stream)
            fn = getattr(lib.lib, spec.c_name.replace("_batch", ""))
            params = [x for x in spec.scalars if x.name not in ("seriesLen", "numSeries", "maxConcurrency", "stream")]
            fn.argtypes = [ct.c_void_p] * (len(c.ins) + len(c.outs)) + [ct.c_int] + [ct.c_int] * len(params) + [ct.c_void_p]
            fn.restype = ct.c_int
            args = c.ins + c.outs + [c.t] + [c.p[x.name] for x in params] + [stream]
            graph, executable = ct.c_void_p(), ct.c_void_p()
            try:
                assert fn(*args) == 0
                first = c.result()
                for _ in range(opts.reps):
                    assert fn(*args) == 0
                    same(first, c.result())
                cuda.check(cuda.cudaStreamBeginCapture(stream, 0), "begin")
                assert fn(*args) == 0
                cuda.check(cuda.cudaStreamEndCapture(stream, ct.byref(graph)), "end")
                cuda.check(cuda.cudaGraphInstantiateWithFlags(ct.byref(executable), graph, 0), "instantiate")
                for _ in range(opts.reps):
                    cuda.check(cuda.cudaGraphLaunch(executable, stream), "replay")
                    same(first, c.result())
                print(f"PASS {name}: {opts.length} elements, {opts.reps} eager + {opts.reps} graph repeats", flush=True)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
                print(f"FAIL {name}: {exc}", flush=True)
            finally:
                if executable.value:
                    cuda.cudaGraphExecDestroy(executable)
                if graph.value:
                    cuda.cudaGraphDestroy(graph)
                c.close()
    finally:
        cuda.cudaStreamDestroy(stream)
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
