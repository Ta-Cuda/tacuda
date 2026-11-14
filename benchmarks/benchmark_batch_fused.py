#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Issue #6 reporter's 13-call pass, baseline vs fused library on identical GPU buffers.

Reports synchronized wall time (no transfers), the per-family breakdown,
and CUDA Graph replay on a non-blocking caller stream. Uses the regression
suite's ctypes loader and seeded inputs so validation and timing agree.
"""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "python"))
from batch_fused import Case, NAMES, data  # noqa: E402
from batch_graph_capture_sweep import Cuda, Tacuda, find_cudart, load_specs  # noqa: E402


def median_ms(fn, cuda, stream, reps):
    def once():
        cuda.check(cuda.cudaStreamSynchronize(stream), "pre-sync")
        start = time.perf_counter()
        fn()
        cuda.check(cuda.cudaStreamSynchronize(stream), "post-sync")
        return (time.perf_counter() - start) * 1000
    once()
    return statistics.median(once() for _ in range(reps))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--ref-lib", required=True)
    ap.add_argument("--cudart-dir")
    ap.add_argument("--shapes", default="4000x234,20000x234,100000x64,4x234,8x1000000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", required=True)
    opts = ap.parse_args()
    cuda = Cuda(find_cudart(opts.cudart_dir))
    specs = {f.c_name[3:-8]: f for f in load_specs() if f.c_name[3:-8] in NAMES[:13]}
    current = Tacuda(str(Path(opts.lib).resolve()), list(specs.values()))
    baseline = Tacuda(str(Path(opts.ref_lib).resolve()), list(specs.values()))
    stream = ct.c_void_p()
    cuda.check(cuda.cudaStreamCreateWithFlags(ct.byref(stream), 1), "stream")
    hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total", "--format=csv"], text=True).strip()
    print(hardware, flush=True)
    results = []
    try:
        for shape in opts.shapes.split(","):
            s, t = map(int, shape.split("x"))
            xs = data(s, t)
            # One shared OHLCV allocation and three output planes are enough
            # for the entire pass; each family consumes only original inputs.
            cases = []
            owner = Case(cuda, current, specs["mfi"], xs, None)
            extra = [owner.alloc(), owner.alloc()]
            ptrs = dict(zip(("high", "low", "close", "volume"), owner.ins))
            ptrs["inputs"] = ptrs["close"]
            outs = owner.outs + extra
            for name in NAMES[:13]:
                c = object.__new__(Case)
                c.cuda, c.lib, c.spec, c.stream = cuda, current, specs[name], None
                c.s, c.t = s, t
                from batch_fused import PARAMS, PARAMS_BY_NAME
                c.p = dict(PARAMS, **PARAMS_BY_NAME.get(name, {}))
                if name in ("stddev", "bbands", "sum", "sma", "min", "max"):
                    c.p["period"] = 20
                c.ins = [ptrs[n[2:].rstrip("s")] if n != "d_inputs" else ptrs["close"] for n in c.spec.inputs]
                c.outs = outs[:len(c.spec.outputs)]
                cases.append(c)
            row = dict(shape=shape, families={})
            try:
                for mode, library in (("baseline", baseline), ("auto", current)):
                    os.environ["TACUDA_BATCH_FUSED"] = "auto"
                    for c in cases:
                        c.lib, c.stream = library, None
                    def run():
                        for c in cases:
                            c.launch()
                    row[mode + "_pass_ms"] = median_ms(run, cuda, None, opts.reps)
                    if shape == "4000x234":
                        for name, c in zip(NAMES[:13], cases):
                            row["families"].setdefault(name, {})[mode + "_ms"] = median_ms(c.launch, cuda, None, opts.reps)
                # Graph timing is meaningful only if all families qualify for
                # fused execution; don't force the serial long-history path.
                if t <= 4 * s:
                    for c in cases:
                        c.lib, c.stream = current, stream
                    row["user_stream_pass_ms"] = median_ms(run, cuda, stream, opts.reps)
                    graph, executable = ct.c_void_p(), ct.c_void_p()
                    cuda.check(cuda.cudaStreamBeginCapture(stream, 0), "capture")
                    run()
                    cuda.check(cuda.cudaStreamEndCapture(stream, ct.byref(graph)), "end")
                    try:
                        count = ct.c_size_t()
                        cuda.check(cuda.cudaGraphGetNodes(graph, None, ct.byref(count)), "nodes")
                        assert count.value == 13, count.value
                        row["graph_nodes"] = count.value
                        cuda.check(cuda.cudaGraphInstantiateWithFlags(ct.byref(executable), graph, 0), "instantiate")
                        def replay():
                            cuda.check(cuda.cudaGraphLaunch(executable, stream), "launch graph")
                        row["graph_pass_ms"] = median_ms(replay, cuda, stream, max(10, opts.reps))
                    finally:
                        if executable.value:
                            cuda.cudaGraphExecDestroy(executable)
                        cuda.cudaGraphDestroy(graph)
                results.append(row)
                print(json.dumps(row), flush=True)
            finally:
                owner.close()
    finally:
        cuda.cudaStreamDestroy(stream)
    Path(opts.json).write_text(json.dumps(dict(hardware=hardware, results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
