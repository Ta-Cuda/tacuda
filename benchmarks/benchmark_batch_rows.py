#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Row-batched legacy families: per-series dispatcher vs whole-matrix execution.

Synchronized wall time of one *_batch_d call on GPU-resident float32
matrices (no transfers), in one library build: TACUDA_BATCH_FUSED=0 (the
per-series dispatcher) against the default dispatcher, both on stream 0 with
the default pool. The two paths return the same bits
(tests/python/batch_rows.py), so the difference is dispatch alone.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "python"))
from batch_rows import Run, data, gen  # noqa: E402
from batch_graph_capture_sweep import Cuda, Tacuda, find_cudart, load_specs  # noqa: E402


def median_ms(fn, cuda, reps):
    def once():
        cuda.check(cuda.cudaStreamSynchronize(None), "pre-sync")
        start = time.perf_counter()
        fn()
        cuda.check(cuda.cudaStreamSynchronize(None), "post-sync")
        return (time.perf_counter() - start) * 1000
    once()
    return statistics.median(once() for _ in range(reps))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--cudart-dir")
    ap.add_argument("--shapes", default="4000x234,1000x2000,100x20000,8x250000")
    ap.add_argument("--only", default="", help="comma-separated family names")
    ap.add_argument("--short", action="store_true", help="measure the short-series path")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--json", required=True)
    opts = ap.parse_args()
    cuda = Cuda(find_cudart(opts.cudart_dir))
    names = [n for n in opts.only.split(",") if n] or sorted(gen.SHORT_ROWS if opts.short else gen.ROWS)
    specs = {f.c_name[3:-8]: f for f in load_specs() if f.c_name[3:-8] in names}
    lib = Tacuda(str(Path(opts.lib).resolve()), list(specs.values()))
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader"], text=True).strip()
    print(hardware, flush=True)
    results = []
    try:
        for shape in opts.shapes.split(","):
            s, t = map(int, shape.split("x"))
            xs = data(s, t, "walk")
            totals = {"0": 0.0, "auto": 0.0}
            for name in names:
                run = Run(cuda, lib, specs[name], xs)
                try:
                    row = dict(shape=shape, family=name,
                               kind="short" if name in gen.SHORT_ROWS else gen.ROWS[name].kind)
                    for mode in ("0", "auto"):
                        os.environ["TACUDA_BATCH_FUSED"] = mode
                        native_call = lib.fns[specs[name].c_name]
                        native_args = run.args({}, None)
                        def call():
                            if row["kind"] == "short":
                                st = native_call(*native_args)
                                assert st == 0, (name, mode, st, lib.last_error())
                            else:
                                st, err = run.call({}, None, mode)
                                assert st == 0, (name, mode, st, err)
                        row[f"mode_{mode}_ms"] = median_ms(call, cuda, opts.reps)
                        totals[mode] += row[f"mode_{mode}_ms"]
                    row["speedup"] = row["mode_0_ms"] / row["mode_auto_ms"]
                    results.append(row)
                    print(f"{shape:>11} {name:<26} {row['kind']:<4} per-series {row['mode_0_ms']:9.3f} ms"
                          f"  batched {row['mode_auto_ms']:8.3f} ms  x{row['speedup']:.1f}", flush=True)
                finally:
                    run.close()
            summary = dict(shape=shape, family="TOTAL", kind="", mode_0_ms=totals["0"],
                           mode_auto_ms=totals["auto"], speedup=totals["0"] / totals["auto"])
            results.append(summary)
            print(f"{shape:>11} TOTAL ({len(names)} families) per-series {totals['0']:.1f} ms"
                  f"  batched {totals['auto']:.1f} ms  x{summary['speedup']:.1f}", flush=True)
    finally:
        os.environ.pop("TACUDA_BATCH_FUSED", None)
    Path(opts.json).write_text(json.dumps(dict(hardware=hardware, results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
