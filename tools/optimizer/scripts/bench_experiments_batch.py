#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Batch-API benchmark for the EXPERIMENTS strategy pack.

Runs exp* strategies through `tacuda-opt portfolio` on the same synthetic
multi-symbol data set with four backend configurations:

  tacuda         one ct_*_batch_d call per indicator request over a
                 numSeries x seriesLen device matrix (the batch API)
  tacuda-device  one single-series ct_*_d call per symbol per request
                 (the device-API baseline the batch API is compared to)
  talib          per-symbol TA-Lib on the CPU, one thread
  talib-mt       per-symbol TA-Lib on the CPU, all hardware cores
                 (--threads 0; the honest CPU baseline)

Signals and simulation are identical CPU work in every run, so the JSON
`timing.indicatorSec` field isolates the part the backends actually own.
Net profit must agree across backends: the GPU paths to float determinism
limits (scan-based kernels — RSI/BBANDS/EMA/ATR/club scans — are
run-to-run nondeterministic at the ULP level because the CUB decoupled
look-back grouping depends on scheduling, and a knife-edge signal
threshold can flip a single trade on long series), TA-Lib within a wider
tolerance for its float conventions.

Usage (on the GPU server):
    python3 bench_experiments_batch.py --bin ./build/tools/optimizer/tacuda-opt \
        --out exp_batch_results --bars 500000 --symbols 16
    # scaling subset:
    python3 bench_experiments_batch.py --bin ... --out scaling_64 \
        --bars 100000 --symbols 64 --strategies exp03_obtr,exp06_asi,...
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

# label -> (--backend value, extra CLI args)
RUNS: list[tuple[str, str, list[str]]] = [
    ("tacuda", "tacuda", []),
    ("tacuda-device", "tacuda-device", []),
    ("talib", "talib", ["--threads", "1"]),
    ("talib-mt", "talib", ["--threads", "0"]),
]
LABELS = [label for label, _, _ in RUNS]


def strategy_names(binary: str) -> list[str]:
    out = subprocess.run([binary, "list"], capture_output=True, text=True,
                         check=True).stdout
    names = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("exp") and " " not in line:
            names.append(line)
    return sorted(names)


def run_one(binary: str, strategy: str, backend: str, extra: list[str], args,
            json_path: str):
    cmd = [binary, "portfolio",
           "--synthetic", str(args.bars), "--symbols", str(args.symbols),
           "--seed", str(args.seed), "--strategy", strategy,
           "--backend", backend, *extra,
           "--commission-bps", "2", "--slippage-ticks", "1",
           "--quiet", "--out-json", json_path]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=args.timeout)
    wall = time.time() - t0
    if proc.returncode != 0:
        return {"rc": proc.returncode, "wallSec": round(wall, 3),
                "err": (proc.stderr or proc.stdout)[-400:]}
    with open(json_path, encoding="utf-8") as fh:
        rep = json.load(fh)
    return {
        "rc": 0,
        "wallSec": round(wall, 3),
        "indicatorSec": rep["timing"]["indicatorSec"],
        "signalSec": rep["timing"]["signalSec"],
        "netProfit": rep["metrics"]["netProfit"],
        "trades": rep["metrics"]["numTrades"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bars", type=int, default=500000)
    ap.add_argument("--symbols", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="exp")
    ap.add_argument("--strategies", default="",
                    help="comma-separated subset (overrides --prefix)")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.strategies:
        todo = [s.strip() for s in args.strategies.split(",") if s.strip()]
    else:
        todo = [n for n in strategy_names(args.bin) if n.startswith(args.prefix)]
    print(f"{len(todo)} strategies x {len(RUNS)} backend configs, "
          f"{args.symbols} symbols x {args.bars} bars", flush=True)

    results: dict[str, dict] = {}
    t0 = time.time()
    for idx, name in enumerate(todo, 1):
        entry: dict = {}
        for label, backend, extra in RUNS:
            jp = os.path.join(args.out, f"{name}.{label}.json")
            entry[label] = run_one(args.bin, name, backend, extra, args, jp)
        results[name] = entry

        # Cross-backend parity on net profit. The two GPU paths run the same
        # kernels, but scan-based ones are only ULP-reproducible, so allow a
        # 0.1% band; TA-Lib gets a wider 5% band for convention differences.
        ok = all(entry[b].get("rc") == 0 for b in LABELS)
        parity = "?"
        if ok:
            np_batch = entry["tacuda"]["netProfit"]
            np_dev = entry["tacuda-device"]["netProfit"]
            np_talib = entry["talib"]["netProfit"]
            scale = max(1.0, abs(np_batch))
            gpu_close = abs(np_batch - np_dev) <= 1e-3 * scale
            talib_close = abs(np_batch - np_talib) <= 0.05 * scale + 1e-6
            parity = ("gpu=" + ("ok" if gpu_close else "MISMATCH") +
                      " talib=" + ("ok" if talib_close else "DIFF"))
            entry["parity"] = {"gpuClose": gpu_close, "talibClose": talib_close}
        status = " ".join(
            f"{b}:{'ok' if entry[b].get('rc') == 0 else entry[b].get('rc')}"
            for b in LABELS)
        print(f"[{idx}/{len(todo)}] {name}  {status}  {parity}", flush=True)

    # Aggregate speedups over strategies where all runs succeeded.
    ind = {b: [] for b in LABELS}
    for entry in results.values():
        if all(entry[b].get("rc") == 0 for b in LABELS):
            for b in LABELS:
                ind[b].append(entry[b]["indicatorSec"])
    agg = {}
    if ind["tacuda"]:
        agg = {
            "strategies": len(ind["tacuda"]),
            "indicatorSecTotal": {b: round(sum(ind[b]), 3) for b in LABELS},
            "indicatorSecMedian": {b: round(statistics.median(ind[b]), 4)
                                   for b in LABELS},
            "speedupBatchVsDevice": round(sum(ind["tacuda-device"]) /
                                          sum(ind["tacuda"]), 2),
            "speedupBatchVsTalib": round(sum(ind["talib"]) /
                                         sum(ind["tacuda"]), 2),
            "speedupBatchVsTalibMt": round(sum(ind["talib-mt"]) /
                                           sum(ind["tacuda"]), 2),
        }

    meta = {
        "bars": args.bars, "symbols": args.symbols, "seed": args.seed,
        "costModel": {"commissionBps": 2, "slippageTicks": 1, "tickSize": 0.01},
        "totalSec": round(time.time() - t0, 1),
        "aggregate": agg,
        "strategies": results,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)

    bad = [n for n, e in results.items()
           if any(e[b].get("rc") not in (0,) for b in LABELS)]
    mism = [n for n, e in results.items()
            if e.get("parity") and not e["parity"]["gpuClose"]]
    print(f"done in {meta['totalSec']} s; failures: {bad if bad else 'none'}; "
          f"gpu parity mismatches: {mism if mism else 'none'}", flush=True)
    if agg:
        t = agg["indicatorSecTotal"]
        print(f"indicator phase totals: batch {t['tacuda']} s, "
              f"device-loop {t['tacuda-device']} s, "
              f"talib-1t {t['talib']} s, talib-mt {t['talib-mt']} s "
              f"(batch vs device x{agg['speedupBatchVsDevice']}, "
              f"vs talib-1t x{agg['speedupBatchVsTalib']}, "
              f"vs talib-mt x{agg['speedupBatchVsTalibMt']})", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
