#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Benchmark driver for the EXPERIMENTS strategy pack.

Runs three passes for every exp* strategy registered in tacuda-opt:

1. backtest  -- default parameters, trade list included
2. genetic   -- GA search over the full declared parameter space
3. wfo       -- 6-window walk-forward with a per-window genetic optimizer

All passes share the same synthetic daily series (--synthetic N --seed S) and
a light cost model (commission + 1 tick slippage per side) so results across
strategies are comparable.  Outputs one JSON per pass per strategy plus a
combined summary.json, all in --out DIR.

Usage (on the GPU server):
    python3 bench_experiments.py --bin ./build/tools/optimizer/tacuda-opt \
        --out exp_results --bars 5000 --seed 42 --backend tacuda
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time


def parse_strategy_list(bin_path: str) -> dict[str, list[dict[str, str]]]:
    """Parse `tacuda-opt list` output into {strategy: [param dicts]}."""
    proc = subprocess.run([bin_path, "list"], capture_output=True, text=True,
                          timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"list failed: {proc.stderr}")
    strategies: dict[str, list[dict[str, str]]] = {}
    current = None
    strat_re = re.compile(r"^  (\S+)\s*$")
    param_re = re.compile(
        r"^    (\w+)\s+default=(\S+)\s+range (\S+)\.\.(\S+) step (\S+)")
    for line in proc.stdout.splitlines():
        m = strat_re.match(line)
        if m:
            current = m.group(1)
            strategies[current] = []
            continue
        m = param_re.match(line)
        if m and current is not None:
            strategies[current].append({
                "name": m.group(1),
                "default": m.group(2),
                "lo": m.group(3),
                "hi": m.group(4),
                "step": m.group(5),
            })
    return strategies


def grid_args(params: list[dict[str, str]]) -> list[str]:
    args: list[str] = []
    for p in params:
        args += ["--grid", f"{p['name']}={p['lo']}..{p['hi']}:{p['step']}"]
    return args


def run_cmd(cmd: list[str], timeout: float) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
        return proc.returncode, (proc.stderr or "")[-400:]
    except subprocess.TimeoutExpired:
        return -9, "timeout"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bars", default="5000")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--backend", default="tacuda")
    ap.add_argument("--prefix", default="exp",
                    help="only run strategies whose name starts with this")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-command timeout, seconds")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    strategies = parse_strategy_list(args.bin)
    todo = {k: v for k, v in strategies.items() if k.startswith(args.prefix)}
    print(f"{len(todo)} strategies to benchmark", flush=True)

    common = ["--synthetic", args.bars, "--seed", args.seed,
              "--backend", args.backend,
              "--commission-bps", "2", "--slippage-ticks", "1",
              "--quiet"]

    summary: dict[str, dict] = {}
    t0 = time.time()
    for idx, (name, params) in enumerate(sorted(todo.items()), 1):
        entry: dict = {"params": params, "runs": {}}
        base = os.path.join(args.out, name)

        # 1. Backtest with defaults (trade list for equity charts).
        bt = [args.bin, "backtest", "--strategy", name, *common,
              "--json-trades", "--out-json", f"{base}.bt.json"]
        t = time.time()
        rc, err = run_cmd(bt, args.timeout)
        entry["runs"]["backtest"] = {"rc": rc, "sec": round(time.time() - t, 2)}
        if rc != 0:
            entry["runs"]["backtest"]["err"] = err

        if params:
            axes = grid_args(params)
            # 2. Genetic over the full declared space.
            ga = [args.bin, "genetic", "--strategy", name, *common, *axes,
                  "--population", "48", "--generations", "30",
                  "--stagnation", "10", "--top", "10",
                  "--fitness", "netprofit - 2*maxdd",
                  "--out-json", f"{base}.ga.json"]
            t = time.time()
            rc, err = run_cmd(ga, args.timeout)
            entry["runs"]["genetic"] = {"rc": rc, "sec": round(time.time() - t, 2)}
            if rc != 0:
                entry["runs"]["genetic"]["err"] = err

            # 3. Walk-forward with a genetic per-window optimizer.
            wfo = [args.bin, "wfo", "--strategy", name, *common, *axes,
                   "--optimizer", "genetic", "--population", "32",
                   "--generations", "15", "--stagnation", "6",
                   "--windows", "6", "--is", "0.75", "--top", "5",
                   "--fitness", "netprofit - 2*maxdd",
                   "--out-json", f"{base}.wfo.json"]
            t = time.time()
            rc, err = run_cmd(wfo, args.timeout)
            entry["runs"]["wfo"] = {"rc": rc, "sec": round(time.time() - t, 2)}
            if rc != 0:
                entry["runs"]["wfo"]["err"] = err
        else:
            entry["runs"]["genetic"] = {"rc": None, "note": "no parameters"}
            entry["runs"]["wfo"] = {"rc": None, "note": "no parameters"}

        summary[name] = entry
        status = " ".join(
            f"{k}:{'ok' if v.get('rc') == 0 else v.get('rc')}"
            for k, v in entry["runs"].items())
        print(f"[{idx}/{len(todo)}] {name}  {status}", flush=True)

    meta = {
        "bars": args.bars, "seed": args.seed, "backend": args.backend,
        "costModel": {"commissionBps": 2, "slippageTicks": 1, "tickSize": 0.01},
        "totalSec": round(time.time() - t0, 1),
        "strategies": summary,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    bad = [n for n, e in summary.items()
           if any(r.get("rc") not in (0, None) for r in e["runs"].values())]
    print(f"done in {meta['totalSec']} s; failures: {bad if bad else 'none'}",
          flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
