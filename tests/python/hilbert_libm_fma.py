#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Host HT_* results must not depend on the math library's FMA dispatch.

glibc picks FMA variants of sin/cos/atan at load time on CPUs with FMA and
AVX2 (GLIBC_TUNABLES=glibc.cpu.hwcaps=-FMA,-AVX2 turns them off); the MSVC
CRT switches with _set_FMA3_enable. The DC phase of a flat or alternating
series is atan(noise/noise), so a last-bit change in one sin/cos weight moved
the host HT_DCPHASE by 24.8 degrees at bar 71 of a flat series of 100
(issue #6) while the GPU used build-time weights.

The parent runs this script as a child once per FMA setting. It requires
that the setting really changed the math library (otherwise the check proves
nothing and the test is skipped with code 77), that every host HT_* output is
bit-identical across the settings, and that the host matches the GPU batch
path of the same build in each setting.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import struct
import subprocess
import sys

import numpy as np

from batch_graph_capture_sweep import find_cudart

SKIP = 77
N_FLAT = 512
LEVELS = (0.0, 1.0, 100.0, 123.45, 1000000.0)
# name, number of outputs, host-vs-GPU tolerance (absolute; None = exact)
FAMILIES = (
    ("ht_dcperiod", 1, 1e-4),
    ("ht_dcphase", 1, 1e-4),
    ("ht_phasor", 2, 1e-4),
    ("ht_sine", 2, 1e-6),
    ("ht_trendline", 1, 1e-3),
    ("ht_trendmode", 1, None),
)
NO_FMA_TUNABLE = "glibc.cpu.hwcaps=-FMA,-AVX2,-FMA_Usable,-AVX2_Usable"


def fixtures():
    cases = []
    for level in LEVELS:
        for alternating in (False, True):
            values = np.full(N_FLAT, level, dtype=np.float32)
            if alternating:
                values += np.where(np.arange(N_FLAT) % 2 == 1, 0.1, -0.1).astype(np.float32)
            cases.append((f"level={level:g} alternating={alternating}", values))
    walk = 100.0 + np.cumsum(np.random.default_rng(7).standard_normal(2048))
    cases.append(("random walk", walk.astype(np.float32)))
    return cases


def libm_probe():
    """The DC-phase weights and a few atan values, as the math library computes them."""
    values = []
    for period in range(1, 51):
        for i in range(period):
            phase = (float(i) * 6.283185307179586) / period
            values += [math.sin(phase), math.cos(phase)]
    values += [math.atan(x / 7.0) for x in range(-50, 51)]
    return [struct.pack("<d", v).hex() for v in values]


def run_child(args):
    if args.fma == "off" and sys.platform.startswith("win"):
        crt = ctypes.CDLL("ucrtbase")
        if hasattr(crt, "_set_FMA3_enable"):
            crt._set_FMA3_enable(0)
    find_cudart(args.cudart_dir)  # loads the runtime the library links against
    lib = ctypes.CDLL(args.lib)
    lib.ct_get_last_error.restype = ctypes.c_char_p
    ptr = ctypes.POINTER(ctypes.c_float)

    def buf(a):
        return a.ctypes.data_as(ptr)

    def check(status, what):
        if status != 0:
            raise SystemExit(f"{what} failed with status {status}: {lib.ct_get_last_error()!r}")

    out = {"probe": libm_probe(), "host": {}, "device": {}}
    for label, values in fixtures():
        n = values.size
        for name, outputs, _ in FAMILIES:
            host = [np.empty(n, dtype=np.float32) for _ in range(outputs)]
            check(getattr(lib, "ct_" + name)(buf(values), *map(buf, host), ctypes.c_int(n), None),
                  f"ct_{name} ({label})")
            device = [np.empty(n, dtype=np.float32) for _ in range(outputs)]
            check(getattr(lib, f"ct_{name}_batch")(buf(values), *map(buf, device), ctypes.c_int(n),
                                                  ctypes.c_int(1), ctypes.c_int(0), None),
                  f"ct_{name}_batch ({label})")
            key = f"{name} | {label}"
            out["host"][key] = [h.tobytes().hex() for h in host]
            out["device"][key] = [d.tobytes().hex() for d in device]
    json.dump(out, sys.stdout)
    return 0


def child_result(args, fma):
    env = dict(os.environ)
    if fma == "off" and not sys.platform.startswith("win"):
        env["GLIBC_TUNABLES"] = ":".join(t for t in (env.get("GLIBC_TUNABLES"), NO_FMA_TUNABLE) if t)
    cmd = [sys.executable, os.path.abspath(__file__), "--child", "--fma", fma, "--lib", args.lib]
    if args.cudart_dir:
        cmd += ["--cudart-dir", args.cudart_dir]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"child with FMA {fma} failed ({proc.returncode}):\n{proc.stderr}")
    return json.loads(proc.stdout)


def floats(hex_text):
    return np.frombuffer(bytes.fromhex(hex_text), dtype=np.float32)


def first_difference(a, b):
    """Index of the first element whose bits differ (NaN payloads included), or -1."""
    differ = np.nonzero(a.view(np.uint32) != b.view(np.uint32))[0]
    return int(differ[0]) if differ.size else -1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lib", required=True)
    p.add_argument("--cudart-dir")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--fma", choices=("default", "off"), default="default", help=argparse.SUPPRESS)
    p.add_argument("--json", help="write the comparison summary here")
    args = p.parse_args()
    if args.child:
        return run_child(args)

    results = {fma: child_result(args, fma) for fma in ("default", "off")}
    changed = sum(a != b for a, b in zip(results["default"]["probe"], results["off"]["probe"]))
    summary = {"libm_values_changed": changed, "libm_values": len(results["default"]["probe"]),
               "host_mismatches": [], "device_mismatches": []}
    print(f"math library: the FMA setting changed {changed} of {summary['libm_values']} sin/cos/atan values")
    if changed == 0:
        print("SKIP: the FMA setting does not change the math library on this host")
        return SKIP

    for key, default_outputs in results["default"]["host"].items():
        for k, (a_hex, b_hex) in enumerate(zip(default_outputs, results["off"]["host"][key])):
            a, b = floats(a_hex), floats(b_hex)
            i = first_difference(a, b)
            if i >= 0:
                summary["host_mismatches"].append(
                    f"{key} output {k}: bar {i}: FMA {a[i]!r} vs no FMA {b[i]!r}, "
                    f"max |diff| {float(np.nanmax(np.abs(a.astype(np.float64) - b))):.6g}")

    tolerance = {name: tol for name, _, tol in FAMILIES}
    worst = {}
    for fma, result in results.items():
        for key, host_outputs in result["host"].items():
            name = key.split(" | ")[0]
            for k, (h_hex, d_hex) in enumerate(zip(host_outputs, result["device"][key])):
                h, d = floats(h_hex), floats(d_hex)
                if not np.array_equal(np.isnan(h), np.isnan(d)):
                    summary["device_mismatches"].append(f"FMA {fma}: {key} output {k}: NaN positions differ")
                    continue
                finite = ~np.isnan(h)
                gap = np.where(finite, np.abs(h.astype(np.float64) - d.astype(np.float64)), 0.0)
                diff = float(gap.max()) if gap.size else 0.0
                worst[name] = max(worst.get(name, 0.0), diff)
                tol = tolerance[name]
                if (tol is None and diff != 0.0) or (tol is not None and diff > tol):
                    i = int(np.argmax(gap))
                    summary["device_mismatches"].append(
                        f"FMA {fma}: {key} output {k}: bar {i}: host {h[i]!r} vs GPU {d[i]!r}")
    summary["host_vs_gpu_max_abs_diff"] = worst

    for name, diff in sorted(worst.items()):
        print(f"{name:13} host vs GPU max |diff| {diff:.3g}")
    for line in summary["host_mismatches"][:10]:
        print("HOST DEPENDS ON FMA:", line)
    for line in summary["device_mismatches"][:10]:
        print("HOST != GPU:", line)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    failed = summary["host_mismatches"] or summary["device_mismatches"]
    print(f"{'FAIL' if failed else 'PASS'}: {len(summary['host_mismatches'])} host outputs depend on FMA, "
          f"{len(summary['device_mismatches'])} host/GPU mismatches")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
