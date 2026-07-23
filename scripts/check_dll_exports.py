#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Smoke-check native tacuda binaries for required export symbols.

Usage:
    python scripts/check_dll_exports.py [path ...]

Without arguments checks the default local build outputs. Exits non-zero if
any listed binary is missing or lacks a required export, so it can gate CI
release jobs (see .github/workflows/native-build.yml).
"""
import pathlib
import sys

DEFAULT_TARGETS = [
    r"E:\TaCuda\build-win\tacuda.dll",
    "build/libtacuda.so",
]

# One representative symbol per API generation; extend when the ABI grows.
REQUIRED_EXPORTS = [
    b"ct_sma",                # core host API
    b"ct_macdext",            # 0.1.2 MA-type MACD
    b"ct_sin_d",              # 0.1.2 vector math (device)
    b"ct_add",                # 0.1.2 vector math (host)
    b"ct_device_alloc_ex",    # dtype-aware device allocations
    b"ct_sma_ex_d",           # dtype-aware kernels
    b"ct_version",            # 0.2.0 version introspection
    b"ct_sma_batch",          # 0.2.0 batched host API
    b"ct_stochastic_batch_d", # 0.2.0 batched device API
    b"ct_wma_batch",          # 0.3.0 generated batch families (host)
    b"ct_macdext_batch",      # 0.3.0 multi-output generated batch
    b"ct_aroon_batch_d",      # 0.3.0 generated batch families (device)
    b"ct_mfi_batch",          # 0.3.0 HLCV-input generated batch
]


def check(path: pathlib.Path) -> bool:
    if not path.exists():
        print(f"{path} -> MISSING FILE")
        return False
    blob = path.read_bytes()
    ok = True
    print(path)
    for name in REQUIRED_EXPORTS:
        present = name in blob
        ok &= present
        print(f"    {name.decode():<24} {'OK' if present else 'ABSENT'}")
    return ok


def main() -> int:
    targets = [pathlib.Path(a) for a in sys.argv[1:]] or [
        pathlib.Path(t) for t in DEFAULT_TARGETS
    ]
    results = [check(t) for t in targets]
    if not any(results):
        print("FAIL: no target binary passed the export check")
        return 1
    if not all(results):
        print("FAIL: at least one target binary failed the export check")
        return 1
    print("PASS: all export checks succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
