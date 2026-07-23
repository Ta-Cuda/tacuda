#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Reproducible benchmark datasets for the tacuda benchmarking harness.

Every generator is fully determined by (name, size, seed): identical inputs
produce byte-identical float32 arrays on any platform (NumPy PCG64 stream),
so benchmark numbers from different machines and CI runs are comparable.

Usage inside a benchmark:

    from datasets import ohlcv, closes, portfolio, DATASETS

    o, h, l, c, v = ohlcv("trending", 100_000)
    matrix = portfolio("mixed", num_series=256, series_len=16_384)

CLI (writes .npz snapshots, e.g. to commit as release fixtures):

    python benchmarks/datasets.py --export datasets_v1 --size 100000
"""

from __future__ import annotations

import argparse
import os
from typing import Callable, Dict, Tuple

import numpy as np

DEFAULT_SEED = 20260705  # frozen for the 0.2.x benchmark series

Ohlcv = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _finalize(close: np.ndarray, rng: np.random.Generator) -> Ohlcv:
    """Derive open/high/low/volume around a close path (deterministic)."""
    n = close.shape[0]
    spread = np.abs(rng.normal(0.0, 0.3, n)) + 0.05
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1] + rng.normal(0.0, 0.1, n - 1)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = 1e5 + rng.integers(0, 1000, n).astype(np.float64) * 10.0
    return (open_.astype(np.float32), high.astype(np.float32),
            low.astype(np.float32), np.asarray(close, dtype=np.float32),
            volume.astype(np.float32))


def _random_walk(n: int, rng: np.random.Generator) -> np.ndarray:
    steps = rng.normal(0.0, 0.5, n)
    return np.maximum(100.0 + np.cumsum(steps), 1.0)


def _trending(n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.arange(n, dtype=np.float64)
    trend = 0.002 * x
    seasonal = 5.0 * np.sin(2.0 * np.pi * x / 1000.0)
    noise = rng.normal(0.0, 0.4, n)
    return np.maximum(100.0 + trend + seasonal + noise, 1.0)


def _mean_reverting(n: int, rng: np.random.Generator) -> np.ndarray:
    # Ornstein-Uhlenbeck around 100 (theta=0.05, sigma=0.6).
    out = np.empty(n, dtype=np.float64)
    out[0] = 100.0
    shocks = rng.normal(0.0, 0.6, n)
    for i in range(1, n):
        out[i] = out[i - 1] + 0.05 * (100.0 - out[i - 1]) + shocks[i]
    return np.maximum(out, 1.0)


def _volatile(n: int, rng: np.random.Generator) -> np.ndarray:
    # Regime-switching volatility: calm / stressed segments of ~500 bars.
    regimes = rng.integers(0, 2, max(n // 500 + 1, 1))
    sigma = np.repeat(np.where(regimes == 0, 0.2, 1.5), 500)[:n]
    return np.maximum(100.0 + np.cumsum(rng.normal(0.0, 1.0, n) * sigma), 1.0)


DATASETS: Dict[str, Callable[[int, np.random.Generator], np.ndarray]] = {
    "random_walk": _random_walk,
    "trending": _trending,
    "mean_reverting": _mean_reverting,
    "volatile": _volatile,
}


def closes(name: str, size: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Close-price series `name` of `size` bars (float32, deterministic)."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset '{name}', available: {sorted(DATASETS)}")
    rng = np.random.default_rng([seed, hash(name) & 0x7FFFFFFF, size])
    return DATASETS[name](size, rng).astype(np.float32)


def ohlcv(name: str, size: int, seed: int = DEFAULT_SEED) -> Ohlcv:
    """Full OHLCV bundle derived from the `name` close path."""
    rng = np.random.default_rng([seed, hash(name) & 0x7FFFFFFF, size, 1])
    close = closes(name, size, seed)
    return _finalize(close.astype(np.float64), rng)


def portfolio(name: str, num_series: int, series_len: int,
              seed: int = DEFAULT_SEED) -> np.ndarray:
    """(num_series, series_len) float32 matrix; row i is deterministic in i."""
    rows = [closes(name, series_len, seed + i) for i in range(num_series)]
    return np.ascontiguousarray(np.stack(rows), dtype=np.float32)


def export_snapshots(out_dir: str, size: int, seed: int = DEFAULT_SEED) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name in DATASETS:
        o, h, l, c, v = ohlcv(name, size, seed)
        path = os.path.join(out_dir, f"{name}_{size}_{seed}.npz")
        np.savez_compressed(path, open=o, high=h, low=l, close=c, volume=v)
        print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", metavar="DIR", help="write .npz snapshots")
    parser.add_argument("--size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.export:
        export_snapshots(args.export, args.size, args.seed)
    else:
        for name in DATASETS:
            c = closes(name, 16, args.seed)
            print(f"{name:<16} first4={np.round(c[:4], 3).tolist()}")
