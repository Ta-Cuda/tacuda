# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Check NaN warmup and stable STDDEV through the public CUDA entry points."""
import argparse
import ctypes as ct
import os
from pathlib import Path
import sys

import numpy as np
import talib

from batch_indicator_corpus import Frame, Native
from batch_graph_capture_sweep import Cuda, find_cudart


def outputs(value):
    return value if isinstance(value, tuple) else (value,)


def compare(actual, expected):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, atol=5e-5, rtol=2e-5, equal_nan=True)


def rsi_reference(x):
    # TaCuda keeps the existing rule that an interior NaN poisons RSI.
    # TA-Lib 0.8.3 instead emits neutral 50 after that gap.
    valid = np.flatnonzero(~np.isnan(x))
    if valid.size:
        gaps = np.flatnonzero(np.isnan(x[valid[0]:]))
        if gaps.size:
            end = valid[0] + gaps[0]
            return np.r_[talib.RSI(x[:end], 14), np.full(x.size - end, np.nan)]
    return talib.RSI(x, 14)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lib', required=True)
    parser.add_argument('--cudart-dir')
    args = parser.parse_args()
    os.environ['TACUDA_LIBRARY'] = str(Path(args.lib).resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bindings/python'))
    import tacuda
    from tacuda._generated import FUNCTIONS

    cuda = Cuda(find_cudart(args.cudart_dir))
    native = Native(Path(args.lib).resolve(), FUNCTIONS, cuda)
    stream = ct.c_void_p()
    cuda.check(cuda.cudaStreamCreateWithFlags(ct.byref(stream), 1), 'create stream')
    defaults = dict(fastPeriod=12, slowPeriod=26, signalPeriod=9, maType=1,
                    vFactor=0.7)
    finite = (100 + np.random.default_rng(6).standard_normal(256).cumsum()).astype('f4')
    prefixed = finite.copy()
    prefixed[:14] = np.nan
    # Force a real derived input as well as variable and all-NaN rows.
    derived = tacuda.rsi(finite, 14)
    almost_empty = finite.copy()
    almost_empty[:-1] = np.nan
    rows = np.stack([prefixed, finite, np.full_like(finite, np.nan), almost_empty, derived])
    cases = [
        ('ema', (9,), lambda x: talib.EMA(x, 9)),
        ('dema', (9,), lambda x: talib.DEMA(x, 9)),
        ('tema', (9,), lambda x: talib.TEMA(x, 9)),
        ('kama', (10, 2, 30), lambda x: talib.KAMA(x, 10)),
        ('t3', (5, 0.7), lambda x: talib.T3(x, 5, 0.7)),
        ('rsi', (14,), rsi_reference),
        ('trix', (9,), lambda x: talib.TRIX(x, 9)),
        ('stddev', (20,), lambda x: talib.STDDEV(x, 20)),
        ('sum', (10,), lambda x: talib.SUM(x, 10)),
        ('macd', (12, 26, 9), lambda x: talib.MACD(x, 12, 26, 9)),
        ('apo', (12, 26), lambda x: talib.APO(x, 12, 26, 1)),
        ('apo_ma', (12, 26, 0), lambda x: talib.APO(x, 12, 26, 0)),
        ('cmo', (14,), lambda x: talib.CMO(x, 14)),
        ('max', (10,), lambda x: talib.MAX(x, 10)),
        ('min', (10,), lambda x: talib.MIN(x, 10)),
        ('midpoint', (14,), lambda x: talib.MIDPOINT(x, 14)),
        ('ht_trendline', (), talib.HT_TRENDLINE),
    ]

    def check(name, fn_args, matrix, reference, params):
        expected = [outputs(reference(x.astype('f8'))) for x in matrix]
        expected_batch = [np.stack(line) for line in zip(*expected)]
        for i, row in enumerate(matrix):
            compare(outputs(getattr(tacuda, name)(row, *fn_args)), expected[i])
            device = Frame(native, FUNCTIONS[name + '_d'], [row[None, :]])
            try:
                compare(device.call(params, stream), [x[None, :] for x in expected[i]])
            finally:
                device.close()
        batch = Frame(native, FUNCTIONS[name + '_batch_d'], [matrix])
        try:
            for mode in ('auto', '0', '1'):
                os.environ['TACUDA_BATCH_FUSED'] = mode
                for path, actual in (
                    ('host batch', batch.call(params, stream, host=True)),
                    ('device batch', batch.call(params, stream)),
                ):
                    try:
                        compare(actual, expected_batch)
                    except AssertionError as exc:
                        raise AssertionError(f'{name}, {mode}, {path}: {exc}') from exc
                if name in ('stddev', 'ema') and cuda.graph_memory_supported():
                    _, replays = batch.graph(params, stream, replays=2)
                    for replay in replays:
                        compare(replay, expected_batch)
        finally:
            batch.close()
        print(f'PASS {name} {fn_args}', flush=True)

    try:
        offset = np.array([1_000_000., 1_000_000.0625] * 128, dtype='f4')
        outlier = offset.copy()
        outlier[90] = np.float32(1e30)
        gap = offset.copy()
        gap[22:34] = np.nan
        for period in (20, 64, 65, 128, 257):
            def centered(x):
                out = np.full_like(x, np.nan)
                if x.size >= period:
                    windows = np.lib.stride_tricks.sliding_window_view(x, period)
                    out[period - 1:] = windows.std(axis=1)
                return out
            check('stddev', (period,), np.stack([offset, gap, outlier, prefixed]),
                  centered, dict(period=period))
        sum_gap = finite.copy()
        sum_gap[:5] = np.nan
        sum_gap[28:36] = np.nan
        sum_gap[50] = np.inf
        sum_gap[70] = -np.inf
        for period in (10, 65):
            def window_sum(x):
                with np.errstate(invalid='ignore'):
                    valid = np.lib.stride_tricks.sliding_window_view(x, period).sum(axis=1)
                return np.r_[np.full(period - 1, np.nan), valid]
            check('sum', (period,), np.stack([sum_gap, prefixed, finite]),
                  window_sum, dict(period=period))
        for name, fn_args, reference in cases:
            params = dict(defaults)
            if name == 'apo_ma':
                params['maType'] = 0
            elif fn_args and name not in ('macd',):
                params['period'] = fn_args[0]
            if name == 'kama':
                params.update(fastPeriod=2, slowPeriod=30)
            inputs = rows
            if name in ('ema', 'rsi', 'macd'):
                # A prefix delays initialization. A later gap still poisons the recurrence.
                middle_gap = prefixed.copy()
                middle_gap[100] = np.nan
                inputs = np.vstack([rows, middle_gap])
                compare(outputs(getattr(tacuda, name)(middle_gap, *fn_args, dtype='float64')),
                        outputs(reference(middle_gap.astype('f8'))))
            check(name, fn_args, inputs, reference, params)
    finally:
        cuda.cudaStreamDestroy(stream)


if __name__ == '__main__':
    main()
