# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Regression matrix for the short-series dispatcher and its fallback."""
import argparse
import ctypes as ct
import os
from pathlib import Path
import sys

import numpy as np

from batch_indicator_corpus import Frame, Native, equal_bits
from batch_graph_capture_sweep import Cuda, find_cudart
from batch_rows import COLS, data, gen


def variants(name, spec):
    if name == 'ma':
        return [dict(period=5, maType=m) for m in (0, 1, 2, 3, 4, 5, 6, 8)]
    if name == 'kama':
        return [dict(period=10, fastPeriod=2, slowPeriod=30),
                dict(period=3, fastPeriod=3, slowPeriod=17)]
    if name == 'stochrsi':
        return [dict(rsiPeriod=14, kPeriod=5, dPeriod=3),
                dict(rsiPeriod=2, kPeriod=3, dPeriod=1)]
    if name in ('apo', 'ppo', 'pvo', 'macd_line'):
        return [{}, dict(fastPeriod=3, slowPeriod=5)]
    if name == 'hpi':
        return [dict(factor=0.18), dict(factor=1.0)]
    if name in ('cmo', 'plus_dm', 'minus_dm'):
        return [dict(period=p) for p in (14, 2, 64)]
    if any(s.name == 'period' for s in spec.scalars):
        return [dict(period=p) for p in (2, 14, 64)]
    return [{}]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lib', required=True)
    p.add_argument('--cudart-dir')
    p.add_argument('--only', default='')
    p.add_argument('--graph-only', action='store_true')
    args = p.parse_args()
    os.environ['TACUDA_LIBRARY'] = str(Path(args.lib).resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'bindings/python'))
    from tacuda._generated import FUNCTIONS

    cuda = Cuda(find_cudart(args.cudart_dir))
    native = Native(Path(args.lib).resolve(), FUNCTIONS, cuda)
    stream = ct.c_void_p()
    cuda.check(cuda.cudaStreamCreateWithFlags(ct.byref(stream), 1), 'create stream')
    failures, count = [], 0
    try:
        for name in args.only.split(',') if args.only else sorted(gen.SHORT_ROWS):
            spec = FUNCTIONS[name + '_batch_d']
            for shape in [(67, 234)] if args.graph_only else [(5, 63), (67, 234), (9, 1024)]:
                s, t = shape
                for kind in ('walk',) if args.graph_only else ('walk', 'flat', 'nonfinite'):
                    fixture = data(s, t, kind)
                    frame = Frame(native, spec, [fixture[COLS[k]] for k in spec.inputs])
                    try:
                        options = variants(name, spec)
                        for j, params in enumerate(options[:1] if args.graph_only else options):
                            try:
                                if name == 'xaverage' and params['period'] > t:
                                    for mode in ('0', '1'):
                                        os.environ['TACUDA_BATCH_FUSED'] = mode
                                        try:
                                            frame.call(params, stream)
                                        except RuntimeError as error:
                                            assert 'native status 4' in str(error), error
                                        else:
                                            raise AssertionError('period beyond the row was accepted')
                                    count += 1
                                    continue
                                os.environ['TACUDA_BATCH_FUSED'] = '1'
                                actual = frame.call(params, stream)
                                if not args.graph_only:
                                    os.environ['TACUDA_BATCH_FUSED'] = '0'
                                    reference = frame.call(params, stream)
                                    for a, b in zip(actual, reference):
                                        np.testing.assert_allclose(a, b, rtol=3e-5, atol=3e-4)
                                if j == 0 and kind == 'walk':
                                    os.environ['TACUDA_BATCH_FUSED'] = 'auto' if s >= 32 and t <= 4 * s else '1'
                                    nodes, replay = frame.graph(params, stream, replays=2)
                                    assert nodes == 1, f'expected one captured node, got {nodes}'
                                    assert all(equal_bits(a, b) for run in replay for a, b in zip(run, actual))
                                    host = frame.call(params, stream, host=True)
                                    assert all(equal_bits(a, b) for a, b in zip(host, actual))
                                if (name in ('cmo', 'plus_dm', 'minus_dm') and
                                        t == 1024 and params['period'] == 2 and kind == 'walk' and
                                        cuda.graph_memory_supported()):
                                    os.environ['TACUDA_BATCH_FUSED'] = '1'
                                    nodes, _ = frame.graph(params, stream, replays=1)
                                    assert nodes > 1, 'expected the legacy scan for fast decay'
                                count += 1
                            except Exception as e:
                                failures.append(f'{name} {s}x{t} {kind} {params}: {e}')
                    finally:
                        frame.close()
    finally:
        os.environ.pop('TACUDA_BATCH_FUSED', None)
        cuda.cudaStreamDestroy(stream)
    for message in failures:
        print('FAIL', message)
    print(f'{count} passed, {len(failures)} failed')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
