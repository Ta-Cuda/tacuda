#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Repeat eager and graph execution of every batch family on long series."""
from __future__ import annotations
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import numpy as np
from batch_graph_capture_sweep import Cuda,find_cudart,SCALAR_DEFAULTS
from batch_indicator_corpus import Native,Frame,equal_bits


def difference(a,b):
    count,nans,infs,max_abs=0,0,0,0.0
    for x,y in zip(a,b):
        nans+=int(np.count_nonzero(np.isnan(x)!=np.isnan(y)))
        infs+=int(np.count_nonzero((np.isinf(x)|np.isinf(y)) & (x!=y)))
        finite=np.isfinite(x)&np.isfinite(y)
        count+=int(np.count_nonzero(x[finite].view('u4')!=y[finite].view('u4')))
        if finite.any():max_abs=max(max_abs,float(np.max(np.abs(x[finite].astype('f8')-y[finite].astype('f8')))))
    return dict(differing_finite_elements=count,nan_mismatches=nans,inf_mismatches=infs,max_absolute_error=max_abs)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lib',required=True)
    ap.add_argument('--cudart-dir')
    ap.add_argument('--json',required=True)
    ap.add_argument('--length',type=int,default=1048577)
    ap.add_argument('--series',type=int,default=2)
    ap.add_argument('--sequential-length',type=int,default=16385,
                    help='Cap for HT_*, MAMA and SAR state machines; 0 disables the cap')
    ap.add_argument('--only',default='')
    ap.add_argument('--period',type=int,default=14)
    ap.add_argument('--repeats',type=int,default=4)
    ap.add_argument('--mode',default='0')
    args=ap.parse_args()
    os.environ['TACUDA_LIBRARY']=str(Path(args.lib).resolve())
    os.environ['TACUDA_BATCH_FUSED']=args.mode
    os.environ.pop('TACUDA_BATCH_COUNT',None)
    from tacuda._generated import FUNCTIONS
    cuda=Cuda(find_cudart(args.cudart_dir));native=Native(Path(args.lib).resolve(),FUNCTIONS,cuda)
    stream=ct.c_void_p();cuda.check(cuda.cudaStreamCreateWithFlags(ct.byref(stream),1),'stream')
    rng=np.random.default_rng(91)
    shape=(args.series,args.length)
    close=(100+rng.normal(0,.005,shape).cumsum(axis=1)).astype('f4')
    data=dict(close=close,high=close+rng.uniform(.1,1,shape).astype('f4'),low=close-rng.uniform(.1,1,shape).astype('f4'),
              open=close+rng.uniform(-.1,.1,shape).astype('f4'),volume=rng.uniform(100,2000,shape).astype('f4'))
    data['close2']=data['high'].copy();data['periods']=rng.integers(2,31,shape).astype('f4')
    mapping=dict(inputs='close',values='close',closes='close',highs='high',lows='low',opens='open',volumes='volume',xs='close',ys='close2',periods='periods')
    wanted=set(args.only.split(',')) if args.only else None
    results=[]
    try:
        for key,spec in FUNCTIONS.items():
            if not spec.c_name.endswith('_batch_d'):continue
            name=key[:-8]
            if wanted and name not in wanted:continue
            sequential=name.startswith('ht_') or name in {'mama','sar','sarext','sar_el'}
            length=min(args.length,args.sequential_length) if sequential and args.sequential_length else args.length
            cell=dict(family=name,mode=args.mode,period=args.period,length=length,series=args.series)
            f=None
            try:
                f=Frame(native,spec,[data[mapping[n.removeprefix('d_')]][:,:length] for n in spec.inputs])
                p=dict(SCALAR_DEFAULTS,period=args.period)
                initial=f.call(p,stream)
                diffs=[]
                for i in range(args.repeats):
                    repeat=f.call(p,stream)
                    diffs.append(dict(kind='eager',iteration=i,**difference(initial,repeat)))
                nodes,replays=f.graph(p,stream,args.repeats)
                diffs.extend(dict(kind='graph',iteration=i,**difference(initial,repeat)) for i,repeat in enumerate(replays))
                cell.update(ok=not any(x['differing_finite_elements'] or x['nan_mismatches'] or x['inf_mismatches'] for x in diffs),graph_nodes=nodes,repeats=diffs)
            except Exception as exc:
                cell.update(ok=False,error=f'{type(exc).__name__}: {exc}')
            finally:
                if f:f.close()
            results.append(cell)
            Path(args.json).write_text(json.dumps(dict(cells=results),indent=1)+'\n')
            print(name,'PASS' if cell['ok'] else 'FAIL',flush=True)
    finally:cuda.cudaStreamDestroy(stream)
    summary=dict(families=len(results),passed=sum(c['ok'] for c in results),failed=sum(not c['ok'] for c in results),length=args.length,sequential_length=args.sequential_length,series=args.series,period=args.period,mode=args.mode)
    Path(args.json).write_text(json.dumps(dict(summary=summary,cells=results),indent=1)+'\n')
    print(json.dumps(summary),flush=True)
    return 1 if summary['failed'] else 0


if __name__=='__main__':raise SystemExit(main())
