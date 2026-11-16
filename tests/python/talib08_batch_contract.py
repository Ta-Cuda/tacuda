#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Pinned numerical, row-isolation and graph contracts for the 0.8.3 APIs."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import sys
import numpy as np
import talib

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'benchmarks'),str(ROOT/'bindings/python')]
from talib08_catalog import ENTRIES
from talib_reference import require_reference
from talib08_corpus import data,compare,COLUMN
from batch_graph_capture_sweep import Cuda,find_cudart
from batch_indicator_corpus import Native,Frame,equal_bits

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lib',required=True);ap.add_argument('--cudart-dir');ap.add_argument('--json',required=True)
    ap.add_argument('--reference-only',action='store_true')
    args=ap.parse_args();require_reference(talib)
    os.environ['TACUDA_LIBRARY']=str(Path(args.lib).resolve())
    from tacuda._generated import FUNCTIONS
    cuda=Cuda(find_cudart(args.cudart_dir));native=Native(Path(args.lib).resolve(),FUNCTIONS,cuda)
    stream=ct.c_void_p();cuda.check(cuda.cudaStreamCreateWithFlags(ct.byref(stream),1),'create stream')
    results=[]
    try:
        for e in ENTRIES:
            if args.reference_only and not e['public_new']:continue
            if e['legacy'] and not e['public_new'] and e['name'] not in ['MA','MAVP','MACDEXT']:continue
            family=e['api'];frames=[]
            try:
                rows=[data(234,'random'),data(234,'random'),data(234,'flat')]
                for key in ['open','high','low','close']:rows[1][key]+=20
                if e['name']!='MA':
                    for key in rows[1]:
                        if key!='periods':rows[1][key][117]=np.nan
                for key in rows[0]:
                    if key!='periods':rows[0][key][:2]=np.nan
                arrays=[np.stack([r[COLUMN[n]] for r in rows]) for n in e['inputs']]
                values=[int(float(o['default'])) if t=='int' else float(o['default']) for (t,_),o in zip(e['scalars'],e['options'])]
                params=dict(zip([n for _,n in e['scalars']],values))
                main=Frame(native,FUNCTIONS[family+'_batch_d'],arrays);frames.append(main)
                baseline=main.call(params,stream)
                host=main.call(params,stream,host=True)
                assert all(equal_bits(a,b) for a,b in zip(host,baseline)),'host batch differs'
                default=main.call(params,None)
                assert all(equal_bits(a,b) for a,b in zip(default,baseline)),'default stream differs'
                for row in range(3):
                    single=Frame(native,FUNCTIONS[family+'_d'],[a[row:row+1] for a in arrays]);frames.append(single)
                    actual=single.call(params,stream)
                    assert all(equal_bits(a[0],b[row]) for a,b in zip(actual,baseline)),'row depends on series count'
                    reference=getattr(talib,e['name'])(*(a[row].astype('f8') for a in arrays),*values)
                    reference=reference if isinstance(reference,tuple) else (reference,)
                    for ours,theirs,t in zip(actual,reference,e['output_dtypes']):
                        problem=compare(ours[0],theirs,t=='int')
                        assert not problem,problem
                order=[2,0,1]
                permuted=Frame(native,FUNCTIONS[family+'_batch_d'],[a[order] for a in arrays]);frames.append(permuted)
                actual=permuted.call(params,stream)
                assert all(equal_bits(a,b[order]) for a,b in zip(actual,baseline)),'row permutation changes output'
                nodes,replays=main.graph(params,stream,replays=2)
                assert nodes>=1,'empty graph'
                assert all(equal_bits(a,b) for replay in replays for a,b in zip(replay,baseline)),'graph replay differs'
                results.append({'function':e['name'],'pass':True,'graph_nodes':nodes})
            except Exception as exc:
                results.append({'function':e['name'],'pass':False,'error':str(exc)})
            finally:
                for frame in reversed(frames):frame.close()
    finally:
        cuda.check(cuda.cudaStreamDestroy(stream),'destroy stream')
    failed=[r for r in results if not r['pass']]
    for row in failed:print('FAIL',row)
    Path(args.json).write_text(json.dumps({'families':len(results),'failed':len(failed),'results':results},indent=2)+'\n')
    print(f'{len(results)-len(failed)}/{len(results)} families passed typed host/device/batch, row and graph contracts')
    return bool(failed)

if __name__=='__main__':raise SystemExit(main())
