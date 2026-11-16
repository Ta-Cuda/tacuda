#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Row permutation and non-finite isolation for every generated batch family."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import numpy as np
from batch_indicator_corpus import Native,Frame,equal_bits
from batch_graph_capture_sweep import Cuda,find_cudart,SCALAR_DEFAULTS


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lib',required=True);ap.add_argument('--cudart-dir');ap.add_argument('--json',required=True)
    args=ap.parse_args();os.environ['TACUDA_LIBRARY']=str(Path(args.lib).resolve())
    from tacuda._generated import FUNCTIONS
    cuda=Cuda(find_cudart(args.cudart_dir));native=Native(Path(args.lib).resolve(),FUNCTIONS,cuda)
    stream=ct.c_void_p();cuda.check(cuda.cudaStreamCreateWithFlags(ct.byref(stream),1),'stream')
    rng=np.random.default_rng(93);shape=(67,234)
    c=(100+rng.normal(0,.1,shape).cumsum(axis=1)).astype('f4')
    d=dict(close=c,open=c+rng.uniform(-.2,.2,shape).astype('f4'),high=c+1,low=c-1,volume=rng.uniform(1,1000,shape).astype('f4'),close2=c+2,periods=rng.integers(2,31,shape).astype('f4'))
    mapping=dict(inputs='close',values='close',closes='close',highs='high',lows='low',opens='open',volumes='volume',xs='close',ys='close2',periods='periods')
    permutation=rng.permutation(shape[0]);cells=[]
    def run(spec,arrays,p):
        f=Frame(native,spec,arrays)
        try:return f.call(p,stream)
        finally:f.close()
    try:
        for key,spec in FUNCTIONS.items():
            if not spec.c_name.endswith('_batch_d'):continue
            name=key[:-8];arrays=[d[mapping[n.removeprefix('d_')]] for n in spec.inputs]
            for mode in ('auto','0','1'):
                os.environ['TACUDA_BATCH_FUSED']=mode
                p=dict(SCALAR_DEFAULTS)
                label=dict(family=name,mode=mode)
                try:
                    base=run(spec,arrays,p)
                    shuffled=run(spec,[a[permutation] for a in arrays],p)
                    assert all(equal_bits(a[permutation],b) for a,b in zip(base,shuffled)),'row permutation changes output bits'
                    for at in (0,110,233):
                        for value in (np.nan,np.inf):
                            bad=[a.copy() for a in arrays]
                            for k,a in enumerate(bad):
                                if spec.inputs[k]!='d_periods':a[0,at]=value
                            poisoned=run(spec,bad,p)
                            assert all(equal_bits(a[1:],b[1:]) for a,b in zip(base,poisoned)),'non-finite input crossed row boundary'
                    cells.append(dict(label,ok=True,checks=7))
                except Exception as exc:cells.append(dict(label,ok=False,error=f'{type(exc).__name__}: {exc}'))
            print(name,flush=True)
    finally:cuda.cudaStreamDestroy(stream)
    summary=dict(families=len({c['family'] for c in cells}),cases=len(cells),passed=sum(c['ok'] for c in cells),failed=sum(not c['ok'] for c in cells))
    Path(args.json).write_text(json.dumps(dict(summary=summary,cells=cells),indent=1)+'\n')
    print(json.dumps(summary),flush=True)
    return 1 if summary['failed'] else 0


if __name__=='__main__':raise SystemExit(main())
