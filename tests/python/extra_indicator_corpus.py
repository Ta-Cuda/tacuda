#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Independent CPU-formula checks for all 23 non-catalog batch families."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import sys
import numpy as np
from batch_indicator_corpus import Native,Frame,summarize
from batch_graph_capture_sweep import Cuda,find_cudart,SCALAR_DEFAULTS
from indicator_extra_references import reference

NAMES=('asi','cdl_bearish_engulfing','cdl_bullish_engulfing','change','hpi','kurtosis',
       'market_strength','nvi','obtr','pivot_high','pivot_low','pvi','pvo','refdev',
       'rolling_max','rolling_mean','rolling_min','rolling_std','sar_el','skew',
       'trendscore','wobv','xaverage')
MAPPING=dict(inputs='close',values='close',closes='close',highs='high',lows='low',
             opens='open',volumes='volume',xs='close',ys='close2',periods='periods')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lib',required=True);ap.add_argument('--cudart-dir');ap.add_argument('--json',required=True)
    args=ap.parse_args()
    os.environ['TACUDA_LIBRARY']=str(Path(args.lib).resolve())
    from tacuda._generated import FUNCTIONS
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'benchmarks'))
    from differential_corpus import build_corpus
    corpus=build_corpus(256)
    cuda=Cuda(find_cudart(args.cudart_dir));native=Native(Path(args.lib).resolve(),FUNCTIONS,cuda)
    stream=ct.c_void_p();cuda.check(cuda.cudaStreamCreateWithFlags(ct.byref(stream),1),'stream')
    cells,errors=[],[]
    try:
        for name in NAMES:
            start=len(cells)
            for ds,source in corpus.items():
                # Three different rows, calculated independently by the CPU oracle.
                rows=[]
                for i in range(3):
                    d={k:np.asarray(v,dtype='f4').copy() for k,v in source.items()}
                    for k in ('close','open','high','low','close2'):
                        d[k]+=np.float32(i*3)
                    d['volume']*=np.float32(1+.1*i)
                    rows.append(d)
                p=dict(SCALAR_DEFAULTS)
                if name=='hpi':p['factor']=.1
                ref=np.asarray([reference(name,d,p) for d in rows],dtype='f4')
                spec=FUNCTIONS[name+'_batch_d']
                arrays=[np.stack([d[MAPPING[k.removeprefix('d_')]] for d in rows]) for k in spec.inputs]
                f=Frame(native,spec,arrays)
                try:
                    for mode in ('auto','0','1'):
                        os.environ['TACUDA_BATCH_FUSED']=mode
                        for path,st,host in [('device-caller',stream,False),('device-default',None,False),('host-batch',stream,True)]:
                            label=dict(family=name,dataset=ds,mode=mode,path=path,oracle='independent CPU formula')
                            try:
                                result=f.call(p,st,host)
                                assert len(result)==1
                                cells.append(dict(label,**summarize(result[0],ref,1e-4)))
                            except Exception as exc:errors.append(dict(label,error=f'{type(exc).__name__}: {exc}'))
                finally:f.close()
            own=cells[start:]
            print(name,sum(c['ok'] for c in own),'/',len(own),'formula cells',flush=True)
    finally:cuda.cudaStreamDestroy(stream)
    summary=dict(families=len(NAMES),cells=len(cells),passed=sum(c['ok'] for c in cells),failed=sum(not c['ok'] for c in cells),errors=len(errors))
    Path(args.json).write_text(json.dumps(dict(summary=summary,cells=cells,errors=errors),indent=1)+'\n')
    print(json.dumps(summary),flush=True)
    return 1 if errors or summary['failed'] else 0


if __name__=='__main__':raise SystemExit(main())
