#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Long low-volatility series against centered CPU windows and TA-Lib.

The ordinary random-walk corpus does not expose cancellation in differences
of long prefixes. These fixtures retain realistic positive prices, while
making rolling changes small enough to expose that loss of precision.
"""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import numpy as np
import talib
from batch_indicator_corpus import Native,Frame,summarize
from batch_graph_capture_sweep import Cuda,find_cudart,SCALAR_DEFAULTS


def windows(a,period):
    return np.lib.stride_tricks.sliding_window_view(a,period,axis=1)


def references(name,data,period):
    x=data['close'].astype('f8');y=data['close2'].astype('f8')
    out=np.full_like(x,np.nan)
    if name in ('nvi','pvi'):
        volume=data['volume']
        active=volume[:,1:]<volume[:,:-1] if name=='nvi' else volume[:,1:]>volume[:,:-1]
        factors=np.ones_like(x)
        np.divide(x[:,1:],x[:,:-1],out=factors[:,1:],where=active & (x[:,:-1]!=0))
        return [1000*np.cumprod(factors,axis=1)],None
    if name=='beta':
        x=np.diff(x,axis=1)/x[:,:-1];y=np.diff(y,axis=1)/y[:,:-1]
    wx=windows(x,period);mx=wx.mean(axis=2)
    cx=wx-mx[:,:,None];vx=np.einsum('ijk,ijk->ij',cx,cx)
    if name in ('correl','beta'):
        wy=windows(y,period);cy=wy-wy.mean(axis=2)[:,:,None]
        cov=np.einsum('ijk,ijk->ij',cx,cy)
        if name=='correl':
            vy=np.einsum('ijk,ijk->ij',cy,cy)
            divisor=np.sqrt(vx*vy);begin=period-1;fn=talib.CORREL
        else:divisor=vx;begin=period;fn=talib.BETA
        out[:,begin:]=np.divide(cov,divisor,out=np.zeros_like(cov),where=divisor>0)
        ta=[np.asarray([fn(a.astype('f8'),b.astype('f8'),period) for a,b in zip(data['close'],data['close2'])])]
        return [out],ta
    variance=vx/period
    if name in ('stddev','var'):
        out[:,period-1:]=np.sqrt(variance) if name=='stddev' else variance
        fn=talib.STDDEV if name=='stddev' else talib.VAR
        ta=[np.asarray([fn(a.astype('f8'),period) for a in data['close']])]
        return [out],ta
    if name=='bbands':
        middle=np.full_like(out,np.nan);middle[:,period-1:]=mx
        deviation=np.full_like(out,np.nan);deviation[:,period-1:]=2*np.sqrt(variance)
        ta_rows=[talib.BBANDS(a.astype('f8'),period,2,2) for a in data['close']]
        return [middle+deviation,middle,middle-deviation],[np.asarray([r[j] for r in ta_rows]) for j in range(3)]
    raise KeyError(name)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lib',required=True);ap.add_argument('--cudart-dir');ap.add_argument('--json',required=True)
    ap.add_argument('--length',type=int,default=1048577)
    ap.add_argument('--only',default='correl,beta,nvi,pvi,stddev,var,bbands')
    args=ap.parse_args();os.environ['TACUDA_LIBRARY']=str(Path(args.lib).resolve());os.environ.pop('TACUDA_BATCH_COUNT',None)
    from tacuda._generated import FUNCTIONS
    cuda=Cuda(find_cudart(args.cudart_dir));native=Native(Path(args.lib).resolve(),FUNCTIONS,cuda)
    stream=ct.c_void_p();cuda.check(cuda.cudaStreamCreateWithFlags(ct.byref(stream),1),'stream')
    rng=np.random.default_rng(91);shape=(2,args.length)
    c=(100+rng.normal(0,.005,shape).cumsum(axis=1)).astype('f4')
    high=c+rng.uniform(.1,1,shape).astype('f4')
    low=c-rng.uniform(.1,1,shape).astype('f4');opened=c+rng.uniform(-.1,.1,shape).astype('f4')
    data=dict(close=c,close2=high,volume=rng.uniform(100,2000,shape).astype('f4'))
    mapping=dict(d_inputs='close',d_xs='close',d_ys='close2',d_closes='close',d_volumes='volume')
    cells,errors=[],[]
    try:
        for name in args.only.split(','):
            for period in ([14] if name in ('nvi','pvi') else [14,65]):
                stable,ta=references(name,data,period)
                spec=FUNCTIONS[name+'_batch_d'];f=Frame(native,spec,[data[mapping[n]] for n in spec.inputs])
                try:
                    for mode in ('auto','0','1'):
                        os.environ['TACUDA_BATCH_FUSED']=mode
                        label=dict(family=name,period=period,mode=mode,shape=list(shape))
                        try:
                            actual=f.call(dict(SCALAR_DEFAULTS,period=period),stream)
                            for oracle,ref in [('centered CPU / double cumulative product',stable),('TA-Lib',ta)]:
                                if ref is None:continue
                                for j,(a,b) in enumerate(zip(actual,ref)):
                                    b=b.astype('f4');finite=np.isfinite(a)&np.isfinite(b)
                                    diff=np.abs(a[finite].astype('f8')-b[finite].astype('f8'))
                                    cells.append(dict(label,oracle=oracle,output=j,max_absolute_error=float(diff.max(initial=0)),**summarize(a,b,1e-4)))
                        except Exception as exc:errors.append(dict(label,error=f'{type(exc).__name__}: {exc}'))
                finally:f.close()
                print(name,period,flush=True)
    finally:cuda.cudaStreamDestroy(stream)
    summary=dict(cells=len(cells),passed=sum(c['ok'] for c in cells),failed=sum(not c['ok'] for c in cells),errors=len(errors),length=args.length)
    Path(args.json).write_text(json.dumps(dict(summary=summary,cells=cells,errors=errors),indent=1)+'\n')
    print(json.dumps(summary),flush=True)
    return 1 if summary['failed'] or errors else 0


if __name__=='__main__':raise SystemExit(main())
