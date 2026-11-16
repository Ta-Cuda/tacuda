#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Parameter/data/shape matrix for every fused family and its fallback."""
from __future__ import annotations
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import sys
import numpy as np
import talib
from batch_graph_capture_sweep import Cuda,find_cudart,SCALAR_DEFAULTS
from batch_fused import NAMES,data,reference
from batch_indicator_corpus import Frame,Native,equal_bits,summarize


INPUTS={'d_inputs':'close','d_highs':'high','d_lows':'low','d_closes':'close','d_volumes':'volume'}
WINDOWS={'sma','wma','sum','min','max','stddev','bbands','stochastic'}
TOLERANCE={'adosc':5e-2,'sum':1.5e-4}


def supported(name,p):
    if name in WINDOWS and not 1<=p['period']<=64 and name!='stochastic': return False
    if name=='stochastic':
        return 1<=p['kPeriod']<=64 and 1<=p['dPeriod']<=64 and (2*p['dPeriod']-1)*p['kPeriod']+p['dPeriod']**2<=1024
    if name=='mfi': return 2<=p['period']<=64
    return name!='macd' or p['maType']==1


def macd_signal_one(xs,p):
    fast,slow=sorted((p['fastPeriod'],p['slowPeriod']))
    values=[]
    for row in xs['close'].astype('f8'):
        out=np.full(len(row),np.nan)
        if len(row)>=slow:
            f=float(row[slow-fast:slow].sum()/fast)
            s=float(row[:slow].sum()/slow)
            out[slow-1]=f-s
            for i in range(slow,len(row)):
                f=(row[i]-f)*(2.0/(fast+1))+f
                s=(row[i]-s)*(2.0/(slow+1))+s
                out[i]=f-s
        values.append(out)
    line=np.asarray(values,dtype='f4')
    histogram=np.where(np.isnan(line),np.nan,0).astype('f4')
    return [line,line.copy(),histogram]


def oracle(name,xs,p):
    if p['period']==1 and name in {'sma','wma','sum','min','max','stddev','bbands'}:
        if name=='stddev': return [np.where(np.isfinite(xs['close']),0,np.nan).astype('f4')]
        return [xs['close'].copy() for _ in range(3 if name=='bbands' else 1)]
    if name=='ema' and p['period']==1: return [xs['close'].copy()]
    if name=='macd' and p['signalPeriod']==1: return macd_signal_one(xs,p)
    if name=='macd' and p['maType']!=1:
        rows=[talib.MACDEXT(v.astype('f8'),p['fastPeriod'],p['maType'],p['slowPeriod'],p['maType'],p['signalPeriod'],p['maType']) for v in xs['close']]
        return [np.asarray([r[i] for r in rows],dtype='f4') for i in range(3)]
    return reference(name,xs,p)


def variants(name):
    if name in {'obv','ad'}: return [{}]
    if name=='adosc':
        return [dict(fastPeriod=f,slowPeriod=s) for f,s in [(2,2),(2,3),(3,10),(10,3),(12,26),(63,64),(65,67)]]
    if name=='macd':
        return [dict(fastPeriod=f,slowPeriod=s,signalPeriod=g,maType=m) for f,s,g,m in
                [(2,3,1,1),(2,3,2,1),(3,2,2,1),(12,26,9,1),(26,12,9,1),(12,26,1,1),(65,70,9,1),(12,26,9,0)]]
    if name=='stochastic':
        return [dict(kPeriod=k,dPeriod=d) for k,d in [(1,1),(2,1),(5,3),(14,3),(32,8),(64,1),(65,3),(14,33)]]
    periods=[1,2,3,5,14,31,63,64,65,96,128]
    if name in {'rsi','mfi'}: periods=periods[1:]
    return [dict(period=p) for p in periods]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lib',required=True)
    ap.add_argument('--cudart-dir')
    ap.add_argument('--json',required=True)
    ap.add_argument('--only',default='')
    ap.add_argument('--long',action='store_true')
    opts=ap.parse_args()
    os.environ['TACUDA_LIBRARY']=str(Path(opts.lib).resolve())
    os.environ.pop('TACUDA_BATCH_COUNT',None)
    from tacuda._generated import FUNCTIONS
    cuda=Cuda(find_cudart(opts.cudart_dir))
    native=Native(Path(opts.lib).resolve(),FUNCTIONS,cuda)
    stream=ct.c_void_p()
    cuda.check(cuda.cudaStreamCreateWithFlags(ct.byref(stream),1),'create stream')
    cells,errors=[],[]
    names=opts.only.split(',') if opts.only else NAMES

    def run(name,xs,p,mode):
        os.environ['TACUDA_BATCH_FUSED']=mode
        spec=FUNCTIONS[name+'_batch_d']
        f=Frame(native,spec,[xs[INPUTS[k]] for k in spec.inputs])
        try: return f.call(p,stream)
        finally: f.close()

    def record(label,result,ref,tol):
        assert len(result)==len(ref)
        for j,(a,b) in enumerate(zip(result,ref)):
            cells.append(dict(label,output=j,tolerance=tol,**summarize(a,b,tol)))

    try:
        for name in names:
            before=len(cells)
            base=dict(SCALAR_DEFAULTS,period=14,fastPeriod=12,slowPeriod=26,signalPeriod=9,maType=1)
            if name=='adosc': base.update(fastPeriod=3,slowPeriod=10)
            for v in variants(name):
                p=dict(base,**v)
                # Around lookbacks, window limit and series-length dispatcher boundary.
                shapes=[(1,1),(2,16),(8,31),(8,32),(8,33),(17,63),(17,64),(17,65),(67,234),(3,513)]
                if name=='stochastic':
                    minimum=p['kPeriod']+2*(p['dPeriod']-1)
                elif name=='macd':
                    minimum=max(p['fastPeriod'],p['slowPeriod'])+p['signalPeriod']-1
                elif name=='adosc':
                    minimum=max(p['fastPeriod'],p['slowPeriod'])
                elif name in {'obv','ad'}:
                    minimum=1
                else:
                    minimum=p['period']+(name in {'atr','natr','rsi','mfi'})
                shapes=list(dict.fromkeys(shapes+[(2,max(1,minimum-1)),(2,minimum),(2,minimum+1)]))
                for s,t in shapes:
                    xs=data(s,t,seed=17)
                    ref=oracle(name,xs,p)
                    for mode in ('auto','0','1'):
                        label=dict(family=name,parameters=v,shape=[s,t],mode=mode,check='oracle')
                        try:
                            out=run(name,xs,p,mode)
                            fused=supported(name,p) and mode!='0' and (name in WINDOWS or mode=='1' or t<=4*s)
                            record(label,out,ref,3e-6 if fused else TOLERANCE.get(name,1e-4))
                        except Exception as exc:
                            errors.append(dict(label,error=f'{type(exc).__name__}: {exc}'))
                # Direct graph/caller/default/host agreement for every parameter variant.
                xs=data(67,234,seed=43)
                for mode in ('auto','0','1'):
                    os.environ['TACUDA_BATCH_FUSED']=mode
                    f=Frame(native,FUNCTIONS[name+'_batch_d'],[xs[INPUTS[k]] for k in FUNCTIONS[name+'_batch_d'].inputs])
                    label=dict(family=name,parameters=v,shape=[67,234],mode=mode,check='graph/streams/host')
                    try:
                        expected=f.call(p,stream)
                        for path,st,host in [('default',None,False),('host',stream,True)]:
                            actual=f.call(p,st,host)
                            assert all(equal_bits(a,b) for a,b in zip(actual,expected)),path+' bits differ'
                        nodes,replays=f.graph(p,stream)
                        should_fuse=supported(name,p) and mode!='0'
                        assert (nodes==1)==should_fuse,('graph nodes',nodes,should_fuse)
                        assert all(all(equal_bits(a,b) for a,b in zip(replay,expected)) for replay in replays),'graph replay differs'
                        cells.append(dict(label,ok=True,graph_nodes=nodes))
                    except Exception as exc:
                        errors.append(dict(label,error=f'{type(exc).__name__}: {exc}'))
                    finally: f.close()
            # Degenerate finite inputs and separate poisoned rows.
            for fixture in ('flat','zero-volume','alternating','nan-first','nan-middle','nan-last','inf-middle'):
                xs=data(67,234,seed=71)
                if fixture=='flat':
                    for k in ('close','high','low'): xs[k].fill(100)
                if fixture=='zero-volume': xs['volume'].fill(0)
                if fixture=='alternating':
                    xs['close'][:]=np.resize(np.array([100,102,102,99,100],dtype='f4'),xs['close'].shape)
                    xs['high']=xs['close']+1;xs['low']=xs['close']-1
                p=base
                if fixture.startswith(('nan-','inf-')):
                    clean={k:v.copy() for k,v in xs.items()}
                    at=0 if fixture=='nan-first' else 233 if fixture=='nan-last' else 110
                    for a in xs.values(): a[0,at]=np.inf if fixture.startswith('inf-') else np.nan
                    for mode in ('auto','0','1'):
                        label=dict(family=name,fixture=fixture,mode=mode,check='row isolation')
                        try:
                            good=run(name,clean,p,mode);bad=run(name,xs,p,mode)
                            assert all(equal_bits(a[1:],b[1:]) for a,b in zip(good,bad)),'non-finite value leaked into another row'
                            cells.append(dict(label,ok=True))
                        except Exception as exc: errors.append(dict(label,error=f'{type(exc).__name__}: {exc}'))
                else:
                    ref=oracle(name,xs,p)
                    for mode in ('auto','0','1'):
                        label=dict(family=name,fixture=fixture,mode=mode,check='degenerate oracle')
                        try: record(label,run(name,xs,p,mode),ref,3e-6 if mode!='0' else TOLERANCE.get(name,1e-4))
                        except Exception as exc: errors.append(dict(label,error=f'{type(exc).__name__}: {exc}'))
            xs=data(67,234,seed=83)
            permutation=np.random.default_rng(41).permutation(67)
            for mode in ('auto','0','1'):
                label=dict(family=name,mode=mode,check='permutation/row/prefix')
                try:
                    out=run(name,xs,base,mode)
                    shuffled=run(name,{k:v[permutation] for k,v in xs.items()},base,mode)
                    assert all(equal_bits(a[permutation],b) for a,b in zip(out,shuffled)),'permutation changed row bits'
                    if mode=='1':
                        row=run(name,{k:v[31:32] for k,v in xs.items()},base,mode)
                        prefix=run(name,{k:np.ascontiguousarray(v[:,:117]) for k,v in xs.items()},base,mode)
                        assert all(equal_bits(a[31:32],b) for a,b in zip(out,row)),'batch size changed row bits'
                        assert all(equal_bits(a[:,:117],b) for a,b in zip(out,prefix)),'future bars changed prefix bits'
                    cells.append(dict(label,ok=True))
                except Exception as exc: errors.append(dict(label,error=f'{type(exc).__name__}: {exc}'))
            if opts.long:
                xs=data(2,1048577,seed=91)
                ref=oracle(name,xs,base)
                for mode in ('auto','0','1'):
                    label=dict(family=name,mode=mode,shape=[2,1048577],check='long oracle and repeat')
                    try:
                        out=run(name,xs,base,mode)
                        record(label,out,ref,3e-6 if mode=='1' else TOLERANCE.get(name,1e-4))
                        for _ in range(3):
                            repeat=run(name,xs,base,mode)
                            assert all(equal_bits(a,b) for a,b in zip(out,repeat)),'repeated eager output differs'
                        cells.append(dict(label,ok=True))
                    except Exception as exc: errors.append(dict(label,error=f'{type(exc).__name__}: {exc}'))
            own=cells[before:]
            print(name,sum(c['ok'] for c in own),'/',len(own),'cells',flush=True)
    finally:
        cuda.cudaStreamDestroy(stream)
    summary=dict(cells=len(cells),passed=sum(c['ok'] for c in cells),failed=sum(not c['ok'] for c in cells),errors=len(errors),families=list(names),long=opts.long)
    Path(opts.json).write_text(json.dumps(dict(summary=summary,cells=cells,errors=errors),indent=1)+'\n')
    print(json.dumps(summary),flush=True)
    return 1 if errors or summary['failed'] else 0


if __name__=='__main__':
    raise SystemExit(main())
