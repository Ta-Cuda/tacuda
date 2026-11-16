#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Independent pinned-TA oracle for new APIs, typed outputs and MA modes."""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import talib

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts'),str(ROOT/'benchmarks'),str(ROOT/'bindings/python')]
from talib08_catalog import ENTRIES
from talib_reference import require_reference

COLUMN = {'inputs':'close','highs':'high','lows':'low','opens':'open',
          'closes':'close','volumes':'volume','periods':'periods'}

def data(length, pattern):
    rng = np.random.default_rng(19)
    close = 100 + rng.normal(0,.6,length).cumsum()
    if pattern=='flat': close[:]=100
    if pattern=='zeros': close=np.sin(np.arange(length)*.4);close[::17]=0
    if pattern=='negative': close=-np.abs(close)
    if pattern=='tiny': close*=1e-18
    values = dict(close=close,high=close+rng.uniform(.1,2,length),low=close-rng.uniform(.1,2,length),
                  open=close+rng.uniform(-.3,.3,length),volume=rng.uniform(10,2000,length),
                  periods=rng.integers(2,31,length))
    if pattern=='flat':
        values.update(open=close.copy(),high=close.copy(),low=close.copy(),volume=np.full(length,1000.))
    if pattern in ['leading_nan','middle_nan','end_nan']:
        at=0 if pattern=='leading_nan' else length//2 if pattern=='middle_nan' else length-1
        for name in values:
            if name!='periods': values[name][at]=np.nan
    if pattern=='all_nan':
        for name in values:
            if name!='periods': values[name][:]=np.nan
    if pattern=='staggered_nan':
        values['high'][:3]=np.nan;values['close'][:1]=np.nan;values['volume'][:2]=np.nan
    return {k:np.ascontiguousarray(v,dtype='f4') for k,v in values.items()}

def compare(actual, expected, integer=False):
    a,b=np.asarray(actual),np.asarray(expected)
    if a.shape!=b.shape:return {'shape': [list(a.shape),list(b.shape)]}
    if integer:
        if a.dtype!=np.int32:return {'dtype':str(a.dtype)}
        mismatch=np.flatnonzero(a!=b)
    else:
        if a.dtype!=np.float32:return {'dtype':str(a.dtype)}
        mismatch=np.flatnonzero(~np.isclose(a,b,rtol=2e-5,atol=2e-5,equal_nan=True))
    if len(mismatch):
        i=int(mismatch[0]);return {'count':len(mismatch),'first':i,'actual':str(a.flat[i]),'expected':str(b.flat[i])}
    return None

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lib',required=True);ap.add_argument('--json',required=True)
    ap.add_argument('--patterns',default='random,flat,zeros,negative,tiny,leading_nan,middle_nan,end_nan,all_nan,staggered_nan')
    ap.add_argument('--length',type=int,default=512);ap.add_argument('--only',default='')
    ap.add_argument('--skip-ma-matrix',action='store_true')
    args=ap.parse_args();require_reference(talib)
    os.environ['TACUDA_LIBRARY']=str(Path(args.lib).resolve())
    import tacuda
    wanted=set(args.only.split(',')) if args.only else None
    results=[]
    for e in ENTRIES:
        if e['legacy'] and not e['public_new']:continue
        if wanted and e['name'] not in wanted:continue
        params=[int(float(o['default'])) if typ=='int' else float(o['default'])
                for (typ,_),o in zip(e['scalars'],e['options'])]
        for pattern in args.patterns.split(','):
            d=data(args.length,pattern);ins=[d[COLUMN[n]] for n in e['inputs']]
            try:
                actual=getattr(tacuda,e['api'])(*ins,*params)
                reference=getattr(talib,e['name'])(*(a.astype('f8') for a in ins),*params)
                ours=actual if isinstance(actual,tuple) else (actual,)
                expected=reference if isinstance(reference,tuple) else (reference,)
                assert len(ours)==len(expected)==len(e['output_dtypes']),'output count mismatch'
                issues=[compare(a,b,t=='int') for a,b,t in zip(ours,expected,e['output_dtypes'])]
                issues=[x for x in issues if x]
                results.append({'function':e['name'],'pattern':pattern,'pass':not issues,'issues':issues})
            except Exception as exc:
                results.append({'function':e['name'],'pattern':pattern,'pass':False,'error':str(exc)})
        if not args.skip_ma_matrix:
            # Exercise real option boundaries against the independently
            # compiled oracle, including whether invalid options are refused.
            d=data(args.length,'random');ins=[d[COLUMN[n]] for n in e['inputs']]
            for index,((typ,name),option) in enumerate(zip(e['scalars'],e['options'])):
                choices=[]
                if option['minimum'] is not None:
                    minimum=float(option['minimum'])
                    choices += [('minimum',int(minimum) if typ=='int' else minimum)]
                    choices += [('below_minimum',int(minimum)-1 if typ=='int' else np.nextafter(minimum,-np.inf))]
                if option['maximum'] is not None:
                    maximum=float(option['maximum'])
                    choices += [('above_maximum',int(maximum)+1 if typ=='int' else np.nextafter(maximum,np.inf))]
                if name.lower().endswith('matype'):choices += [('invalid_matype',16)]
                for boundary,value in choices:
                    values=params.copy();values[index]=value
                    ours_error=reference_error=None;actual=reference=None
                    try:actual=getattr(tacuda,e['api'])(*ins,*values)
                    except Exception as exc:ours_error=str(exc)
                    try:reference=getattr(talib,e['name'])(*(a.astype('f8') for a in ins),*values)
                    except Exception as exc:reference_error=str(exc)
                    issues=[]
                    if bool(ours_error)!=bool(reference_error):
                        issues=[{'validation':{'actual':ours_error,'reference':reference_error}}]
                    elif not ours_error:
                        ours=actual if isinstance(actual,tuple) else (actual,)
                        expected=reference if isinstance(reference,tuple) else (reference,)
                        issues=[compare(a,b,t=='int') for a,b,t in zip(ours,expected,e['output_dtypes'])]
                        issues=[x for x in issues if x]
                    results.append({'function':e['name'],'option':name,'boundary':boundary,
                                    'pass':not issues,'issues':issues})
            positions=[i for i,(_,n) in enumerate(e['scalars']) if n.lower().endswith('matype')]
            if positions:
                d=data(args.length,'random');ins=[d[COLUMN[n]] for n in e['inputs']]
                for mode in range(16):
                    values=params.copy()
                    for i in positions:values[i]=mode
                    try:
                        actual=getattr(tacuda,e['api'])(*ins,*values)
                        reference=getattr(talib,e['name'])(*(a.astype('f8') for a in ins),*values)
                        ours=actual if isinstance(actual,tuple) else (actual,)
                        expected=reference if isinstance(reference,tuple) else (reference,)
                        issues=[compare(a,b,t=='int') for a,b,t in zip(ours,expected,e['output_dtypes'])]
                        issues=[x for x in issues if x]
                        results.append({'function':e['name'],'matype':mode,'pass':not issues,'issues':issues})
                    except Exception as exc:results.append({'function':e['name'],'matype':mode,'pass':False,'error':str(exc)})
    if not args.skip_ma_matrix:
        d=data(args.length,'random');x=d['close'];y=x.astype('f8');p=d['periods']
        for mode in range(16):
            calls=[('MA',lambda:tacuda.ma(x,14,mode),lambda:talib.MA(y,14,mode)),
                   ('MAVP',lambda:tacuda.mavp(x,p,2,30,mode),lambda:talib.MAVP(y,p.astype('f8'),2,30,mode)),
                   ('APO',lambda:tacuda.apo_ma(x,12,26,mode),lambda:talib.APO(y,12,26,mode)),
                   ('PPO',lambda:tacuda.ppo_ma(x,12,26,mode),lambda:talib.PPO(y,12,26,mode)),
                   ('MACDEXT',lambda:tacuda.macdext(x,12,mode,26,mode,9,mode),lambda:talib.MACDEXT(y,12,mode,26,mode,9,mode))]
            for name,ours,theirs in calls:
                if wanted and name not in wanted:continue
                try:
                    a,b=ours(),theirs();a=a if isinstance(a,tuple) else (a,);b=b if isinstance(b,tuple) else (b,)
                    issues=[compare(l,r) for l,r in zip(a,b)];issues=[x for x in issues if x]
                    results.append({'function':name,'matype':mode,'pass':not issues,'issues':issues})
                except Exception as exc:results.append({'function':name,'matype':mode,'pass':False,'error':str(exc)})
        if not wanted or 'MACD' in wanted:
            try:
                actual=tacuda.macd(x,12,26,9,11)
                expected=talib.MACD(y,12,26,9)
                issues=[compare(a,b) for a,b in zip(actual,expected)];issues=[x for x in issues if x]
                results.append({'function':'MACD','matype':'DEFAULT','pass':not issues,'issues':issues})
            except Exception as exc:results.append({'function':'MACD','matype':'DEFAULT','pass':False,'error':str(exc)})
    failed=[r for r in results if not r['pass']]
    Path(args.json).write_text(json.dumps({'cases':len(results),'failed':len(failed),'results':results},indent=2)+'\n')
    for r in failed:print('FAIL',r)
    print(f'{len(results)-len(failed)}/{len(results)} passed')
    return bool(failed)

if __name__=='__main__':raise SystemExit(main())
