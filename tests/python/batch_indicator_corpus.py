#!/usr/bin/env python3
# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""Compare the complete batch/device surface with the existing strict oracle.

Uses the established corpus tolerances without widening them. Library extras
without a TA-Lib counterpart are checked against isolated single-series host
calls; these are dispatcher checks, not independent mathematical oracles.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import talib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks'))
from batch_graph_capture_sweep import Cuda, find_cudart, SCALAR_DEFAULTS
from talib_compare_helpers import build_specs, compare_arrays, compare_discrete

GUARD=16
CANARY=np.uint32(0x5a5a5a5a)


def equal_bits(a,b):
    return (a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b))
            and np.array_equal(a[~np.isnan(a)].view('u4'),b[~np.isnan(b)].view('u4')))


class Recorder:
    def __init__(self,specs):
        self.specs,self.calls,self.result=specs,[],None

    def __getattr__(self,name):
        if name not in self.specs:
            raise AttributeError(name)
        def call(*args,**kwargs):
            self.calls.append((name,args,kwargs))
            if self.result is not None:
                return self.result
            n=len(self.specs[name].outputs)
            out=tuple(np.full_like(args[0],np.nan,dtype='f4') for _ in range(n))
            return out[0] if n==1 else out
        return call


def recorded_call(recorder,spec,data):
    recorder.calls=[]
    recorder.result=None
    spec.tacuda_fn(data)
    assert len(recorder.calls)==1,(spec.name,len(recorder.calls))
    name,args,kwargs=recorder.calls[0]
    original=recorder.specs[name]
    arrays=[np.ascontiguousarray(v,dtype='f4') for v in args[:len(original.inputs)]]
    scalar_specs=[v for v in original.scalars if v.name!='stream']
    scalars=dict(zip((v.name for v in scalar_specs),args[len(arrays):]))
    scalars.update({k:v for k,v in kwargs.items() if k!='dtype'})
    aliases={'type':'maType'}
    if name=='ultosc':
        aliases.update(shortPeriod='period1',mediumPeriod='period2',longPeriod='period3')
    else:
        aliases.update(shortPeriod='fastPeriod',longPeriod='slowPeriod')
    scalars={aliases.get(k,k):v for k,v in scalars.items()}
    if name=='aroon':
        assert scalars['upPeriod']==scalars['downPeriod'],'batch AROON has one shared period'
        scalars['period']=scalars.pop('upPeriod')
        scalars.pop('downPeriod')
    return name,arrays,scalars


def formatted(recorder,spec,data,raw):
    recorder.result=raw[0] if len(raw)==1 else tuple(raw)
    try:
        value=spec.tacuda_fn(data)
        return value if isinstance(value,tuple) else (value,)
    finally:
        recorder.result=None


class Native:
    def __init__(self,path,specs,cuda):
        self.lib,self.specs,self.cuda=ct.CDLL(str(path)),specs,cuda
        self.lib.ct_get_last_error.restype=ct.c_char_p

    def function(self,spec):
        fn=getattr(self.lib,spec.c_name)
        types=[ct.c_void_p]*(len(spec.inputs)+len(spec.outputs))
        if spec.size_param:
            types.append(ct.c_int)
        types += [ct.c_void_p if v.ctype=='cudaStream_t' else ct.c_float if v.ctype=='float' else ct.c_double if v.ctype=='double' else ct.c_int for v in spec.scalars]
        fn.argtypes,fn.restype=types,ct.c_int
        return fn

    def check(self,status,label):
        if status:
            msg=self.lib.ct_get_last_error()
            raise RuntimeError(f'{label}: native status {status}: {msg.decode() if msg else ""}')


class Frame:
    def __init__(self,native,spec,arrays):
        self.native,self.cuda,self.spec=native,native.cuda,spec
        self.arrays=[np.ascontiguousarray(x,dtype='f4') for x in arrays]
        self.s,self.t=self.arrays[0].shape
        self.count=self.s*self.t
        self.ptrs=[]
        self.device_inputs=[self.allocate(x) for x in self.arrays]
        self.device_outputs=[self.allocate() for _ in spec.outputs]
        # Pageable H2D may return after staging, before its default-stream
        # DMA completes. Finish initialization before using a nonblocking
        # caller stream, including the final output-canary upload.
        self.cuda.check(self.cuda.cudaStreamSynchronize(None),'finish default-stream buffer initialization')

    def allocate(self,values=None):
        host=np.full(self.count+2*GUARD,CANARY,dtype='u4')
        if values is not None:
            host[GUARD:-GUARD]=values.reshape(-1).view('u4')
        base=ct.c_void_p()
        self.cuda.check(self.cuda.cudaMalloc(ct.byref(base),host.nbytes),'allocate guarded buffer')
        self.cuda.check(self.cuda.cudaMemcpy(base,host.ctypes.data,host.nbytes,1),'upload guarded buffer')
        self.ptrs.append(base)
        return ct.c_void_p(base.value+GUARD*4)

    def scalars(self,spec,params,stream):
        values=dict(SCALAR_DEFAULTS,**params)
        for prefix in ['fast','slow','signal']:
            if prefix+'MAType' in params: values[prefix+'Type']=params[prefix+'MAType']
        values.update(seriesLen=self.t,numSeries=self.s,maxConcurrency=4,stream=stream)
        if 'type' in values: values['maType']=values['type']
        if 'upPeriod' in values:
            assert values['upPeriod']==values['downPeriod']
            values['period']=values['upPeriod']
        return [values[v.name] for v in spec.scalars]

    def result(self):
        result=[]
        for index,p in enumerate(self.device_outputs):
            a=np.empty(self.count+2*GUARD,dtype='u4')
            self.cuda.check(self.cuda.cudaMemcpy(a.ctypes.data,p.value-GUARD*4,a.nbytes,2),'download guarded output')
            assert np.all(a[:GUARD]==CANARY) and np.all(a[-GUARD:]==CANARY),'output guard overwritten'
            missing=np.flatnonzero(a[GUARD:-GUARD]==CANARY)
            assert not missing.size,f'{self.spec.c_name}: {missing.size} output elements retain poison, first={missing[:8].tolist()}, shape={self.s}x{self.t}'
            element = getattr(self.spec,'output_dtypes',())
            dtype = 'i4' if element and element[index]=='int' else 'f4'
            result.append(a[GUARD:-GUARD].view(dtype).reshape(self.s,self.t).copy())
        return result

    def call(self,params,stream=None,host=False):
        spec=self.spec
        if host:
            spec=self.native.specs[spec.c_name[3:-2]]
            outs=[np.full(self.count+2*GUARD,CANARY,dtype='u4') for _ in spec.outputs]
            pointers=[x.ctypes.data for x in self.arrays]+[x.ctypes.data+GUARD*4 for x in outs]
        else:
            pointers=self.device_inputs+self.device_outputs
            for p in self.device_outputs:
                self.cuda.check(self.cuda.cudaMemsetAsync(p,0x5a,self.count*4,stream),'poison output')
        args=pointers+([self.count] if spec.size_param else [])+self.scalars(spec,params,stream)
        self.native.check(self.native.function(spec)(*args),spec.c_name)
        self.cuda.check(self.cuda.cudaStreamSynchronize(stream),'synchronize result')
        if not host:
            return self.result()
        for a in outs:
            assert np.all(a[:GUARD]==CANARY) and np.all(a[-GUARD:]==CANARY),'host output guard overwritten'
            assert not np.any(a[GUARD:-GUARD]==CANARY),'host output element not initialized'
        elements = getattr(spec,'output_dtypes',()) or ('float',)*len(outs)
        return [a[GUARD:-GUARD].view('i4' if t=='int' else 'f4').reshape(self.s,self.t).copy() for a,t in zip(outs,elements)]

    def close(self):
        problems=[]
        for p,a in zip(self.device_inputs,self.arrays):
            out=np.empty(self.count+2*GUARD,dtype='u4')
            self.cuda.check(self.cuda.cudaMemcpy(out.ctypes.data,p.value-GUARD*4,out.nbytes,2),'check immutable input')
            if not (np.all(out[:GUARD]==CANARY) and np.all(out[-GUARD:]==CANARY)):
                problems.append('input guard overwritten')
            if not np.array_equal(out[GUARD:-GUARD],a.reshape(-1).view('u4')):
                problems.append('input overwritten')
        for base in reversed(self.ptrs):
            self.cuda.check(self.cuda.cudaFree(base),'free guarded buffer')
        assert not problems,', '.join(problems)

    def graph(self,params,stream,replays=3):
        graph,executable=ct.c_void_p(),ct.c_void_p()
        self.cuda.check(self.cuda.cudaStreamSynchronize(stream),'sync before capture')
        self.cuda.check(self.cuda.cudaStreamBeginCapture(stream,0),'begin strict capture')
        args=self.device_inputs+self.device_outputs+([self.count] if self.spec.size_param else [])+self.scalars(self.spec,params,stream)
        status=self.native.function(self.spec)(*args)
        end=self.cuda.cudaStreamEndCapture(stream,ct.byref(graph))
        try:
            self.native.check(status,self.spec.c_name)
            self.cuda.check(end,'end strict capture')
            count=ct.c_size_t()
            self.cuda.check(self.cuda.cudaGraphGetNodes(graph,None,ct.byref(count)),'graph nodes')
            self.cuda.check(self.cuda.cudaGraphInstantiateWithFlags(ct.byref(executable),graph,0),'instantiate graph')
            out=[]
            for _ in range(replays):
                for p in self.device_outputs:
                    self.cuda.check(self.cuda.cudaMemsetAsync(p,0x5a,self.count*4,stream),'poison replay output')
                self.cuda.check(self.cuda.cudaGraphLaunch(executable,stream),'graph replay')
                self.cuda.check(self.cuda.cudaStreamSynchronize(stream),'graph replay sync')
                out.append(self.result())
            return count.value,out
        finally:
            if executable.value: self.cuda.cudaGraphExecDestroy(executable)
            if graph.value: self.cuda.cudaGraphDestroy(graph)


def summarize(a,b,tolerance,discrete=False):
    a,b=np.asarray(a),np.asarray(b)
    oracle=compare_discrete if discrete else compare_arrays
    ok,md,_,nm=oracle(a,b,tolerance)
    finite=np.isfinite(a)&np.isfinite(b)
    diff=int(np.count_nonzero(a[finite].astype('f4').view('u4')!=b[finite].astype('f4').view('u4')))
    return dict(ok=bool(ok),max_scaled_error=float(md),nan_mismatch=int(nm),finite_bit_differences=diff)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--lib',required=True)
    ap.add_argument('--cudart-dir')
    ap.add_argument('--json',required=True)
    ap.add_argument('--size',type=int,default=512)
    ap.add_argument('--series',type=int,default=8)
    ap.add_argument('--modes',default='auto,0,1')
    ap.add_argument('--only',default='')
    opts=ap.parse_args()
    os.environ.pop('TACUDA_BATCH_COUNT',None)
    os.environ['TACUDA_LIBRARY']=str(Path(opts.lib).resolve())
    import tacuda
    from tacuda._generated import FUNCTIONS
    from differential_corpus import build_corpus
    modes=opts.modes.split(',')
    cuda=Cuda(find_cudart(opts.cudart_dir))
    native=Native(Path(opts.lib).resolve(),FUNCTIONS,cuda)
    stream=ct.c_void_p()
    cuda.check(cuda.cudaStreamCreateWithFlags(ct.byref(stream),1),'create caller stream')
    recorder=Recorder(FUNCTIONS)
    specs=build_specs(recorder,talib)
    corpus=build_corpus(opts.size)
    rows,errors,skips,covered=[],[],[],set()
    batch_names={k for k,v in FUNCTIONS.items() if v.c_name.endswith('_batch_d')}
    wanted=set(opts.only.split(',')) if opts.only else None
    try:
        for spec in specs:
            sample=next(iter(corpus.values()))
            name,_,_=recorded_call(recorder,spec,sample)
            if wanted and name not in wanted: continue
            key=name+'_batch_d'
            vector=spec.group=='vector_math'
            if key not in FUNCTIONS:
                if vector and name+'_d' in FUNCTIONS:
                    key=name+'_d'
                else:
                    skips.append(dict(function=spec.name,reason='no corresponding device/batch entry point'))
                    continue
            target=FUNCTIONS[key]
            if key in batch_names: covered.add(key)
            groups=defaultdict(list)
            for ds,d in corpus.items():
                if ds in spec.skip_datasets:
                    skips.append(dict(function=spec.name,dataset=ds,reason='existing documented TA-Lib oracle deviation'))
                    continue
                groups[len(d['close'])].append((ds,d))
            start=len(rows)
            for length,items in groups.items():
                datasets=[items[i%len(items)] for i in range(max(opts.series,len(items)))]
                calls=[recorded_call(recorder,spec,d) for _,d in datasets]
                params=calls[0][2]
                unused=set(params)-{v.name for v in target.scalars}
                assert not unused,(name,'unmapped oracle parameters',unused)
                arrays=[np.stack([r[1][j] for r in calls]) for j in range(len(calls[0][1]))]
                frame=Frame(native,target,arrays)
                references=[]
                for _,d in datasets:
                    v=spec.talib_fn(d)
                    references.append(v if isinstance(v,tuple) else (v,))
                try:
                    for mode in modes:
                        os.environ['TACUDA_BATCH_FUSED']=mode
                        paths=[('device-caller',stream,False),('device-default',None,False)]
                        if key in batch_names: paths.append(('host-batch',stream,True))
                        for path,st,host in paths:
                            label=dict(function=spec.name,family=name,mode=mode,path=path,length=length,series=frame.s)
                            try:
                                out=frame.call(params,st,host)
                                for i,(ds,d) in enumerate(datasets):
                                    raw=[a[i] for a in out]
                                    if name=='aroon': raw=[raw[1],raw[0],raw[1]-raw[0]]
                                    result=formatted(recorder,spec,d,raw)
                                    assert len(result)==len(references[i]),'oracle output count differs'
                                    for j,(a,b) in enumerate(zip(result,references[i])):
                                        cell=dict(label,dataset=ds,row=i,output=j,tolerance=spec.tolerance,
                                                  **summarize(a,b,spec.tolerance,spec.discrete))
                                        rows.append(cell)
                            except Exception as exc:
                                errors.append(dict(label,error=f'{type(exc).__name__}: {exc}'))
                finally:
                    frame.close()
            cells=rows[start:]
            print(f'{name}: {sum(x["ok"] for x in cells)}/{len(cells)} oracle cells',flush=True)
        # Every extra batch family is exercised even when TA-Lib has no name
        # for it. Compare against the independently dispatched host function.
        extras=sorted(batch_names-covered)
        default_data=next(iter(corpus.values()))
        input_keys=dict(inputs='close',values='close',closes='close',highs='high',lows='low',
                        opens='open',volumes='volume',xs='close',ys='close2',periods='periods')
        for key in extras:
            name=key[:-8]
            if wanted and name not in wanted: continue
            target=FUNCTIONS[key]
            params=dict(SCALAR_DEFAULTS)
            arrays=[np.stack([default_data[input_keys[n.removeprefix('d_')]].astype('f4') for _ in range(opts.series)]) for n in target.inputs]
            original=FUNCTIONS.get(name)
            if original is None:
                errors.append(dict(family=name,error='no independent single-series host entry point'))
                continue
            host_fn=native.function(original)
            ref=[np.empty_like(arrays[0]) for _ in original.outputs]
            for i in range(opts.series):
                values=dict(params)
                if 'type' in (p.name for p in original.scalars): values['type']=values['maType']
                args=[a[i].ctypes.data for a in arrays+ref]+([arrays[0].shape[1]] if original.size_param else [])
                args += [None if p.name=='stream' else values[p.name] for p in original.scalars]
                native.check(host_fn(*args),original.c_name)
            frame=Frame(native,target,arrays)
            try:
                for mode in modes:
                    os.environ['TACUDA_BATCH_FUSED']=mode
                    for path,st,host in [('device-caller',stream,False),('device-default',None,False),('host-batch',stream,True)]:
                        label=dict(function=name,family=name,mode=mode,path=path,oracle='single-host dispatcher equivalence')
                        try:
                            out=frame.call(params,st,host)
                            for j,(a,b) in enumerate(zip(out,ref)):
                                rows.append(dict(label,output=j,tolerance=1e-4,**summarize(a,b,1e-4)))
                        except Exception as exc:
                            errors.append(dict(label,error=f'{type(exc).__name__}: {exc}'))
                covered.add(key)
            finally:
                frame.close()
            print('extra',name,flush=True)
    finally:
        cuda.cudaStreamDestroy(stream)
    summary=dict(cells=len(rows),passed=sum(x['ok'] for x in rows),failed=sum(not x['ok'] for x in rows),
        errors=len(errors),batch_families=len(covered),batch_families_total=len(batch_names),
        missing_batch_families=sorted(batch_names-covered) if not wanted else [],
        talib=talib.__version__,talib_native=str(talib.__ta_version__),native_sha256=hashlib.sha256(Path(opts.lib).read_bytes()).hexdigest())
    Path(opts.json).write_text(json.dumps(dict(summary=summary,cells=rows,errors=errors,skips=skips),indent=1)+'\n')
    print(json.dumps(summary),flush=True)
    return 1 if summary['failed'] or errors or summary['missing_batch_families'] else 0


if __name__=='__main__':
    raise SystemExit(main())
