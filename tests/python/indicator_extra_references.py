# ============================================================================
#  TaCuda | GPU-accelerated technical analysis | https://tacuda.com
#  Copyright (c) 2024-2026 Paul Deecalov <paul@deecalov.com>
#  SPDX-License-Identifier: Apache-2.0
# ============================================================================
"""CPU formulas for batch families outside the TA-Lib function catalog.

Club-pack contracts are documented in include/indicators/ClubPack.h. PVO is
also checked against the conventional percentage difference of volume EMAs,
not against the library's own finite weighted-window implementation.
"""
import numpy as np
import talib


def reference(name,data,p):
    d={k:np.asarray(v,dtype='f4').astype('f8') for k,v in data.items()}
    c,h,l,o,v=(d[k] for k in ('close','high','low','open','volume'))
    n=len(c);period=p['period'];out=np.full(n,np.nan)
    if name=='change':
        out[period:]=c[period:]-c[:-period]
    elif name in ('cdl_bullish_engulfing','cdl_bearish_engulfing'):
        # These legacy names are aliases of the combined engulfing predicate.
        out[:]=0
        for i in range(1,n):
            bullish=c[i]>=o[i] and c[i-1]<o[i-1] and ((c[i]>=o[i-1] and o[i]<c[i-1]) or (c[i]>o[i-1] and o[i]<=c[i-1]))
            bearish=c[i]<o[i] and c[i-1]>=o[i-1] and ((o[i]>=c[i-1] and c[i]<o[i-1]) or (o[i]>c[i-1] and c[i]<=o[i-1]))
            if bullish or bearish:
                out[i]=(1 if c[i]>=o[i] else -1)*(100 if o[i]!=c[i-1] and c[i]!=o[i-1] else 80)
    elif name in ('nvi','pvi'):
        acc=1000.0;out[0]=acc
        for i in range(1,n):
            active=v[i]<v[i-1] if name=='nvi' else v[i]>v[i-1]
            if active and c[i-1]!=0:acc*=c[i]/c[i-1]
            out[i]=acc
    elif name=='pvo':
        return talib.PPO(v,p['fastPeriod'],p['slowPeriod'],matype=1).astype('f4')
    elif name=='xaverage':
        seeded=False;acc=0.0;alpha=2.0/(period+1)
        for i,x in enumerate(c):
            if not np.isfinite(x):continue
            acc=x if not seeded else alpha*x+(1-alpha)*acc
            seeded=True;out[i]=acc
    elif name in ('rolling_mean','rolling_min','rolling_max','rolling_std','skew','kurtosis'):
        for i in range(period-1,n):
            window=c[i-period+1:i+1]
            if not np.isfinite(window).all():continue
            if name=='rolling_mean':out[i]=window.mean()
            elif name=='rolling_min':out[i]=window.min()
            elif name=='rolling_max':out[i]=window.max()
            elif name=='rolling_std':out[i]=window.std(ddof=0)
            else:
                centered=window-window.mean();m2=np.mean(centered**2)
                out[i]=0 if m2<=0 else np.mean(centered**3)/m2**1.5 if name=='skew' else np.mean(centered**4)/m2**2-3
    elif name in ('pivot_high','pivot_low'):
        strength=p['strength']
        for i in range(2*strength,n):
            j=i-strength;before=c[j-strength:j];after=c[j+1:i+1]
            ok=np.all(c[j]>before) and np.all(c[j]>=after) if name=='pivot_high' else np.all(c[j]<before) and np.all(c[j]<=after)
            if ok:out[i]=c[j]
    elif name=='market_strength':
        for i in range(period,n):
            changes=np.diff(c[i-period:i+1]);net=changes.sum()
            denominator=np.sum(changes[changes>0]) if net>=0 else -np.sum(changes[changes<0])
            out[i]=100*net/denominator if denominator else 0
    elif name=='trendscore':
        lookback=p['lookback']
        for i in range(lookback+10,n):
            out[i]=sum(1 if c[i]>=c[i-lookback-k] else -1 for k in range(1,11))
    elif name=='refdev':
        a,b=p['smaPeriod'],p['sumPeriod']
        mean=np.full(n,np.nan)
        for i in range(a-1,n):mean[i]=c[i-a+1:i+1].mean()
        drd=c-mean
        for i in range(a+b-2,n):
            window=drd[i-b+1:i+1];den=np.abs(window).sum()
            if den>0:out[i]=100*window.sum()/den
    elif name=='hpi':
        acc=0.0;mid=(h+l)/2
        for i in range(1,n):
            acc+=p['factor']*((mid[i]-mid[i-1])*v[i]-acc);out[i]=acc
    elif name=='wobv':
        contribution=np.divide((c-o)*v,h-l,out=np.zeros(n),where=h>l)
        out=np.cumsum(contribution)
    elif name=='obtr':
        tr=np.maximum.reduce([h[1:]-l[1:],np.abs(h[1:]-c[:-1]),np.abs(l[1:]-c[:-1])])
        out[1:]=np.cumsum(np.sign(np.diff(c))*tr)
    elif name=='asi':
        acc=0.0
        for i in range(1,n):
            hc,lc,hl,coy=abs(h[i]-c[i-1]),abs(l[i]-c[i-1]),h[i]-l[i],abs(c[i-1]-o[i-1])
            r=hc-.5*lc+.25*coy if hc>=lc and hc>=hl else lc-.5*hc+.25*coy if lc>=hc and lc>=hl else hl+.25*coy
            if r>1e-12:
                k=max(hc,lc)
                acc+=50*((c[i]-c[i-1])+.5*(c[i]-o[i])+.25*(c[i-1]-o[i-1]))/r*(k/max(1e-12,hl if hl>0 else k))
            out[i]=acc
    elif name=='sar_el':
        if n<2:return out.astype('f4')
        af=np.float32(p['accFact']);maximum=np.float32(p['maxAccel']);step=af
        long=c[1]>=c[0];ep=np.float32(h[1] if long else l[1]);sar=np.float32(l[0] if long else h[0]);out[1]=sar
        for i in range(2,n):
            sar=np.float32(sar+np.float32(af*np.float32(ep-sar)))
            if long:
                sar=np.float32(min(sar,l[i-1],l[i-2]))
                if l[i]<sar:long=False;sar=ep;ep=np.float32(l[i]);af=step
                elif h[i]>ep:ep=np.float32(h[i]);af=min(maximum,np.float32(af+step))
            else:
                sar=np.float32(max(sar,h[i-1],h[i-2]))
                if h[i]>sar:long=True;sar=ep;ep=np.float32(h[i]);af=step
                elif l[i]<ep:ep=np.float32(l[i]);af=min(maximum,np.float32(af+step))
            out[i]=sar
    else:raise KeyError('No CPU formula for '+name)
    return np.asarray(out,dtype='f4')
