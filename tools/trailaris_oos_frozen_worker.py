#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

FIXED={"AUDJPY","USDCAD","US100","GER40","UK100","JP225","XAUUSD","XAGUSD"}
ENERGY={"WTI","BRENT"}
GATED={"EURUSD","GBPUSD","USDJPY","AUDUSD","USDCHF","NZDUSD","EURJPY","GBPJPY","EURGBP","CADJPY","GBPCHF","BTCUSD","ETHUSD","SOLUSD","BNBUSD","XRPUSD","US500","US30","NATGAS"}

def load_core():
    p=Path(__file__).with_name("trailaris_replay_core_frozen.py")
    spec=importlib.util.spec_from_file_location("trailaris_core",p)
    m=importlib.util.module_from_spec(spec);sys.modules["trailaris_core"]=m;spec.loader.exec_module(m)
    return m

def first_true(mask):
    z=np.flatnonzero(mask);return int(z[0]) if len(z) else None

def arrays_for(df,direction):
    if direction>0:return (df['bid_open'].to_numpy(float),df['bid_low'].to_numpy(float),df['bid_high'].to_numpy(float),df['bid_close'].to_numpy(float))
    return (df['ask_open'].to_numpy(float),df['ask_high'].to_numpy(float),df['ask_low'].to_numpy(float),df['ask_close'].to_numpy(float))

def stop_target_masks(direction,lows,highs,stop,target):
    if direction>0:return lows<=stop,highs>=target
    return lows>=stop,highs<=target

def segment_first(direction,lo,hi,stop,target,start,end):
    if start>end:return None,None
    sm,tm=stop_target_masks(direction,lo[start:end+1],hi[start:end+1],stop,target);z=first_true(sm|tm)
    if z is None:return None,None
    return start+z,('stop' if sm[z] else 'target')

def exec_stop(direction,open_px,stop):
    if direction>0:return min(open_px,stop) if open_px<stop else stop
    return max(open_px,stop) if open_px>stop else stop

def close_r(direction,px,entry,risk):return direction*(px-entry)/risk

def make_candidates(core,asset,df,start,end):
    H1,H4,D1=core.htf_features(df)
    if asset=="XAUUSD":c=core.gold_candidates(asset,df,H1,H4,D1)
    elif asset in core.INDEX:c=core.index_candidates(asset,df,H1,H4,D1)
    else:c=core.compression_candidates(asset,df,H1,H4,D1)
    if c is None or not len(c):return c
    tt=pd.to_datetime(c['actual_entry_time'],utc=True);return c[(tt>=start)&(tt<end)].copy()

def simulate_fixed(asset,df,cands,target_r=.75,cost=.02):
    if cands is None or not len(cands):return []
    idx=df.index;out=[];next_free=pd.Timestamp.min.tz_localize('UTC')
    for _,c in cands.sort_values('actual_entry_time').iterrows():
        et=pd.Timestamp(c.actual_entry_time)
        if et<next_free:continue
        p=idx.searchsorted(et)
        if p>=len(idx):continue
        d=int(c.direction);entry=float(c.entry);risk=float(c.risk_distance);stop=float(c.stop)
        if risk<=0:continue
        max_end=idx[-1] if asset!='XAUUSD' else min(idx[-1],et+pd.Timedelta(days=7));qend=max(p,min(idx.searchsorted(max_end,side='right')-1,len(idx)-1));op,lo,hi,cl=arrays_for(df,d);target=entry+d*target_r*risk;q,ev=segment_first(d,lo,hi,stop,target,p,qend)
        if q is None:q=qend;px=cl[q];reason='WINDOW_END'
        elif ev=='stop':px=exec_stop(d,op[q],stop);reason='STOP'
        else:px=target;reason='TARGET'
        rg=close_r(d,px,entry,risk);rn=rg-cost;sl=slice(p,q+1);mfe=((np.nanmax(hi[sl])-entry)/risk if d>0 else (entry-np.nanmin(hi[sl]))/risk);mae=((np.nanmin(lo[sl])-entry)/risk if d>0 else (entry-np.nanmax(lo[sl]))/risk)
        out.append(dict(asset=asset,profile=str(c.profile),direction=d,entry_time=str(et),exit_time=str(idx[q]),entry=entry,stop_initial=stop,risk_distance=risk,target_r=float(c.target_r),micro=bool(c.micro),r_gross=float(rg),r_net=float(rn),reason=reason,mfe_r=float(mfe),mae_r=float(mae)));next_free=idx[q]+pd.Timedelta(microseconds=1)
    return out

def simulate_energy(asset,df,cands,cost=.03):
    if cands is None or not len(cands):return []
    idx=df.index;out=[];next_free=pd.Timestamp.min.tz_localize('UTC')
    for _,c in cands.sort_values('actual_entry_time').iterrows():
        et=pd.Timestamp(c.actual_entry_time)
        if et<next_free:continue
        p=idx.searchsorted(et)
        if p>=len(idx):continue
        d=int(c.direction);entry=float(c.entry);risk=float(c.risk_distance);stop=float(c.stop)
        if risk<=0:continue
        qend=len(idx)-1;op,lo,hi,cl=arrays_for(df,d);pt=entry+d*1.0*risk;rt=entry+d*2.0*risk;q,ev=segment_first(d,lo,hi,stop,pt,p,qend)
        if q is None:exitq=qend;rg=close_r(d,cl[exitq],entry,risk);reason='WINDOW_END_PREPARTIAL'
        elif ev=='stop':exitq=q;rg=close_r(d,exec_stop(d,op[q],stop),entry,risk);reason='STOP_PREPARTIAL'
        else:
            realized=.70;rem=.30
            if (d>0 and hi[q]>=rt) or (d<0 and hi[q]<=rt):exitq=q;rg=realized+rem*2.0;reason='PARTIAL_RUNNER_TARGET_SAME_MIN'
            else:
                activate=idx[q].floor('5min')+pd.Timedelta(minutes=5);aq=idx.searchsorted(activate);qq,ee=segment_first(d,lo,hi,stop,rt,q+1,min(qend,aq-1))
                if qq is not None:
                    exitq=qq
                    if ee=='stop':rg=realized+rem*close_r(d,exec_stop(d,op[qq],stop),entry,risk);reason='PARTIAL_RUNNER_INITIAL_STOP'
                    else:rg=realized+rem*2.0;reason='PARTIAL_RUNNER_TARGET'
                else:
                    newstop=entry;qq,ee=segment_first(d,lo,hi,newstop,rt,max(aq,q+1),qend)
                    if qq is not None:
                        exitq=qq
                        if ee=='stop':rg=realized+rem*close_r(d,exec_stop(d,op[qq],newstop),entry,risk);reason='PARTIAL_RUNNER_LOCK'
                        else:rg=realized+rem*2.0;reason='PARTIAL_RUNNER_TARGET'
                    else:exitq=qend;rg=realized+rem*close_r(d,cl[exitq],entry,risk);reason='WINDOW_END_POSTPARTIAL'
        rn=rg-cost;sl=slice(p,exitq+1);mfe=((np.nanmax(hi[sl])-entry)/risk if d>0 else (entry-np.nanmin(hi[sl]))/risk);mae=((np.nanmin(lo[sl])-entry)/risk if d>0 else (entry-np.nanmax(lo[sl]))/risk)
        out.append(dict(asset=asset,profile=str(c.profile),direction=d,entry_time=str(et),exit_time=str(idx[exitq]),entry=entry,stop_initial=stop,risk_distance=risk,target_r=float(c.target_r),micro=bool(c.micro),r_gross=float(rg),r_net=float(rn),reason=reason,mfe_r=float(mfe),mae_r=float(mae)));next_free=idx[exitq]+pd.Timedelta(microseconds=1)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('asset');ap.add_argument('--entry-start',required=True);ap.add_argument('--entry-end',required=True);ap.add_argument('--outdir',default='trailaris_oos_results');a=ap.parse_args();core=load_core();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True);start=pd.Timestamp(a.entry_start,tz='UTC');end=pd.Timestamp(a.entry_end,tz='UTC');meta={'asset':a.asset,'scope':'ACTIVE_IN_CYCLE','entry_start':str(start),'entry_end_exclusive':str(end)}
    if a.asset in GATED:
        meta.update(status='EVIDENCE_GATE_ZERO',trades=0,policy='NONE');pd.DataFrame(columns=['asset','profile','direction','entry_time','exit_time','r_net']).to_csv(out/f'{a.asset}_trades.csv',index=False)
    else:
        df=core.load(a.asset);c=make_candidates(core,a.asset,df,start,end)
        if a.asset in FIXED:rr=simulate_fixed(a.asset,df,c);policy='FIXED_TARGET_0.75R'
        elif a.asset in ENERGY:rr=simulate_energy(a.asset,df,c);policy='PARTIAL_1R_70_RUNNER_2R_BE_NEXT_M5'
        else:raise SystemExit('unmapped asset '+a.asset)
        pd.DataFrame(rr).to_csv(out/f'{a.asset}_trades.csv',index=False);meta.update(status='REPLAYED',policy=policy,candidates=int(0 if c is None else len(c)),trades=len(rr),mean_r=(float(np.mean([x['r_net'] for x in rr])) if rr else None),sum_r=(float(np.sum([x['r_net'] for x in rr])) if rr else None),win_rate=(float(np.mean([x['r_net']>0 for x in rr])) if rr else None),data_start=str(df.index.min()),data_end=str(df.index.max()))
    (out/f'{a.asset}_meta.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
if __name__=='__main__':main()
