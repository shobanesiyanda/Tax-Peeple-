#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

def load_core():
 p=Path(__file__).with_name('trailaris_replay_core_frozen.py');s=importlib.util.spec_from_file_location('tr',p);m=importlib.util.module_from_spec(s);sys.modules['tr']=m;s.loader.exec_module(m);return m

def first(mask):
 z=np.flatnonzero(mask);return int(z[0]) if len(z) else None

def arrays(df,d):
 return (df.bid_open.to_numpy(float),df.bid_low.to_numpy(float),df.bid_high.to_numpy(float),df.bid_close.to_numpy(float)) if d>0 else (df.ask_open.to_numpy(float),df.ask_high.to_numpy(float),df.ask_low.to_numpy(float),df.ask_close.to_numpy(float))
def exstop(d,o,s):return min(o,s) if d>0 and o<s else max(o,s) if d<0 and o>s else s
def extarg(d,o,t):return max(o,t) if d>0 and o>t else min(o,t) if d<0 and o<t else t

def simulate(asset,df,cands,target_r,cost=.02,max_hold_days=5):
 if cands is None or cands.empty:return []
 idx=df.index;rr=[];free=pd.Timestamp.min.tz_localize('UTC')
 for _,c in cands.sort_values('actual_entry_time').iterrows():
  et=pd.Timestamp(c.actual_entry_time)
  if et<free:continue
  p=idx.searchsorted(et)
  if p>=len(idx):continue
  d=int(c.direction);e=float(c.entry);risk=float(c.risk_distance);st=float(c.stop)
  if not np.isfinite(risk) or risk<=0:continue
  qend=min(len(idx)-1,idx.searchsorted(min(idx[-1],et+pd.Timedelta(days=max_hold_days)),side='right')-1);o,lo,hi,cl=arrays(df,d);tg=e+d*target_r*risk
  sm=(lo[p:qend+1]<=st) if d>0 else (lo[p:qend+1]>=st);tm=(hi[p:qend+1]>=tg) if d>0 else (hi[p:qend+1]<=tg);z=first(sm|tm)
  if z is None:q=qend;px=cl[q];why='TIME'
  else:
   q=p+z
   if sm[z]:px=exstop(d,o[q],st);why='STOP'
   else:px=extarg(d,o[q],tg);why='TARGET'
  rg=d*(px-e)/risk;rn=rg-cost;rr.append({'asset':asset,'family':str(c.family),'direction':d,'entry_time':str(et),'exit_time':str(idx[q]),'r_net':rn,'r_gross':rg,'target_r':target_r,'reason':why});free=idx[q]+pd.Timedelta(microseconds=1)
 return rr

def entryify(tr,df,sig,START,END,stopbuf=.15):
 if sig is None or sig.empty:return sig
 E=tr.entry_rows(df,sig)
 if E.empty:return E
 E['entry']=np.where(E.direction>0,E.ask_entry,E.bid_entry);E['stop']=np.where(E.direction>0,E.signal_low-stopbuf*E.signal_atr,E.signal_high+E.spread_entry+stopbuf*E.signal_atr);E['risk_distance']=(E.entry-E.stop).abs();good=(E.risk_distance>=.30*E.signal_atr)&(E.risk_distance<=3.25*E.signal_atr)&(E.spread_entry<=.06*E.risk_distance);E=E.loc[good].copy();tt=pd.to_datetime(E.actual_entry_time,utc=True);return E[(tt>=START)&(tt<END)].copy()
def merge_htf(tr,sig,H1,H4):
 a=tr.asof_features(sig.entry_time,H1,['close','ema20','ema50','atr14'],'h1_');b=tr.asof_features(sig.entry_time,H4,['close','ema20','ema50','atr14'],'h4_');return pd.concat([a.reset_index(drop=True),b.drop(columns='entry_time').reset_index(drop=True)],axis=1)

def donchian(tr,df,H1,H4,D1,START,END):
 S=tr.resample(df,'15min');S['ph']=S.high.shift(1).rolling(20,min_periods=20).max();S['pl']=S.low.shift(1).rolling(20,min_periods=20).min();S['rng']=S.high-S.low;S['body']=(S.close-S.open).abs()/S.rng.replace(0,np.nan);long=(S.close>S.ph+.08*S.atr14)&(S.close>S.open);short=(S.close<S.pl-.08*S.atr14)&(S.close<S.open);S['direction']=np.select([long,short],[1,-1],0);sig=S[(S.direction!=0)&(S.body>=.50)&(S.rng<=2.75*S.atr14)].copy()
 if sig.empty:return sig
 z=merge_htf(tr,sig,H1,H4);di=sig.direction.to_numpy();ok=(((di>0)&(z.h1_ema20>z.h1_ema50)&(z.h1_close>z.h1_ema20)&(z.h4_ema20>z.h4_ema50))|((di<0)&(z.h1_ema20<z.h1_ema50)&(z.h1_close<z.h1_ema20)&(z.h4_ema20<z.h4_ema50))).fillna(False).to_numpy();sig=sig.loc[ok].copy()
 if sig.empty:return sig
 sig['signal_low']=sig.low;sig['signal_high']=sig.high;sig['signal_atr']=sig.atr14;sig['family']='DONCHIAN_TREND';return entryify(tr,df,sig,START,END,.12)
def pullback(tr,df,H1,H4,D1,START,END):
 S=tr.resample(df,'15min');S['ema20']=S.close.ewm(span=20,adjust=False,min_periods=20).mean();S['ema50']=S.close.ewm(span=50,adjust=False,min_periods=50).mean();S['ph']=S.high.shift(1);S['pl']=S.low.shift(1);S['rng']=S.high-S.low;S['body']=(S.close-S.open).abs()/S.rng.replace(0,np.nan);long=(S.ema20>S.ema50)&(S.low<=S.ema20+.20*S.atr14)&(S.close>S.ema20)&(S.close>S.ph)&(S.close>S.open);short=(S.ema20<S.ema50)&(S.high>=S.ema20-.20*S.atr14)&(S.close<S.ema20)&(S.close<S.pl)&(S.close<S.open);S['direction']=np.select([long,short],[1,-1],0);sig=S[(S.direction!=0)&(S.body>=.35)&(S.rng<=2.5*S.atr14)].copy()
 if sig.empty:return sig
 z=merge_htf(tr,sig,H1,H4);di=sig.direction.to_numpy();ok=(((di>0)&(z.h1_ema20>z.h1_ema50)&(z.h4_ema20>z.h4_ema50))|((di<0)&(z.h1_ema20<z.h1_ema50)&(z.h4_ema20<z.h4_ema50))).fillna(False).to_numpy();sig=sig.loc[ok].copy()
 if sig.empty:return sig
 sig['signal_low']=sig.low;sig['signal_high']=sig.high;sig['signal_atr']=sig.atr14;sig['family']='PULLBACK_CONT';return entryify(tr,df,sig,START,END,.12)
def fade(tr,df,H1,H4,D1,START,END):
 S=tr.resample(df,'15min');S['ph']=S.high.shift(1).rolling(12,min_periods=12).max();S['pl']=S.low.shift(1).rolling(12,min_periods=12).min();S['rng']=S.high-S.low;up=(S.high>S.ph+.05*S.atr14)&(S.close<S.ph)&(S.close<S.open);dn=(S.low<S.pl-.05*S.atr14)&(S.close>S.pl)&(S.close>S.open);S['direction']=np.select([dn,up],[1,-1],0);sig=S[(S.direction!=0)&(S.rng<=2.75*S.atr14)].copy()
 if sig.empty:return sig
 z=merge_htf(tr,sig,H1,H4);strength=(abs(z.h4_ema20-z.h4_ema50)/z.h4_atr14);ok=(strength<.80).fillna(False).to_numpy();sig=sig.loc[ok].copy()
 if sig.empty:return sig
 sig['signal_low']=sig.low;sig['signal_high']=sig.high;sig['signal_atr']=sig.atr14;sig['family']='FALSE_BREAKOUT_FADE';return entryify(tr,df,sig,START,END,.10)
def session_break(tr,asset,df,H1,H4,D1,START,END):
 if asset not in tr.FX and asset not in {'XAUUSD','XAGUSD'}:return pd.DataFrame()
 M=tr.resample(df,'15min');rows=[]
 for day,g in M.groupby(M.index.date):
  d0=pd.Timestamp(day,tz='UTC');asia=g[(g.index>=d0)&(g.index<d0+pd.Timedelta(hours=6))]
  if len(asia)<16:continue
  ah,al=asia.high.max(),asia.low.min();scan=g[(g.index>=d0+pd.Timedelta(hours=6))&(g.index<d0+pd.Timedelta(hours=10))]
  if scan.empty:continue
  prev=scan.close.shift(1);long=(prev<=ah)&(scan.close>ah+.03*scan.atr14)&(scan.close>scan.open);short=(prev>=al)&(scan.close<al-.03*scan.atr14)&(scan.close<scan.open);s=scan[long|short].copy()
  if s.empty:continue
  s=s.iloc[[0]].copy();s['direction']=np.where(long.loc[s.index],1,-1);rows.append(s)
 if not rows:return pd.DataFrame()
 sig=pd.concat(rows).sort_index();z=merge_htf(tr,sig,H1,H4);di=sig.direction.to_numpy();ok=(((di>0)&(z.h4_ema20>z.h4_ema50))|((di<0)&(z.h4_ema20<z.h4_ema50))).fillna(False).to_numpy();sig=sig.loc[ok].copy()
 if sig.empty:return sig
 sig['signal_low']=sig.low;sig['signal_high']=sig.high;sig['signal_atr']=sig.atr14;sig['family']='ASIA_LONDON_BREAK';return entryify(tr,df,sig,START,END,.10)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('asset');ap.add_argument('--start',required=True);ap.add_argument('--end',required=True);ap.add_argument('--outdir',default='trailaris_r3_research');a=ap.parse_args();tr=load_core();START=pd.Timestamp(a.start,tz='UTC');END=pd.Timestamp(a.end,tz='UTC');out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 if a.asset in tr.SYNTH:(out/f'{a.asset}_meta.json').write_text(json.dumps({'asset':a.asset,'status':'TICK_FIDELITY_BLOCKED'},indent=2));return
 df=tr.load(a.asset);H1,H4,D1=tr.htf_features(df);fams={'donchian':lambda:donchian(tr,df,H1,H4,D1,START,END),'pullback':lambda:pullback(tr,df,H1,H4,D1,START,END),'fade':lambda:fade(tr,df,H1,H4,D1,START,END),'session':lambda:session_break(tr,a.asset,df,H1,H4,D1,START,END)};meta={'asset':a.asset,'status':'RESEARCHED','period':[str(START),str(END)],'policies':{}}
 for fn,f in fams.items():
  try:c=f()
  except Exception as e:meta['policies'][fn]={'error':repr(e)};continue
  for target in (.75,1.0,1.5):
   rr=simulate(a.asset,df,c,target);name=f'{fn}_{target:.2f}';pd.DataFrame(rr).to_csv(out/f'{a.asset}__{name}.csv',index=False);meta['policies'][name]={'signals':int(0 if c is None else len(c)),'trades':len(rr),'mean_r':float(np.mean([x['r_net'] for x in rr])) if rr else None,'sum_r':float(np.sum([x['r_net'] for x in rr])) if rr else None,'win_rate':float(np.mean([x['r_net']>0 for x in rr])) if rr else None}
 (out/f'{a.asset}_meta.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
if __name__=='__main__':main()
