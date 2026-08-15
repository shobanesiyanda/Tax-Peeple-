#!/usr/bin/env python3
from __future__ import annotations
import json, math, os
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT=Path(os.environ.get('TRAILARIS_RAW_ROOT','trailaris_oos_raw'))
OUT=Path(os.environ.get('TRAILARIS_OUT','trailaris_oos_core_out'))
OUT.mkdir(exist_ok=True)

ROUTES=['EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD','EURJPY','GBPJPY','EURGBP','AUDJPY','CADJPY','GBPCHF','XAUUSD','XAGUSD','BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD','US100','US500','US30','GER40','UK100','JP225','WTI','BRENT','NATGAS','V75','V50','V100','V25','V10']
SYNTH={'V75','V50','V100','V25','V10'}
INDEX={'US100','US500','US30','GER40','UK100','JP225'}
FX=set(['EURUSD','GBPUSD','USDJPY','AUDUSD','USDCAD','USDCHF','NZDUSD','EURJPY','GBPJPY','EURGBP','AUDJPY','CADJPY','GBPCHF'])
CRYPTO={'BTCUSD','ETHUSD','SOLUSD','BNBUSD','XRPUSD'}

# (timeframe, lookback, compression_atr, breakout_atr, stop_buffer_atr, target_r, require_daily, micro)
PARAM={
 'XAGUSD':('1h',10,3.50,.05,.15,3.0,True,False),
 'EURUSD':('15min',8,2.25,.05,.15,3.0,True,False),
 'GBPUSD':('15min',10,2.75,.05,.15,3.0,True,False),
 'USDJPY':('5min',12,3.25,.08,.12,3.0,True,True),
 'AUDUSD':('15min',10,2.50,.05,.15,3.0,True,False),
 'USDCAD':('15min',10,2.75,.05,.15,3.0,True,False),
 'USDCHF':('15min',10,2.75,.05,.15,3.0,True,False),
 'NZDUSD':('15min',10,2.75,.05,.15,3.0,True,False),
 'EURJPY':('15min',10,2.75,.05,.15,3.0,True,False),
 'GBPJPY':('15min',10,2.75,.05,.15,3.0,True,False),
 'EURGBP':('15min',10,2.75,.05,.15,3.0,True,False),
 'AUDJPY':('15min',10,2.75,.05,.15,3.0,True,False),
 'CADJPY':('15min',10,2.75,.05,.15,3.0,True,False),
 'GBPCHF':('15min',10,2.75,.05,.15,3.0,True,False),
 'BTCUSD':('15min',12,4.00,.08,.20,2.5,False,False),
 'ETHUSD':('15min',12,4.25,.08,.20,2.5,False,False),
 'SOLUSD':('15min',12,4.75,.10,.25,2.5,False,False),
 'BNBUSD':('15min',12,4.00,.08,.20,2.5,False,False),
 'XRPUSD':('15min',12,4.00,.08,.20,2.5,False,False),
 'WTI':('15min',10,3.50,.08,.20,3.0,True,False),
 'BRENT':('15min',10,3.50,.08,.20,3.0,True,False),
 'NATGAS':('15min',10,3.50,.08,.20,3.0,True,False),
}
INDEX_OPEN={'US100':'13:30','US500':'13:30','US30':'13:30','GER40':'07:00','UK100':'07:00','JP225':'00:00'}

USECOLS=['timestamp','open','high','low','close','spread','bid_open','bid_high','bid_low','bid_close','ask_open','ask_high','ask_low','ask_close','source','source_symbol','raw_kind']

def load(asset):
    p=ROOT/f'{asset}_M1_normalized.csv'
    df=pd.read_csv(p,usecols=USECOLS)
    df['timestamp']=pd.to_datetime(df.timestamp,utc=True)
    df=df.drop_duplicates('timestamp').sort_values('timestamp').set_index('timestamp')
    return df

def resample(df, rule):
    o=df['open'].resample(rule,label='left',closed='left').first()
    h=df['high'].resample(rule,label='left',closed='left').max()
    l=df['low'].resample(rule,label='left',closed='left').min()
    c=df['close'].resample(rule,label='left',closed='left').last()
    x=pd.concat({'open':o,'high':h,'low':l,'close':c},axis=1).dropna()
    delta=pd.Timedelta(rule)
    x['entry_time']=x.index+delta
    prev=x['close'].shift(1)
    tr=pd.concat([(x.high-x.low).abs(),(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1)
    x['atr14']=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    return x

def htf_features(df):
    H1=resample(df,'1h');H4=resample(df,'4h');D1=resample(df,'1d')
    for X in (H1,H4):
        X['ema20']=X.close.ewm(span=20,adjust=False,min_periods=20).mean();X['ema50']=X.close.ewm(span=50,adjust=False,min_periods=50).mean()
    D1['ema20']=D1.close.ewm(span=20,adjust=False,min_periods=20).mean();D1['ema50']=D1.close.ewm(span=50,adjust=False,min_periods=50).mean();D1['ema10']=D1.close.ewm(span=10,adjust=False,min_periods=10).mean();D1['ema30']=D1.close.ewm(span=30,adjust=False,min_periods=30).mean();D1['close10ago']=D1.close.shift(10)
    return H1,H4,D1

def asof_features(times, feat, cols, prefix):
    L=pd.DataFrame({'entry_time':pd.to_datetime(times,utc=True)}).sort_values('entry_time')
    R=feat[['entry_time']+cols].dropna(subset=['entry_time']).sort_values('entry_time');R=R.rename(columns={c:prefix+c for c in cols})
    return pd.merge_asof(L,R,on='entry_time',direction='backward')

def align_flags(sig,H1,H4,D1, require_daily=True):
    times=sig['entry_time'];a=asof_features(times,H1,['close','ema20','ema50'],'h1_');b=asof_features(times,H4,['close','ema20','ema50'],'h4_');d=asof_features(times,D1,['close','ema20','ema50'],'d1_')
    z=pd.concat([a.reset_index(drop=True),b.drop(columns='entry_time').reset_index(drop=True),d.drop(columns='entry_time').reset_index(drop=True)],axis=1)
    di=sig['direction'].to_numpy();long=(di>0);short=(di<0)
    h1_long=(z.h1_ema20>z.h1_ema50)&(z.h1_close>z.h1_ema20);h1_short=(z.h1_ema20<z.h1_ema50)&(z.h1_close<z.h1_ema20)
    h4_long=(z.h4_ema20>z.h4_ema50)&(z.h4_close>z.h4_ema20);h4_short=(z.h4_ema20<z.h4_ema50)&(z.h4_close<z.h4_ema20)
    ok=((long & h1_long & h4_long)|(short & h1_short & h4_short)).fillna(False).to_numpy()
    if require_daily:
        dlong=(z.d1_ema20>z.d1_ema50)&(z.d1_close>z.d1_ema20);dshort=(z.d1_ema20<z.d1_ema50)&(z.d1_close<z.d1_ema20)
        ok &= ((long & dlong)|(short & dshort)).fillna(False).to_numpy()
    return ok

def entry_rows(df, sig):
    idx=df.index;pos=idx.searchsorted(pd.DatetimeIndex(sig.entry_time));valid=pos<len(idx);out=sig.loc[valid].copy().reset_index(drop=True);pos=np.asarray(pos)[valid];ent=idx[pos];latency=(ent-pd.DatetimeIndex(out.entry_time)).total_seconds()/60;ok=np.asarray(latency<=5);out=out.loc[ok].copy().reset_index(drop=True);pos=pos[ok];rows=df.iloc[pos]
    out['actual_entry_time']=rows.index.to_numpy();out['bid_entry']=rows.bid_open.to_numpy();out['ask_entry']=rows.ask_open.to_numpy();out['spread_entry']=out.ask_entry-out.bid_entry
    return out

def compression_candidates(asset,df,H1,H4,D1):
    rule,look,comp,brk,stopbuf,target,reqd,micro=PARAM[asset];S=resample(df,rule);S['prior_high']=S.high.shift(1).rolling(look,min_periods=look).max();S['prior_low']=S.low.shift(1).rolling(look,min_periods=look).min();S['prior_range']=S.prior_high-S.prior_low;S['candle_range']=S.high-S.low;S['body_ratio']=(S.close-S.open).abs()/S.candle_range.replace(0,np.nan);long=(S.close>S.prior_high+brk*S.atr14)&(S.close>S.open);short=(S.close<S.prior_low-brk*S.atr14)&(S.close<S.open);base=(S.prior_range<=comp*S.atr14)&(S.body_ratio>=.45)&S.atr14.notna();S['direction']=np.select([long&base,short&base],[1,-1],default=0);sig=S[S.direction!=0].copy()
    if sig.empty:return sig
    sig['signal_low']=sig.low;sig['signal_high']=sig.high;sig['signal_atr']=sig.atr14;ok=align_flags(sig,H1,H4,D1,reqd);sig=sig.loc[ok].copy()
    if sig.empty:return sig
    E=entry_rows(df,sig)
    if E.empty:return E
    E['entry']=np.where(E.direction>0,E.ask_entry,E.bid_entry);E['stop']=np.where(E.direction>0,E.signal_low-stopbuf*E.signal_atr,E.signal_high+E.spread_entry+stopbuf*E.signal_atr);E['risk_distance']=(E.entry-E.stop).abs();E['target_r']=target;E['target']=E.entry+E.direction*target*E.risk_distance;E['micro']=micro;E['profile']='COMPRESSION_MOMENTUM';E['asset']=asset;good=(E.risk_distance>=.40*E.signal_atr)&(E.risk_distance<=3.00*E.signal_atr)&(E.spread_entry<=.06*E.risk_distance)
    return E.loc[good].copy()

def gold_candidates(asset,df,H1,H4,D1):
    S=H4.copy();S['prior_high']=S.high.shift(1).rolling(20,min_periods=20).max();S['prior_low']=S.low.shift(1).rolling(20,min_periods=20).min();S['candle_range']=S.high-S.low;S['body_ratio']=(S.close-S.open).abs()/S.candle_range.replace(0,np.nan);S['ema20']=S.close.ewm(span=20,adjust=False,min_periods=20).mean();S['ema50']=S.close.ewm(span=50,adjust=False,min_periods=50).mean();dg=asof_features(S.entry_time,D1,['close','ema10','ema30','close10ago'],'d_');dg.index=S.index;long_daily=(dg.d_ema10>dg.d_ema30)&(dg.d_close>dg.d_ema10)&(dg.d_close>dg.d_close10ago);short_daily=(dg.d_ema10<dg.d_ema30)&(dg.d_close<dg.d_ema10)&(dg.d_close<dg.d_close10ago);body=(S.body_ratio>=.50)&(S.candle_range>=.65*S.atr14)&(S.candle_range<=2.75*S.atr14);long=body&long_daily&(S.ema20>S.ema50)&(S.close>S.prior_high+.03*S.atr14)&(S.close>S.open);short=body&short_daily&(S.ema20<S.ema50)&(S.close<S.prior_low-.03*S.atr14)&(S.close<S.open);S['direction']=np.select([long,short],[1,-1],default=0);sig=S[S.direction!=0].copy()
    if sig.empty:return sig
    sig['signal_low']=sig.low;sig['signal_high']=sig.high;sig['signal_atr']=sig.atr14;E=entry_rows(df,sig)
    if E.empty:return E
    E['entry']=np.where(E.direction>0,E.ask_entry,E.bid_entry);E['stop']=np.where(E.direction>0,E.signal_low-.15*E.signal_atr,E.signal_high+E.spread_entry+.15*E.signal_atr);E['risk_distance']=(E.entry-E.stop).abs();E['target_r']=3.0;E['target']=E.entry+E.direction*3.0*E.risk_distance;E['micro']=False;E['profile']='GOLD_H4_BREAKOUT';E['asset']=asset;good=(E.risk_distance>=.55*E.signal_atr)&(E.risk_distance<=3.25*E.signal_atr)&(E.spread_entry<=.04*E.risk_distance)
    return E.loc[good].copy()

def index_candidates(asset,df,H1,H4,D1):
    M5=resample(df,'5min');hh,mm=map(int,INDEX_OPEN[asset].split(':'));rows=[]
    for day,g in M5.groupby(M5.index.date):
        open_t=pd.Timestamp(day,tz='UTC')+pd.Timedelta(hours=hh,minutes=mm);range_end=open_t+pd.Timedelta(minutes=15);rg=g[(g.index>=open_t)&(g.index<range_end)]
        if len(rg)<3:continue
        oh=rg.high.max();ol=rg.low.min();scan=g[(g.index>=range_end)&(g.index<open_t+pd.Timedelta(hours=4))]
        if len(scan)<2:continue
        prev_close=scan.close.shift(1);long=(prev_close<=oh)&(prev_close>=ol)&(scan.close>oh)&(scan.close>scan.open);short=(prev_close<=oh)&(prev_close>=ol)&(scan.close<ol)&(scan.close<scan.open);ss=scan[long|short].copy()
        if ss.empty:continue
        ss['direction']=np.where(long.loc[ss.index],1,-1);ss['signal_low']=ss.low;ss['signal_high']=ss.high;ss['signal_atr']=ss.atr14;rows.append(ss)
    if not rows:return pd.DataFrame()
    sig=pd.concat(rows).sort_index();ok=align_flags(sig,H1,H4,D1,False);sig=sig.loc[ok].copy()
    if sig.empty:return sig
    E=entry_rows(df,sig)
    if E.empty:return E
    E['entry']=np.where(E.direction>0,E.ask_entry,E.bid_entry);E['stop']=np.where(E.direction>0,E.signal_low-.10*E.signal_atr,E.signal_high+E.spread_entry+.10*E.signal_atr);E['risk_distance']=(E.entry-E.stop).abs();E['target_r']=2.5;E['target']=E.entry+E.direction*2.5*E.risk_distance;E['micro']=True;E['profile']='INDEX_ORB';E['asset']=asset;good=(E.risk_distance>=.30*E.signal_atr)&(E.risk_distance<=2.50*E.signal_atr)&(E.spread_entry<=.06*E.risk_distance)
    return E.loc[good].copy()

@dataclass
class Trade:
    asset:str; profile:str; direction:int; entry_time:str; exit_time:str; entry:float; stop_initial:float; target:float; risk_distance:float; target_r:float; micro:bool
    exit_price:float; r_gross:float; r_net:float; reason:str; mfe_r:float; mae_r:float; minutes_to_positive:float|None
