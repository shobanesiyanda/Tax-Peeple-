#!/usr/bin/env python3
from __future__ import annotations
import argparse, asyncio, csv, io, json, tempfile, time, urllib.error, urllib.request, zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

EXNESS={
"EURUSD":"EURUSD_Raw_Spread","GBPUSD":"GBPUSD_Raw_Spread","USDJPY":"USDJPY_Raw_Spread","AUDUSD":"AUDUSD_Raw_Spread","USDCAD":"USDCAD_Raw_Spread","USDCHF":"USDCHF_Raw_Spread","NZDUSD":"NZDUSD_Raw_Spread","EURJPY":"EURJPY_Raw_Spread","GBPJPY":"GBPJPY_Raw_Spread","EURGBP":"EURGBP_Raw_Spread","AUDJPY":"AUDJPY_Raw_Spread","CADJPY":"CADJPY_Raw_Spread","GBPCHF":"GBPCHF_Raw_Spread","XAUUSD":"XAUUSD_Raw_Spread","XAGUSD":"XAGUSD_Raw_Spread","BTCUSD":"BTCUSD_Raw_Spread","ETHUSD":"ETHUSD_Raw_Spread","SOLUSD":"SOLUSD_Raw_Spread","BNBUSD":"BNBUSD_Raw_Spread","XRPUSD":"XRPUSD_Raw_Spread","US100":"USTEC_Raw_Spread","US500":"US500_Raw_Spread","US30":"US30_Raw_Spread","GER40":"DE30_Raw_Spread","UK100":"UK100_Raw_Spread","JP225":"JP225_Raw_Spread","WTI":"USOIL_Raw_Spread","BRENT":"UKOIL_Raw_Spread","NATGAS":"XNGUSD_Raw_Spread"}
DERIV={"V75":"R_75","V50":"R_50","V100":"R_100","V25":"R_25","V10":"R_10"}
ROUTES=list(EXNESS)+list(DERIV)
FIELDS=["timestamp","open","high","low","close","spread","bid_open","bid_high","bid_low","bid_close","ask_open","ask_high","ask_low","ask_close","spread_open","spread_high","spread_low","spread_close","tick_count","source","source_symbol","raw_kind"]

def parse_date(s): return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
def iso_min(ts): return ts[:16].replace(' ','T')+':00Z'
def dt_ex(ts): return datetime.fromisoformat(ts.replace('Z','+00:00'))
def newbar(ts,bid,ask):
 m=(bid+ask)/2; sp=ask-bid
 return {"timestamp":iso_min(ts),"open":m,"high":m,"low":m,"close":m,"spread":sp,"bid_open":bid,"bid_high":bid,"bid_low":bid,"bid_close":bid,"ask_open":ask,"ask_high":ask,"ask_low":ask,"ask_close":ask,"spread_open":sp,"spread_high":sp,"spread_low":sp,"spread_close":sp,"tick_count":1}
def upd(b,bid,ask):
 m=(bid+ask)/2;sp=ask-bid;b['high']=max(b['high'],m);b['low']=min(b['low'],m);b['close']=m;b['bid_high']=max(b['bid_high'],bid);b['bid_low']=min(b['bid_low'],bid);b['bid_close']=bid;b['ask_high']=max(b['ask_high'],ask);b['ask_low']=min(b['ask_low'],ask);b['ask_close']=ask;b['spread_high']=max(b['spread_high'],sp);b['spread_low']=min(b['spread_low'],sp);b['spread_close']=sp;b['spread']+=sp;b['tick_count']+=1

def open_retry(url,retries=4,timeout=180):
 last=None
 for i in range(retries):
  try:return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TrailarisResearchRange/1.0'}),timeout=timeout)
  except urllib.error.HTTPError as e:
   if e.code==404:return None
   last=e
  except Exception as e:last=e
  time.sleep(2**i)
 raise last

def download(url,dst):
 r=open_retry(url)
 if r is None:return False
 with r,open(dst,'wb') as f:
  while True:
   c=r.read(2*1024*1024)
   if not c:break
   f.write(c)
 return True

def process_zip(sym,zp,wr,st,START,END):
 with zipfile.ZipFile(zp) as z:
  names=[n for n in z.namelist() if not n.endswith('/')]
  if not names:raise RuntimeError('empty zip')
  with z.open(names[0]) as raw:
   rd=csv.reader(io.TextIOWrapper(raw,encoding='utf-8-sig',newline=''));cur=None
   for row in rd:
    if len(row)<5 or row[0].lower()!='exness':continue
    try:d=dt_ex(row[2])
    except:st['bad_rows']+=1;continue
    if d<START:continue
    if d>=END:break
    try:bid=float(row[3]);ask=float(row[4])
    except:st['bad_rows']+=1;continue
    if not (bid>0 and ask>=bid):st['bad_rows']+=1;continue
    mk=iso_min(row[2])
    if cur is None or cur['timestamp']!=mk:
     if cur:
      cur['spread']/=cur['tick_count'];cur.update(source='EXNESS_RAW_SPREAD',source_symbol=sym,raw_kind='BID_ASK_TICK_AGGREGATED_M1');wr.writerow(cur);st['bars']+=1
     cur=newbar(row[2],bid,ask)
     if st['first_ts'] is None:st['first_ts']=mk
     st['last_ts']=mk
    else:upd(cur,bid,ask)
    st['ticks']+=1
   if cur:
    cur['spread']/=cur['tick_count'];cur.update(source='EXNESS_RAW_SPREAD',source_symbol=sym,raw_kind='BID_ASK_TICK_AGGREGATED_M1');wr.writerow(cur);st['bars']+=1;st['last_ts']=cur['timestamp']

def month_iter(start,end):
 d=datetime(start.year,start.month,1,tzinfo=timezone.utc)
 while d<end:
  yield d.year,d.month
  d=(d.replace(day=28)+timedelta(days=4)).replace(day=1)

def mat_ex(asset,out,START,END):
 sym=EXNESS[asset];st={'asset':asset,'source':'EXNESS_RAW_SPREAD','symbol':sym,'bars':0,'ticks':0,'bad_rows':0,'first_ts':None,'last_ts':None,'objects':[]}
 with open(out,'w',newline='',encoding='utf-8') as f:
  wr=csv.DictWriter(f,fieldnames=FIELDS);wr.writeheader()
  with tempfile.TemporaryDirectory() as td:
   td=Path(td)
   for y,m in month_iter(START,END):
    mm=f'{m:02d}';url=f'https://ticks.ex2archive.com/ticks/{sym}/{y}/{mm}/Exness_{sym}_{y}_{mm}.zip';zp=td/f'{asset}_{y}{mm}.zip'
    if not download(url,zp):raise RuntimeError('missing '+url)
    st['objects'].append({'url':url,'bytes':zp.stat().st_size});process_zip(sym,zp,wr,st,START,END);zp.unlink(missing_ok=True)
 if st['bars']==0:raise RuntimeError('zero bars '+asset)
 return st

async def dreq(sym,start,end):
 import websockets
 async with websockets.connect('wss://ws.derivws.com/websockets/v3?app_id=1089',open_timeout=30,close_timeout=10,ping_interval=20) as ws:
  await ws.send(json.dumps({'ticks_history':sym,'start':start,'end':end,'style':'candles','granularity':60,'adjust_start_time':1,'count':5000,'req_id':1}))
  p=json.loads(await asyncio.wait_for(ws.recv(),45))
  if p.get('error'):raise RuntimeError(json.dumps(p['error']))
  return p.get('candles',[])
async def mat_deriv(asset,out,START,END):
 sym=DERIV[asset];st={'asset':asset,'source':'DERIV_TICKS_HISTORY','symbol':sym,'bars':0,'first_ts':None,'last_ts':None,'chunks':0}
 with open(out,'w',newline='',encoding='utf-8') as f:
  wr=csv.DictWriter(f,fieldnames=FIELDS);wr.writeheader();s=int(START.timestamp());final=int(END.timestamp());seen=set()
  while s<final:
   e=min(final-60,s+3*86400-60);c=None;last=None
   for a in range(4):
    try:c=await dreq(sym,s,e);break
    except Exception as ex:last=ex;await asyncio.sleep(2**a)
   if c is None:raise last
   st['chunks']+=1
   for x in c:
    ep=int(x['epoch'])
    if ep in seen or ep<s or ep>=final:continue
    seen.add(ep);ts=datetime.fromtimestamp(ep,tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:00Z');row={k:'' for k in FIELDS};row.update(timestamp=ts,open=float(x['open']),high=float(x['high']),low=float(x['low']),close=float(x['close']),spread=0.0,source='DERIV_TICKS_HISTORY',source_symbol=sym,raw_kind='DERIV_CANDLE_M1_NO_HISTORICAL_BIDASK');wr.writerow(row);st['bars']+=1
    if st['first_ts'] is None:st['first_ts']=ts
    st['last_ts']=ts
   s+=3*86400
 if st['bars']==0:raise RuntimeError('zero bars '+asset)
 return st

def main():
 p=argparse.ArgumentParser();p.add_argument('asset',choices=ROUTES);p.add_argument('--start',required=True);p.add_argument('--end',required=True);p.add_argument('--outdir',default='trailaris_range');a=p.parse_args();START=parse_date(a.start);END=parse_date(a.end)
 if END<=START:raise SystemExit('bad range')
 od=Path(a.outdir);od.mkdir(parents=True,exist_ok=True);out=od/f'{a.asset}_M1_normalized.csv'
 st=mat_ex(a.asset,out,START,END) if a.asset in EXNESS else asyncio.run(mat_deriv(a.asset,out,START,END));st.update(file=str(out),file_bytes=out.stat().st_size,target_start=START.isoformat(),target_end_exclusive=END.isoformat());(od/f'{a.asset}_manifest.json').write_text(json.dumps(st,indent=2));print(json.dumps(st,indent=2))
if __name__=='__main__':main()
