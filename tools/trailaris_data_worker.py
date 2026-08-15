#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

START = datetime(2026, 5, 4, tzinfo=timezone.utc)
END = datetime(2026, 8, 13, tzinfo=timezone.utc)  # exclusive; includes Aug 12

EXNESS = {
    "EURUSD":"EURUSD_Raw_Spread","GBPUSD":"GBPUSD_Raw_Spread","USDJPY":"USDJPY_Raw_Spread",
    "AUDUSD":"AUDUSD_Raw_Spread","USDCAD":"USDCAD_Raw_Spread","USDCHF":"USDCHF_Raw_Spread",
    "NZDUSD":"NZDUSD_Raw_Spread","EURJPY":"EURJPY_Raw_Spread","GBPJPY":"GBPJPY_Raw_Spread",
    "EURGBP":"EURGBP_Raw_Spread","AUDJPY":"AUDJPY_Raw_Spread","CADJPY":"CADJPY_Raw_Spread",
    "GBPCHF":"GBPCHF_Raw_Spread","XAUUSD":"XAUUSD_Raw_Spread","XAGUSD":"XAGUSD_Raw_Spread",
    "BTCUSD":"BTCUSD_Raw_Spread","ETHUSD":"ETHUSD_Raw_Spread","SOLUSD":"SOLUSD_Raw_Spread",
    "BNBUSD":"BNBUSD_Raw_Spread","XRPUSD":"XRPUSD_Raw_Spread",
    "US100":"USTEC_Raw_Spread","US500":"US500_Raw_Spread","US30":"US30_Raw_Spread",
    "GER40":"DE30_Raw_Spread","UK100":"UK100_Raw_Spread","JP225":"JP225_Raw_Spread",
    "WTI":"USOIL_Raw_Spread","BRENT":"UKOIL_Raw_Spread","NATGAS":"XNGUSD_Raw_Spread",
}
DERIV = {"V75":"R_75","V50":"R_50","V100":"R_100","V25":"R_25","V10":"R_10"}
ROUTES = list(EXNESS) + list(DERIV)

FIELDS = [
    "timestamp","open","high","low","close","spread",
    "bid_open","bid_high","bid_low","bid_close",
    "ask_open","ask_high","ask_low","ask_close",
    "spread_open","spread_high","spread_low","spread_close",
    "tick_count","source","source_symbol","raw_kind"
]

def iso_minute(ts: str) -> str:
    # Exness: 2026-05-01 00:00:00.147Z
    return ts[:16].replace(" ", "T") + ":00Z"

def dt_from_exness(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))

def new_bar(ts: str, bid: float, ask: float):
    mid = (bid + ask) / 2.0
    sp = ask - bid
    return {
        "timestamp": iso_minute(ts),
        "open": mid, "high": mid, "low": mid, "close": mid,
        "spread": sp,
        "bid_open": bid, "bid_high": bid, "bid_low": bid, "bid_close": bid,
        "ask_open": ask, "ask_high": ask, "ask_low": ask, "ask_close": ask,
        "spread_open": sp, "spread_high": sp, "spread_low": sp, "spread_close": sp,
        "tick_count": 1,
    }

def update_bar(b, bid: float, ask: float):
    mid = (bid + ask) / 2.0
    sp = ask - bid
    b["high"] = max(b["high"], mid); b["low"] = min(b["low"], mid); b["close"] = mid
    b["bid_high"] = max(b["bid_high"], bid); b["bid_low"] = min(b["bid_low"], bid); b["bid_close"] = bid
    b["ask_high"] = max(b["ask_high"], ask); b["ask_low"] = min(b["ask_low"], ask); b["ask_close"] = ask
    b["spread_high"] = max(b["spread_high"], sp); b["spread_low"] = min(b["spread_low"], sp); b["spread_close"] = sp
    b["tick_count"] += 1
    # use mean of tick spreads for causal cost burden; preserve spread OHLC separately
    b["spread"] += sp

def urlopen_retry(url: str, retries=4, timeout=120):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"TrailarisResearchMaterializer/1.0"})
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
        except Exception as e:
            last = e
        time.sleep(2 ** i)
    raise last

def download(url: str, dst: Path) -> bool:
    r = urlopen_retry(url)
    if r is None:
        return False
    with r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return True

def process_exness_zip(asset: str, sym: str, zpath: Path, writer, stats):
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if not names:
            raise RuntimeError(f"empty zip {zpath}")
        with z.open(names[0], "r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            rd = csv.reader(text)
            cur = None
            for row in rd:
                if len(row) < 5 or row[0].lower() == "exness" and row[1] == "Symbol":
                    continue
                if row[0].lower() != "exness":
                    continue
                ts = row[2]
                try:
                    d = dt_from_exness(ts)
                except Exception:
                    stats["bad_rows"] += 1
                    continue
                if d < START:
                    continue
                if d >= END:
                    break
                try:
                    bid = float(row[3]); ask = float(row[4])
                except Exception:
                    stats["bad_rows"] += 1
                    continue
                if not (bid > 0 and ask >= bid):
                    stats["bad_rows"] += 1
                    continue
                mk = iso_minute(ts)
                if cur is None or cur["timestamp"] != mk:
                    if cur is not None:
                        cur["spread"] /= cur["tick_count"]
                        cur.update(source="EXNESS_RAW_SPREAD", source_symbol=sym, raw_kind="BID_ASK_TICK_AGGREGATED_M1")
                        writer.writerow(cur); stats["bars"] += 1
                    cur = new_bar(ts, bid, ask)
                    if stats["first_ts"] is None: stats["first_ts"] = mk
                    stats["last_ts"] = mk
                else:
                    update_bar(cur, bid, ask)
                stats["ticks"] += 1
            if cur is not None:
                cur["spread"] /= cur["tick_count"]
                cur.update(source="EXNESS_RAW_SPREAD", source_symbol=sym, raw_kind="BID_ASK_TICK_AGGREGATED_M1")
                writer.writerow(cur); stats["bars"] += 1
                stats["last_ts"] = cur["timestamp"]

def exness_month_url(sym: str, y: int, m: int):
    mm = f"{m:02d}"
    return f"https://ticks.ex2archive.com/ticks/{sym}/{y}/{mm}/Exness_{sym}_{y}_{mm}.zip"

def exness_day_url(sym: str, d: datetime):
    y, m, day = d.year, d.month, d.day
    return f"https://ticks.ex2archive.com/ticks/{sym}/{y}/{m:02d}/{day:02d}/Exness_{sym}_{y}_{m:02d}_{day:02d}.zip"

def materialize_exness(asset: str, out: Path):
    sym = EXNESS[asset]
    stats = {"asset":asset,"source":"EXNESS_RAW_SPREAD","symbol":sym,"bars":0,"ticks":0,"bad_rows":0,"first_ts":None,"last_ts":None,"objects":[]}
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for m in (5,6,7):
                url = exness_month_url(sym, 2026, m)
                zpath = td / f"{asset}_{m:02d}.zip"
                ok = download(url, zpath)
                if not ok:
                    raise RuntimeError(f"required monthly archive missing: {url}")
                stats["objects"].append({"url":url,"bytes":zpath.stat().st_size})
                process_exness_zip(asset, sym, zpath, wr, stats)
                zpath.unlink(missing_ok=True)
            # August is current/incomplete: use daily objects; weekends/closed days may legitimately be absent.
            d = datetime(2026,8,1,tzinfo=timezone.utc)
            while d < END:
                url = exness_day_url(sym, d)
                zpath = td / f"{asset}_{d:%Y%m%d}.zip"
                ok = download(url, zpath)
                if ok:
                    stats["objects"].append({"url":url,"bytes":zpath.stat().st_size})
                    process_exness_zip(asset, sym, zpath, wr, stats)
                    zpath.unlink(missing_ok=True)
                d += timedelta(days=1)
    if stats["bars"] == 0:
        raise RuntimeError(f"zero materialized bars for {asset}")
    return stats

async def deriv_request(symbol: str, start: int, end: int):
    import websockets
    uri = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    async with websockets.connect(uri, open_timeout=30, close_timeout=10, ping_interval=20) as ws:
        payload = {"ticks_history":symbol,"start":start,"end":end,"style":"candles","granularity":60,"adjust_start_time":1,"count":5000,"req_id":1}
        await ws.send(json.dumps(payload))
        raw = await asyncio.wait_for(ws.recv(), timeout=45)
        p = json.loads(raw)
        if p.get("error"):
            raise RuntimeError(json.dumps(p["error"]))
        return p.get("candles", [])

async def materialize_deriv_async(asset: str, out: Path):
    symbol = DERIV[asset]
    stats = {"asset":asset,"source":"DERIV_TICKS_HISTORY","symbol":symbol,"bars":0,"first_ts":None,"last_ts":None,"chunks":0}
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS); wr.writeheader()
        s = int(START.timestamp()); final = int(END.timestamp())
        seen = set()
        while s < final:
            e = min(final-60, s + 3*86400 - 60)
            candles = None
            last = None
            for attempt in range(4):
                try:
                    candles = await deriv_request(symbol, s, e); break
                except Exception as ex:
                    last = ex; await asyncio.sleep(2 ** attempt)
            if candles is None:
                raise last
            stats["chunks"] += 1
            for c in candles:
                ep = int(c["epoch"])
                if ep in seen or ep < int(START.timestamp()) or ep >= final:
                    continue
                seen.add(ep)
                ts = datetime.fromtimestamp(ep, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:00Z")
                row = {k:"" for k in FIELDS}
                row.update(timestamp=ts,open=float(c["open"]),high=float(c["high"]),low=float(c["low"]),close=float(c["close"]),spread=0.0,tick_count="",source="DERIV_TICKS_HISTORY",source_symbol=symbol,raw_kind="DERIV_CANDLE_M1_NO_HISTORICAL_BIDASK")
                wr.writerow(row); stats["bars"] += 1
                if stats["first_ts"] is None: stats["first_ts"] = ts
                stats["last_ts"] = ts
            s += 3*86400
    if stats["bars"] == 0:
        raise RuntimeError(f"zero materialized bars for {asset}")
    return stats

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("asset", choices=ROUTES); ap.add_argument("--outdir", default="trailaris_raw")
    a = ap.parse_args(); outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{a.asset}_M1_normalized.csv"
    if a.asset in EXNESS:
        stats = materialize_exness(a.asset, out)
    else:
        stats = asyncio.run(materialize_deriv_async(a.asset, out))
    stats["file"] = str(out); stats["file_bytes"] = out.stat().st_size; stats["target_start"] = START.isoformat(); stats["target_end_exclusive"] = END.isoformat()
    (outdir / f"{a.asset}_manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
