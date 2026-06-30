from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests


def _ms(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


def _market_id(symbol: str) -> str:
    return symbol.split(":")[0].replace("/", "")


def _get_json(session: requests.Session, url: str, params: dict, timeout: float, retries: int):
    last_error = None
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_error = exc
            wait = min(2**attempt, 30)
            print(f"request failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"request failed after {retries} attempts: {last_error}") from last_error


def fetch_direct(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    session = requests.Session()
    if args.proxy:
        session.proxies.update({"http": args.proxy, "https": args.proxy})

    market_id = _market_id(args.symbol)
    since = _ms(args.start)
    end_ms = _ms(args.end)
    rows = []
    while since < end_ms:
        batch = _get_json(
            session,
            f"{args.base_url.rstrip('/')}/fapi/v1/klines",
            {"symbol": market_id, "interval": "4h", "startTime": since, "endTime": end_ms, "limit": 1500},
            args.timeout,
            args.retries,
        )
        if not batch:
            break
        rows.extend(batch)
        since = int(batch[-1][0]) + 1
        print(f"klines: {len(rows)} rows, latest={pd.to_datetime(batch[-1][0], unit='ms', utc=True)}")
        time.sleep(args.sleep)

    ohlcv = pd.DataFrame(
        [
            {
                "timestamp": x[0],
                "open": float(x[1]),
                "high": float(x[2]),
                "low": float(x[3]),
                "close": float(x[4]),
                "volume": float(x[5]),
            }
            for x in rows
        ]
    ).drop_duplicates("timestamp")
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], unit="ms", utc=True)

    since = _ms(args.start)
    funding_rows = []
    while since < end_ms:
        batch = _get_json(
            session,
            f"{args.base_url.rstrip('/')}/fapi/v1/fundingRate",
            {"symbol": market_id, "startTime": since, "endTime": end_ms, "limit": 1000},
            args.timeout,
            args.retries,
        )
        if not batch:
            break
        funding_rows.extend(batch)
        since = int(batch[-1]["fundingTime"]) + 1
        print(f"funding: {len(funding_rows)} rows, latest={pd.to_datetime(int(batch[-1]['fundingTime']), unit='ms', utc=True)}")
        time.sleep(args.sleep)

    funding = pd.DataFrame(
        [{"timestamp": pd.to_datetime(int(x["fundingTime"]), unit="ms", utc=True), "funding_rate": float(x["fundingRate"])} for x in funding_rows]
    ).drop_duplicates("timestamp")
    return ohlcv, funding


def fetch_ccxt(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    import ccxt

    opts = {"enableRateLimit": True, "timeout": int(args.timeout * 1000)}
    if args.proxy:
        opts["proxies"] = {"http": args.proxy, "https": args.proxy}
    ex = ccxt.binanceusdm(opts)
    since = _ms(args.start)
    end_ms = _ms(args.end)
    rows = []
    while since < end_ms:
        batch = ex.fetch_ohlcv(args.symbol, timeframe="4h", since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + 1
        time.sleep(ex.rateLimit / 1000)
    ohlcv = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).drop_duplicates("timestamp")
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], unit="ms", utc=True)

    funding_rows = []
    since = _ms(args.start)
    market_id = _market_id(args.symbol)
    while since < end_ms:
        batch = ex.fapiPublicGetFundingRate({"symbol": market_id, "startTime": since, "endTime": end_ms, "limit": 1000})
        if not batch:
            break
        funding_rows.extend(batch)
        since = int(batch[-1]["fundingTime"]) + 1
        time.sleep(ex.rateLimit / 1000)
    funding = pd.DataFrame(
        [{"timestamp": pd.to_datetime(int(x["fundingTime"]), unit="ms", utc=True), "funding_rate": float(x["fundingRate"])} for x in funding_rows]
    ).drop_duplicates("timestamp")
    return ohlcv, funding


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--file-prefix", default="BTCUSDT")
    ap.add_argument("--start", default="2019-09-08")
    ap.add_argument("--end", default="2025-11-24")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--method", choices=["direct", "ccxt"], default="direct")
    ap.add_argument("--base-url", default="https://fapi.binance.com")
    ap.add_argument("--proxy", default=None, help="Example: http://127.0.0.1:7890")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    ohlcv, funding = fetch_direct(args) if args.method == "direct" else fetch_ccxt(args)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ohlcv.to_csv(out / f"{args.file_prefix}_4h.csv", index=False)
    funding.to_csv(out / f"{args.file_prefix}_funding.csv", index=False)


if __name__ == "__main__":
    main()

