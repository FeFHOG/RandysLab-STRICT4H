from __future__ import annotations

import argparse
import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests


def _ms(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


def _market_id(symbol: str) -> str:
    return symbol.split(":")[0].replace("/", "")


def _month_starts(start: str, end: str) -> list[pd.Timestamp]:
    start_ts = pd.Timestamp(start, tz="UTC").replace(day=1)
    end_ts = pd.Timestamp(end, tz="UTC")
    months = []
    cur = start_ts
    while cur < end_ts:
        months.append(cur)
        cur = cur + pd.DateOffset(months=1)
    return months


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


def _get_archive_csv(session: requests.Session, url: str, timeout: float, retries: int) -> pd.DataFrame:
    last_error = None
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 404:
                return pd.DataFrame()
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
                names = [name for name in archive.namelist() if name.endswith(".csv")]
                if not names:
                    return pd.DataFrame()
                with archive.open(names[0]) as handle:
                    return _read_archive_csv(handle)
        except (requests.RequestException, zipfile.BadZipFile, pd.errors.ParserError) as exc:
            last_error = exc
            wait = min(2**attempt, 30)
            print(f"archive request failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"archive request failed after {retries} attempts: {last_error}") from last_error


def _read_archive_csv(handle) -> pd.DataFrame:
    raw = pd.read_csv(handle, header=None)
    if raw.empty:
        return raw
    first_value = str(raw.iloc[0, 0])
    if first_value in {"open_time", "calc_time"}:
        raw.columns = [str(value) for value in raw.iloc[0]]
        return raw.iloc[1:].reset_index(drop=True)
    if raw.shape[1] == 12:
        raw.columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
    elif raw.shape[1] == 3:
        raw.columns = ["calc_time", "funding_interval_hours", "last_funding_rate"]
    return raw


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
        latest = pd.to_datetime(int(batch[-1]["fundingTime"]), unit="ms", utc=True)
        print(f"funding: {len(funding_rows)} rows, latest={latest}")
        time.sleep(args.sleep)

    funding = pd.DataFrame(
        [
            {
                "timestamp": pd.to_datetime(int(x["fundingTime"]), unit="ms", utc=True),
                "funding_rate": float(x["fundingRate"]),
            }
            for x in funding_rows
        ]
    ).drop_duplicates("timestamp")
    return ohlcv, funding


def fetch_archive(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    session = requests.Session()
    if args.proxy:
        session.proxies.update({"http": args.proxy, "https": args.proxy})

    market_id = _market_id(args.symbol)
    kline_frames = []
    funding_frames = []
    for month in _month_starts(args.start, args.end):
        ym = month.strftime("%Y-%m")
        kline_url = (
            f"{args.archive_url.rstrip('/')}/data/futures/um/monthly/klines/"
            f"{market_id}/4h/{market_id}-4h-{ym}.zip"
        )
        funding_url = (
            f"{args.archive_url.rstrip('/')}/data/futures/um/monthly/fundingRate/"
            f"{market_id}/{market_id}-fundingRate-{ym}.zip"
        )
        kline = _get_archive_csv(session, kline_url, args.timeout, args.retries)
        funding = _get_archive_csv(session, funding_url, args.timeout, args.retries)
        if not kline.empty:
            kline_frames.append(kline)
        if not funding.empty:
            funding_frames.append(funding)
        print(f"archive {market_id} {ym}: klines={len(kline)} funding={len(funding)}")
        time.sleep(args.sleep)

    if not kline_frames:
        raise RuntimeError(f"No archive kline rows found for {market_id}.")
    if not funding_frames:
        raise RuntimeError(f"No archive funding rows found for {market_id}.")

    raw_ohlcv = _coerce_archive_numeric(
        pd.concat(kline_frames, ignore_index=True),
        ["open_time", "open", "high", "low", "close", "volume"],
    )
    raw_funding = _coerce_archive_numeric(
        pd.concat(funding_frames, ignore_index=True),
        ["calc_time", "last_funding_rate"],
    )
    ohlcv = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw_ohlcv["open_time"], unit="ms", utc=True),
            "open": raw_ohlcv["open"].astype(float),
            "high": raw_ohlcv["high"].astype(float),
            "low": raw_ohlcv["low"].astype(float),
            "close": raw_ohlcv["close"].astype(float),
            "volume": raw_ohlcv["volume"].astype(float),
        }
    ).drop_duplicates("timestamp")
    funding = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw_funding["calc_time"], unit="ms", utc=True),
            "funding_rate": raw_funding["last_funding_rate"].astype(float),
        }
    ).drop_duplicates("timestamp")

    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")
    ohlcv = ohlcv[(ohlcv["timestamp"] >= start_ts) & (ohlcv["timestamp"] < end_ts)].sort_values("timestamp")
    funding = funding[(funding["timestamp"] >= start_ts) & (funding["timestamp"] < end_ts)].sort_values("timestamp")
    return ohlcv, funding


def _coerce_archive_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=columns).reset_index(drop=True)


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
    ohlcv = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).drop_duplicates(
        "timestamp"
    )
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
        [
            {
                "timestamp": pd.to_datetime(int(x["fundingTime"]), unit="ms", utc=True),
                "funding_rate": float(x["fundingRate"]),
            }
            for x in funding_rows
        ]
    ).drop_duplicates("timestamp")
    return ohlcv, funding


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--file-prefix", default="BTCUSDT")
    ap.add_argument("--start", default="2019-09-08")
    ap.add_argument("--end", default="2025-11-24")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--method", choices=["direct", "ccxt", "archive"], default="direct")
    ap.add_argument("--base-url", default="https://fapi.binance.com")
    ap.add_argument("--archive-url", default="https://data.binance.vision")
    ap.add_argument("--proxy", default=None, help="Example: http://127.0.0.1:7890")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    if args.method == "direct":
        ohlcv, funding = fetch_direct(args)
    elif args.method == "archive":
        ohlcv, funding = fetch_archive(args)
    else:
        ohlcv, funding = fetch_ccxt(args)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ohlcv.to_csv(out / f"{args.file_prefix}_4h.csv", index=False)
    funding.to_csv(out / f"{args.file_prefix}_funding.csv", index=False)


if __name__ == "__main__":
    main()

