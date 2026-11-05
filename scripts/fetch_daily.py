"""Extend the universe download with Binance's daily archive: daily klines from 2026-11-01 and monthly funding files.

Forward observations (QuantumRandy docs/PHASE2A_TAKER_FORWARD_PREREGISTRATION.md) need bars after the monthly download
of ``scripts/fetch_universe.py`` (through 2026-10). The daily archive publishes each day's klines the next morning;
funding is only published monthly, a few days after a month ends. Idempotent: every run lists each symbol's daily files
from ``--first-day`` and downloads the ones not yet stored (so late or out-of-order publications are picked up), and
fetches each published month's funding once. Writes ``<out>/{klines_1d,funding}/<SYMBOL>.csv`` and
``<out>/manifest.json``; every file is checksum-verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from randyslab import binance_archive as archive

DAILY = "data/futures/um/daily/klines"
MONTHLY_FUNDING = "data/futures/um/monthly/fundingRate"
FIRST_DAY = "2026-11-01"
REFERENCE = "BTCUSDT"  # its funding file marks a month as published


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "data" / "forward_1d"))
    ap.add_argument("--first-day", default=FIRST_DAY)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--symbols", nargs="*", help="Restrict to these symbols (default: all USDT perpetuals)")
    args = ap.parse_args()

    out = Path(args.out)
    for dataset in ("klines_1d", "funding"):
        (out / dataset).mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"symbols": {}}
    symbols = args.symbols or sorted(
        name
        for name in (prefix.rstrip("/").rsplit("/", 1)[-1] for prefix in archive.list_prefixes(f"{DAILY}/"))
        if archive.is_usdt_perpetual(name)
    )
    months = published_funding_months(args.first_day)
    print(f"{len(symbols)} symbols -> {out}; funding published for {months or 'no month yet'}", flush=True)

    added = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(update_symbol, symbol, out, args.first_day, months, manifest["symbols"].get(symbol, {})): symbol
            for symbol in symbols
        }
        for done, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                entry, new_days = future.result()
                manifest["symbols"][symbol] = entry
                added += new_days
            except Exception as exc:  # recorded and retried on the next run
                manifest["symbols"][symbol] = {**manifest["symbols"].get(symbol, {}), "error": str(exc)[:300]}
            if done % 100 == 0:
                print(f"  {done}/{len(symbols)} symbols", flush=True)

    entries = manifest["symbols"].values()
    manifest.update(
        {
            "source": archive.BASE_URL,
            "first_day": args.first_day,
            "latest_day": max((e["last"] for e in entries if e.get("last")), default=None),
            "funding_months": months,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    errors = [s for s, e in manifest["symbols"].items() if "error" in e]
    print(f"done: {added} new daily bars, latest day {manifest['latest_day']}, {len(errors)} symbols with errors")
    if errors:
        print("  errors: " + ", ".join(sorted(errors)[:20]))


def published_funding_months(first_day: str) -> list[str]:
    """Months from ``first_day`` whose funding file the archive has published (checked on the reference symbol)."""
    first = pd.Timestamp(first_day).to_period("M")
    current = pd.Timestamp.now(tz="UTC").tz_localize(None).to_period("M")
    months = []
    for period in pd.period_range(first, current - 1, freq="M"):
        key = f"{MONTHLY_FUNDING}/{REFERENCE}/{REFERENCE}-fundingRate-{period}.zip"
        if archive.list_keys(key):
            months.append(str(period))
    return months


def update_symbol(symbol: str, out: Path, first_day: str, months: list[str], entry: dict) -> tuple[dict, int]:
    path = out / "klines_1d" / f"{symbol}.csv"
    stored = _read(path)
    have = set(stored["timestamp"].dt.strftime("%Y-%m-%d")) if len(stored) else set()
    prefix = f"{DAILY}/{symbol}/1d/"
    keys = [k for k in archive.list_keys(prefix, after=f"{prefix}{symbol}-1d-{first_day}") if k.endswith(".zip")]
    frames = [stored] if len(stored) else []
    new_days = 0
    for key in keys:
        if _day_of(key) in have:
            continue
        payload = archive.download_verified(key)
        if payload is None:
            continue
        frames.append(archive.normalize_klines(archive.read_zip_csv(payload, archive.KLINE_COLUMNS)))
        new_days += 1
    if new_days:
        _write(path, pd.concat(frames))

    fetched = set(entry.get("funding_months", []))
    trading_months = {day[:7] for day in have | {_day_of(k) for k in keys}}
    funding_path = out / "funding" / f"{symbol}.csv"
    funding = [_read(funding_path)] if funding_path.exists() else []
    for month in months:
        if month in fetched or month not in trading_months:
            continue
        payload = archive.download_verified(f"{MONTHLY_FUNDING}/{symbol}/{symbol}-fundingRate-{month}.zip")
        if payload is not None:
            funding.append(archive.normalize_funding(archive.read_zip_csv(payload, archive.FUNDING_COLUMNS)))
        fetched.add(month)
    if funding and len(fetched) > len(entry.get("funding_months", [])):
        _write(funding_path, pd.concat(funding))

    klines = _read(path)
    result = {
        "days": int(len(klines)),
        "first": str(klines["timestamp"].iloc[0].date()) if len(klines) else None,
        "last": str(klines["timestamp"].iloc[-1].date()) if len(klines) else None,
        "funding_months": sorted(fetched),
        "sha256": {
            name: hashlib.sha256(p.read_bytes()).hexdigest()
            for name, p in (("klines_1d", path), ("funding", funding_path))
            if p.exists()
        },
    }
    return result, new_days


def _day_of(key: str) -> str:
    return key.rsplit("-1d-", 1)[-1].removesuffix(".zip")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp"])
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="mixed")
    return frame


def _write(path: Path, frame: pd.DataFrame) -> None:
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    partial = path.with_suffix(".csv.partial")
    frame.to_csv(partial, index=False)
    partial.replace(path)  # atomic: an interrupted run never leaves a truncated CSV behind


if __name__ == "__main__":
    main()
