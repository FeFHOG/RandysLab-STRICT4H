"""Download Binance's daily metrics archive (open interest, long/short ratios) as one row per contract and day.

Input for Phase 2b (QuantumRandy docs/PHASE2B_CROSS_SECTIONAL_PROTOCOL.md). Each archive file holds 5-minute snapshots
of one contract's day; the stored row is the day's last snapshot (23:55 UTC, known at the close) plus the snapshot
count. ``--spans`` is a JSON object ``{symbol: [first_day, last_day]}``. Requests take seconds from the research
machine, so files are fetched with many requests in flight across all contracts and days. Resumable: days already
stored are skipped, progress is flushed every few thousand files, and failed days are retried on the next run. Writes
``<out>/metrics_1d/<SYMBOL>.csv`` and ``<out>/manifest.json``; every file is checksum-verified.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from randyslab import binance_archive as archive

DAILY = "data/futures/um/daily/metrics"
COLUMNS = [
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]
VALUES = COLUMNS[2:]
FLUSH_EVERY = 5000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spans", required=True, help="JSON {symbol: [first_day, last_day]}")
    ap.add_argument("--out", default=str(ROOT / "data" / "metrics_1d_20261105"))
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    spans = json.loads(Path(args.spans).read_text(encoding="utf-8"))
    out = Path(args.out)
    (out / "metrics_1d").mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"symbols": {}}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        listed = pool.map(lambda item: (item[0], pending_days(item[0], *item[1], out)), sorted(spans.items()))
        tasks = [(symbol, day, key) for symbol, days in listed for day, key in days]
        print(f"{len(spans)} symbols -> {out}: {len(tasks)} days to fetch", flush=True)
        rows: dict[str, list[dict]] = defaultdict(list)
        failed: dict[str, list[str]] = defaultdict(list)
        futures = {pool.submit(fetch_day, day, key): (symbol, day) for symbol, day, key in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            symbol, day = futures[future]
            try:
                row = future.result()
                if row is not None:
                    rows[symbol].append(row)
            except Exception:  # retried on the next run
                failed[symbol].append(day)
            if done % FLUSH_EVERY == 0 or done == len(futures):
                _flush(out, rows, failed, spans, manifest)
                _write_manifest(manifest_path, manifest)
                print(f"  {done}/{len(futures)} days", flush=True)
    _flush(out, rows, failed, spans, manifest, also=[s for s in spans if s not in manifest["symbols"]])
    _write_manifest(manifest_path, manifest)
    entries = manifest["symbols"].values()
    print(f"done: {sum(e.get('days', 0) for e in entries)} days stored, "
          f"{sum(len(e.get('failed_days', [])) for e in entries)} failed days", flush=True)


def pending_days(symbol: str, first: str, last: str, out: Path) -> list[tuple[str, str]]:
    """(day, key) for every archive file in the span that is not stored yet."""
    have = {str(stamp)[:10] for stamp in _stored(out, symbol)["timestamp"]}
    prefix = f"{DAILY}/{symbol}/"
    keys = [k for k in archive.list_keys(prefix, after=f"{prefix}{symbol}-metrics-{first}") if k.endswith(".zip")]
    days = [(k.rsplit("-metrics-", 1)[-1].removesuffix(".zip"), k) for k in keys]
    return [(day, key) for day, key in days if day <= last and day not in have]


def fetch_day(day: str, key: str) -> dict | None:
    payload = archive.download_verified(key)
    return None if payload is None else daily_row(day, archive.read_zip_csv(payload, COLUMNS))


def daily_row(day: str, raw: pd.DataFrame) -> dict:
    """The day's last snapshot taken before the next day begins, with the number of snapshots in the file."""
    times = pd.to_datetime(raw["create_time"], utc=True, format="mixed")
    end = pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1)
    inside = raw.assign(create_time=times)[times < end].sort_values("create_time")
    last = inside.iloc[-1] if len(inside) else None
    row = {
        "timestamp": f"{day} 00:00:00+00:00",
        "snapshot_time": str(last["create_time"]) if last is not None else None,
        "snapshots": int(len(inside)),
    }
    for column in VALUES:
        row[column] = pd.to_numeric(last[column], errors="coerce") if last is not None else None
    return row


def _stored(out: Path, symbol: str) -> pd.DataFrame:
    path = out / "metrics_1d" / f"{symbol}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["timestamp"])


def _flush(out: Path, rows: dict, failed: dict, spans: dict, manifest: dict, also: list[str] = ()) -> None:
    """Merge new rows into each symbol's CSV (atomic replace) and refresh its manifest entry."""
    for symbol in sorted(set(rows) | set(failed) | set(also)):
        frame = _stored(out, symbol)
        if rows.get(symbol):
            frame = pd.concat([frame, pd.DataFrame(rows.pop(symbol))])
            frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
            path = out / "metrics_1d" / f"{symbol}.csv"
            partial = path.with_suffix(".csv.partial")
            frame.to_csv(partial, index=False)
            partial.replace(path)
        stamps = sorted(str(stamp)[:10] for stamp in frame["timestamp"])
        manifest["symbols"][symbol] = {
            "days": len(stamps),
            "first": stamps[0] if stamps else None,
            "last": stamps[-1] if stamps else None,
            "failed_days": sorted(failed.get(symbol, [])),
            "span": spans[symbol],
        }


def _write_manifest(path: Path, manifest: dict) -> None:
    updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest.update({"source": archive.BASE_URL, "updated_at": updated})
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
