"""Download Binance's daily metrics archive (open interest, long/short ratios) as one row per contract and day.

Input for Phase 2b (QuantumRandy docs/PHASE2B_CROSS_SECTIONAL_PROTOCOL.md). Each archive file holds 5-minute snapshots
of one contract's day; the stored row is the day's last snapshot (23:55 UTC, known at the close) plus the snapshot
count. ``--spans`` is a JSON object ``{symbol: [first_day, last_day]}``. Resumable: days already stored are skipped and
failed days are retried on the next run. Writes ``<out>/metrics_1d/<SYMBOL>.csv`` and ``<out>/manifest.json``; every
file is checksum-verified.
"""

from __future__ import annotations

import argparse
import json
import sys
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spans", required=True, help="JSON {symbol: [first_day, last_day]}")
    ap.add_argument("--out", default=str(ROOT / "data" / "metrics_1d_20261105"))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    spans = json.loads(Path(args.spans).read_text(encoding="utf-8"))
    out = Path(args.out)
    (out / "metrics_1d").mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"symbols": {}}
    print(f"{len(spans)} symbols -> {out}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_symbol, s, first, last, out): s for s, (first, last) in sorted(spans.items())}
        for done, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            try:
                manifest["symbols"][symbol] = future.result()
            except Exception as exc:  # recorded and retried on the next run
                manifest["symbols"][symbol] = {"error": str(exc)[:300]}
            if done % 25 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)} symbols", flush=True)
                _write_manifest(manifest_path, manifest)
    _write_manifest(manifest_path, manifest)
    entries = manifest["symbols"].values()
    failed = sum(len(e.get("failed_days", [])) for e in entries)
    errors = [s for s, e in manifest["symbols"].items() if "error" in e]
    print(f"done: {sum(e.get('days', 0) for e in entries)} days stored, {failed} failed days, "
          f"{len(errors)} symbols with errors", flush=True)


def fetch_symbol(symbol: str, first: str, last: str, out: Path) -> dict:
    path = out / "metrics_1d" / f"{symbol}.csv"
    stored = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["timestamp"])
    have = {str(stamp)[:10] for stamp in stored["timestamp"]}
    prefix = f"{DAILY}/{symbol}/"
    keys = [k for k in archive.list_keys(prefix, after=f"{prefix}{symbol}-metrics-{first}") if k.endswith(".zip")]
    rows, failed = [], []
    for key in keys:
        day = key.rsplit("-metrics-", 1)[-1].removesuffix(".zip")
        if day > last or day in have:
            continue
        try:
            payload = archive.download_verified(key)
            if payload is not None:
                rows.append(daily_row(day, archive.read_zip_csv(payload, COLUMNS)))
        except Exception:  # one bad day must not discard the rest
            failed.append(day)
    frame = pd.concat([stored, pd.DataFrame(rows)]) if rows else stored
    if rows:
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        partial = path.with_suffix(".csv.partial")
        frame.to_csv(partial, index=False)
        partial.replace(path)
    stamps = sorted(str(stamp)[:10] for stamp in frame["timestamp"])
    return {"days": len(stamps), "first": stamps[0] if stamps else None, "last": stamps[-1] if stamps else None,
            "failed_days": failed, "span": [first, last]}


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


def _write_manifest(path: Path, manifest: dict) -> None:
    updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest.update({"source": archive.BASE_URL, "updated_at": updated})
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
