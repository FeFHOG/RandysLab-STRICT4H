"""Download daily klines, funding and premium-index klines for every Binance USD-M USDT perpetual (incl. delisted).

Survivorship-free input for cross-sectional research. Resumable: symbols already written are skipped. Writes
``<out>/{klines_1d,funding,premium_1d}/<SYMBOL>.csv`` plus ``<out>/manifest.json``. Files are checksum-verified.
Bars from the sealed holdout are stored but loaders drop them unless a pre-registration opens the holdout.
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

UM = "data/futures/um/monthly"
DATASETS = {
    "klines_1d": f"{UM}/klines/{{symbol}}/1d/",
    "funding": f"{UM}/fundingRate/{{symbol}}/",
    "premium_1d": f"{UM}/premiumIndexKlines/{{symbol}}/1d/",
}
EXCLUDED_BASES = {"USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDP", "USDE", "PYUSD", "EUR", "GBP", "AEUR"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "data" / "universe_1d_20261105"))
    ap.add_argument("--last-month", default="2026-10", help="Last monthly file to include (YYYY-MM)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--symbols", nargs="*", help="Restrict to these symbols (default: all USDT perpetuals)")
    args = ap.parse_args()

    out = Path(args.out)
    for dataset in DATASETS:
        (out / dataset).mkdir(parents=True, exist_ok=True)
    symbols = args.symbols or usdt_perpetuals()
    print(f"{len(symbols)} symbols -> {out}", flush=True)

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"symbols": {}}
    todo = [s for s in symbols if not all((out / d / f"{s}.csv").exists() for d in DATASETS)]
    for symbol in sorted(set(symbols) - set(todo)):
        entry = manifest["symbols"].get(symbol)
        if not entry or "error" in entry:  # finished by an interrupted run before its manifest entry was saved
            manifest["symbols"][symbol] = _entry_from_files(symbol, out)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_symbol, symbol, out, args.last_month): symbol for symbol in todo}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                manifest["symbols"][symbol] = future.result()
            except Exception as exc:  # recorded and retried on the next run
                manifest["symbols"][symbol] = {"error": str(exc)[:300]}
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} symbols", flush=True)
                _write_manifest(manifest_path, manifest, args)
    _write_manifest(manifest_path, manifest, args)
    errors = [s for s, entry in manifest["symbols"].items() if "error" in entry]
    print(f"done: {len(manifest['symbols'])} symbols, {len(errors)} with errors", flush=True)


def usdt_perpetuals() -> list[str]:
    names = [prefix.rstrip("/").rsplit("/", 1)[-1] for prefix in archive.list_prefixes(f"{UM}/klines/")]
    return sorted(
        name
        for name in names
        if name.endswith("USDT") and "_" not in name and name[: -len("USDT")] not in EXCLUDED_BASES
    )


def fetch_symbol(symbol: str, out: Path, last_month: str) -> dict:
    entry: dict = {}
    for dataset, template in DATASETS.items():
        keys = [k for k in archive.list_keys(template.format(symbol=symbol)) if k.endswith(".zip")]
        keys = [k for k in keys if _month_of(k) <= last_month]
        frames = []
        missing = []
        for key in sorted(keys):
            payload = archive.download_verified(key)
            if payload is None:
                missing.append(_month_of(key))
                continue
            columns = archive.FUNDING_COLUMNS if dataset == "funding" else archive.KLINE_COLUMNS
            frames.append(_normalize(dataset, archive.read_zip_csv(payload, columns)))
        frame = pd.concat(frames).sort_values("timestamp").drop_duplicates("timestamp") if frames else _empty(dataset)
        path = out / dataset / f"{symbol}.csv"
        partial = path.with_suffix(".csv.partial")
        frame.to_csv(partial, index=False)
        partial.replace(path)  # atomic: an interrupted run never leaves a truncated CSV behind
        entry[dataset] = {
            "months": len(keys),
            "missing_months": missing,
            "rows": int(len(frame)),
            "first": str(frame["timestamp"].iloc[0]) if len(frame) else None,
            "last": str(frame["timestamp"].iloc[-1]) if len(frame) else None,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return entry


def _entry_from_files(symbol: str, out: Path) -> dict:
    entry: dict = {"rebuilt_from_files": True}
    for dataset in DATASETS:
        path = out / dataset / f"{symbol}.csv"
        frame = pd.read_csv(path, usecols=["timestamp"])
        entry[dataset] = {
            "months": None,
            "missing_months": None,
            "rows": int(len(frame)),
            "first": str(frame["timestamp"].iloc[0]) if len(frame) else None,
            "last": str(frame["timestamp"].iloc[-1]) if len(frame) else None,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return entry


def _normalize(dataset: str, raw: pd.DataFrame) -> pd.DataFrame:
    if dataset == "funding":
        frame = pd.DataFrame(
            {
                "timestamp": archive.to_utc(raw["calc_time"]).dt.floor("min"),
                "funding_rate": pd.to_numeric(raw["last_funding_rate"], errors="coerce"),
                "funding_interval_hours": pd.to_numeric(raw.get("funding_interval_hours"), errors="coerce"),
            }
        )
        return frame.dropna(subset=["funding_rate"])
    frame = pd.DataFrame({"timestamp": archive.to_utc(raw["open_time"])})
    keep = ["open", "high", "low", "close"]
    if dataset == "klines_1d":
        keep += ["volume", "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"]
    for column in keep:
        frame[column] = pd.to_numeric(raw[column], errors="coerce")
    return frame.dropna(subset=["close"])


def _empty(dataset: str) -> pd.DataFrame:
    columns = {
        "funding": ["timestamp", "funding_rate", "funding_interval_hours"],
        "klines_1d": ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "count",
                      "taker_buy_volume", "taker_buy_quote_volume"],
        "premium_1d": ["timestamp", "open", "high", "low", "close"],
    }[dataset]
    return pd.DataFrame(columns=columns)


def _month_of(key: str) -> str:
    return key.rsplit("-", 2)[-2] + "-" + key.rsplit("-", 1)[-1][:2]


def _write_manifest(path: Path, manifest: dict, args: argparse.Namespace) -> None:
    manifest.update(
        {
            "source": archive.BASE_URL,
            "last_month": args.last_month,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
