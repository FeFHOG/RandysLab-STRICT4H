"""Fetch Binance USD-M perpetual 4h klines and funding rates into CSV files.

Methods:
  direct   public REST API (/fapi/v1/klines, /fapi/v1/fundingRate)
  ccxt     the same public data through ccxt binanceusdm
  archive  data.binance.vision zip archives: monthly kline files with a daily-file fallback, monthly funding files.
           Every zip is verified against its published .CHECKSUM (SHA-256).

Guarantees:
  * Only complete bars are stored: bars must open inside [start, end); API klines whose close time is after the
    fetch time (still in progress) are dropped.
  * Funding settlement timestamps are floored to the minute and carry a funding_interval_hours column.
  * Unavailable archive months/days are printed as warnings and recorded in the manifest, never silently dropped.
    Bars missing inside a published monthly kline file are patched from that day's daily file when it has them;
    whatever is still missing is recorded too.
  * Every output CSV gets a sidecar "<csv name>.manifest.json" with provenance and the SHA-256 of the CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

INTERVAL = "4h"
INTERVAL_MS = 4 * 60 * 60 * 1000
DEFAULT_FUNDING_INTERVAL_HOURS = 8
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
KLINE_NUMERIC = ["open_time", "open", "high", "low", "close", "volume"]
FUNDING_COLUMNS = ["timestamp", "funding_rate", "funding_interval_hours"]
REPO_ROOT = Path(__file__).resolve().parents[1]


class ArchiveChecksumError(RuntimeError):
    """An archive zip did not match, or could not be verified against, its published .CHECKSUM file."""


@dataclass
class FetchResult:
    """Fetched frames plus provenance.

    Iterating yields ``(ohlcv, funding)`` so ``ohlcv, funding = fetch_archive(args)`` keeps working.
    """

    ohlcv: pd.DataFrame
    funding: pd.DataFrame
    source: str  # "archive" or "api"
    fetch_time: pd.Timestamp
    ohlcv_missing: list[dict] = field(default_factory=list)
    funding_missing: list[dict] = field(default_factory=list)
    daily_fallback_months: list[dict] = field(default_factory=list)
    daily_gap_fills: list[dict] = field(default_factory=list)
    ohlcv_files_verified: int = 0
    funding_files_verified: int = 0

    def __iter__(self):
        return iter((self.ohlcv, self.funding))


# --- time helpers -----------------------------------------------------------------------------------------------


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _ms(ts) -> int:
    return int(_utc(ts).value // 1_000_000)


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").floor("s")


def _iso(ts) -> str:
    return _utc(ts).isoformat().replace("+00:00", "Z")


def _market_id(symbol: str) -> str:
    return symbol.split(":")[0].replace("/", "")


def _month_starts(start: str, end: str) -> list[pd.Timestamp]:
    start_ts = _utc(start).normalize().replace(day=1)
    end_ts = _utc(end)
    months = []
    cur = start_ts
    while cur < end_ts:
        months.append(cur)
        cur = cur + pd.DateOffset(months=1)
    return months


def _day_starts(month: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """UTC midnights of the days in ``month`` that overlap ``[start, end)``."""
    first = max(month, start.normalize())
    stop = min(month + pd.DateOffset(months=1), end)
    return list(pd.date_range(first, stop, freq="D", inclusive="left"))


def _warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def _summarize_periods(periods: list[str]) -> str:
    """Compress sorted 'YYYY-MM' or 'YYYY-MM-DD' strings into ranges, e.g. '2019-09-08..2019-09-30, 2019-11-02'."""
    if not periods:
        return ""
    monthly = len(periods[0]) == 7
    step = pd.DateOffset(months=1) if monthly else pd.Timedelta(days=1)
    fmt = "%Y-%m" if monthly else "%Y-%m-%d"
    stamps = [pd.Timestamp(p) for p in periods]
    ranges = []
    first = prev = stamps[0]
    for ts in stamps[1:]:
        if ts != prev + step:
            ranges.append((first, prev))
            first = ts
        prev = ts
    ranges.append((first, prev))
    return ", ".join(a.strftime(fmt) if a == b else f"{a.strftime(fmt)}..{b.strftime(fmt)}" for a, b in ranges)


# --- HTTP / archive download layer ------------------------------------------------------------------------------


def _session(proxy: str | None) -> requests.Session:
    session = requests.Session()
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
    return session


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


def _http_get(session: requests.Session, url: str, timeout: float, retries: int) -> bytes | None:
    """Body of ``url``, or None when it is not published (HTTP 404). Other failures are retried, then raised."""
    attempts = max(1, int(retries))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 < attempts:
                wait = min(2**attempt, 30)
                print(f"request failed ({exc}); retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
    raise RuntimeError(f"request failed after {attempts} attempts: {url}: {last_error}") from last_error


def _expected_sha256(checksum_text: str, url: str) -> str:
    """Parse a Binance '.CHECKSUM' file: '<sha256 hex>  <zip file name>'."""
    parts = checksum_text.split()
    digest = parts[0].lower() if parts else ""
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ArchiveChecksumError(f"malformed .CHECKSUM for {url}: {checksum_text[:200]!r}")
    name = url.rsplit("/", 1)[-1]
    if len(parts) > 1 and parts[1].lstrip("*") != name:
        raise ArchiveChecksumError(f".CHECKSUM for {url} refers to {parts[1]!r}, expected {name!r}")
    return digest


def _download_verified_zip(session: requests.Session, url: str, timeout: float, retries: int) -> bytes | None:
    """Archive zip verified against ``<url>.CHECKSUM``; None when the zip is not published (HTTP 404)."""
    payload = _http_get(session, url, timeout, retries)
    if payload is None:
        return None
    checksum = _http_get(session, f"{url}.CHECKSUM", timeout, retries)
    if checksum is None:
        raise ArchiveChecksumError(f"{url} is published but its .CHECKSUM is not; refusing unverified data")
    expected = _expected_sha256(checksum.decode("utf-8", errors="replace"), url)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ArchiveChecksumError(f"SHA-256 mismatch for {url}: published {expected}, downloaded {actual}")
    return payload


def _get_archive_csv(session: requests.Session, url: str, timeout: float, retries: int) -> pd.DataFrame | None:
    """Raw rows of a checksum-verified archive CSV; None when the file is not published (HTTP 404)."""
    payload = _download_verified_zip(session, url, timeout, retries)
    if payload is None:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [name for name in archive.namelist() if name.endswith(".csv")]
            if len(names) != 1:
                raise RuntimeError(f"expected exactly one CSV in {url}, found {names}")
            with archive.open(names[0]) as handle:
                return _read_archive_csv(handle)
    except (zipfile.BadZipFile, pd.errors.ParserError) as exc:
        raise RuntimeError(f"cannot parse checksum-verified archive {url}: {exc}") from exc


def _read_archive_csv(handle) -> pd.DataFrame:
    try:
        raw = pd.read_csv(handle, header=None)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
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


def _coerce_archive_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=columns).reset_index(drop=True)


def _kline_monthly_url(base: str, market_id: str, ym: str) -> str:
    return f"{base}/data/futures/um/monthly/klines/{market_id}/{INTERVAL}/{market_id}-{INTERVAL}-{ym}.zip"


def _kline_daily_url(base: str, market_id: str, ymd: str) -> str:
    return f"{base}/data/futures/um/daily/klines/{market_id}/{INTERVAL}/{market_id}-{INTERVAL}-{ymd}.zip"


def _funding_monthly_url(base: str, market_id: str, ym: str) -> str:
    return f"{base}/data/futures/um/monthly/fundingRate/{market_id}/{market_id}-fundingRate-{ym}.zip"


# --- normalisation ----------------------------------------------------------------------------------------------


def _ms_to_utc(values) -> pd.DatetimeIndex:
    return pd.to_datetime(pd.Series(values).astype("int64").to_numpy(), unit="ms", utc=True)


def _sorted_unique(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = frame.sort_values("timestamp", kind="mergesort")
    duplicated = frame["timestamp"].duplicated()
    if duplicated.any():
        _warn(f"{label}: dropped {int(duplicated.sum())} duplicate timestamp row(s)")
    return frame[~duplicated].reset_index(drop=True)


def _window(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    """Rows with ``start <= timestamp < end`` (end is exclusive, so a bar opening at ``end`` is never kept)."""
    mask = (frame["timestamp"] >= _utc(start)) & (frame["timestamp"] < _utc(end))
    return frame[mask].reset_index(drop=True)


def _missing_bar_times(timestamps: pd.Series, end) -> pd.DatetimeIndex:
    """4h open times absent between the first bar and the last bar opening before ``end``."""
    if timestamps.empty:
        return pd.DatetimeIndex([], tz="UTC")
    last_expected = (_utc(end) - pd.Timedelta(milliseconds=1)).floor(INTERVAL)
    grid = pd.date_range(timestamps.iloc[0], last_expected, freq=INTERVAL)
    return grid.difference(pd.DatetimeIndex(timestamps))


def _ohlcv_frame(raw: pd.DataFrame, label: str = "klines") -> pd.DataFrame:
    """OHLCV frame from numeric ``open_time`` (ms) and price/volume columns, sorted and de-duplicated."""
    frame = pd.DataFrame({"timestamp": _ms_to_utc(raw["open_time"])})
    for column in OHLCV_COLUMNS[1:]:
        frame[column] = raw[column].astype(float).to_numpy()
    return _sorted_unique(frame, label)


def _complete_api_klines(rows: list, start, end, fetch_time: pd.Timestamp, label: str = "klines") -> pd.DataFrame:
    """API klines without in-progress bars (close time after ``fetch_time``) or bars opening outside [start, end)."""
    raw = pd.DataFrame(
        [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), int(x[6])] for x in rows],
        columns=["open_time", *OHLCV_COLUMNS[1:], "close_time"],
    )
    in_progress = raw["close_time"] > _ms(fetch_time)
    if in_progress.any():
        count = int(in_progress.sum())
        print(f"{label}: dropped {count} in-progress bar(s) closing after fetch time {_iso(fetch_time)}")
    return _window(_ohlcv_frame(raw[~in_progress], label), start, end)


def _infer_funding_interval_hours(timestamps: pd.Series, default: int = DEFAULT_FUNDING_INTERVAL_HOURS) -> pd.Series:
    """Hours since the previous settlement (rounded); the first row uses the next gap, a lone row ``default``."""
    timestamps = pd.Series(timestamps).reset_index(drop=True)
    if len(timestamps) < 2:
        return pd.Series([default] * len(timestamps), dtype="int64")
    hours = timestamps.diff().dt.total_seconds().div(3600).round()
    hours.iloc[0] = hours.iloc[1]
    return hours.astype("int64")


def _funding_frame(time_ms, rates, interval_hours=None, label: str = "funding") -> pd.DataFrame:
    """Funding rows with minute-floored timestamps and ``funding_interval_hours``.

    Binance stamps settlements a few ms late (e.g. 08:00:00.008), hence the floor. ``interval_hours`` (the archive
    file's own column) is used where present and positive; other rows are inferred from settlement gaps.
    """
    frame = pd.DataFrame(
        {
            "timestamp": _ms_to_utc(time_ms).floor("min"),
            "funding_rate": pd.Series(rates).astype(float).to_numpy(),
            "archive_hours": (
                pd.to_numeric(pd.Series(interval_hours), errors="coerce").astype(float).to_numpy()
                if interval_hours is not None
                else float("nan")
            ),
        }
    )
    frame = _sorted_unique(frame, label)
    inferred = _infer_funding_interval_hours(frame["timestamp"]).astype(float)
    archive = frame["archive_hours"]
    frame["funding_interval_hours"] = archive.where(archive.notna() & (archive > 0), inferred).round().astype("int64")
    return frame[FUNDING_COLUMNS]


def _missing_entry(granularity: str, period: str, url: str, reason: str) -> dict:
    return {"granularity": granularity, "period": period, "url": url, "reason": reason}


# --- fetchers ---------------------------------------------------------------------------------------------------


def fetch_direct(args) -> FetchResult:
    fetch_time = _utc_now()
    session = _session(args.proxy)
    market_id = _market_id(args.symbol)
    base = args.base_url.rstrip("/")
    start_ms = _ms(args.start)
    end_ms = _ms(args.end)

    rows = []
    since = start_ms
    while since < end_ms:
        batch = _get_json(
            session,
            f"{base}/fapi/v1/klines",
            # endTime is inclusive on open time, so ask for bars opening strictly before end.
            {"symbol": market_id, "interval": INTERVAL, "startTime": since, "endTime": end_ms - 1, "limit": 1500},
            args.timeout,
            args.retries,
        )
        if not batch:
            break
        rows.extend(batch)
        since = int(batch[-1][0]) + 1
        print(f"klines: {len(rows)} rows, latest={pd.to_datetime(int(batch[-1][0]), unit='ms', utc=True)}")
        time.sleep(args.sleep)
    ohlcv = _complete_api_klines(rows, args.start, args.end, fetch_time, f"{market_id} klines")

    funding_rows = []
    since = start_ms
    while since < end_ms:
        batch = _get_json(
            session,
            f"{base}/fapi/v1/fundingRate",
            {"symbol": market_id, "startTime": since, "endTime": end_ms - 1, "limit": 1000},
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
    funding = _funding_frame(
        [int(x["fundingTime"]) for x in funding_rows],
        [float(x["fundingRate"]) for x in funding_rows],
        label=f"{market_id} funding",
    )
    return FetchResult(ohlcv, _window(funding, args.start, args.end), source="api", fetch_time=fetch_time)


def fetch_archive(args) -> FetchResult:
    fetch_time = _utc_now()
    session = _session(args.proxy)
    market_id = _market_id(args.symbol)
    base = args.archive_url.rstrip("/")
    start_ts = _utc(args.start)
    end_ts = _utc(args.end)
    result = FetchResult(pd.DataFrame(), pd.DataFrame(), source="archive", fetch_time=fetch_time)
    kline_frames: list[pd.DataFrame] = []
    funding_frames: list[pd.DataFrame] = []
    daily_days: set[str] = set()  # days already served by their daily file (monthly fallback)

    def download(url: str, dataset: str) -> tuple[pd.DataFrame | None, str]:
        frame = _get_archive_csv(session, url, args.timeout, args.retries)
        if frame is None:
            return None, "not published (HTTP 404)"
        if dataset == "ohlcv":
            result.ohlcv_files_verified += 1
        else:
            result.funding_files_verified += 1
        if frame.empty:
            return None, "published but empty"
        return frame, ""

    for month in _month_starts(args.start, args.end):
        ym = month.strftime("%Y-%m")
        kline_url = _kline_monthly_url(base, market_id, ym)
        kline, reason = download(kline_url, "ohlcv")
        if kline is not None:
            kline_frames.append(kline)
            kline_rows = len(kline)
        else:
            result.daily_fallback_months.append(_missing_entry("month", ym, kline_url, reason))
            _warn(f"{market_id} {INTERVAL} klines {ym}: monthly archive {reason}; falling back to daily files")
            kline_rows = 0
            missing_days = []
            for day in _day_starts(month, start_ts, end_ts):
                ymd = day.strftime("%Y-%m-%d")
                day_url = _kline_daily_url(base, market_id, ymd)
                daily, day_reason = download(day_url, "ohlcv")
                if daily is None:
                    result.ohlcv_missing.append(_missing_entry("day", ymd, day_url, day_reason))
                    missing_days.append(ymd)
                else:
                    kline_frames.append(daily)
                    daily_days.add(ymd)
                    kline_rows += len(daily)
                time.sleep(args.sleep)
            if missing_days:
                _warn(
                    f"{market_id} {INTERVAL} klines {ym}: {len(missing_days)} daily file(s) unavailable, recorded as "
                    f"missing: {_summarize_periods(missing_days)}"
                )

        funding_url = _funding_monthly_url(base, market_id, ym)
        funding, reason = download(funding_url, "funding")
        if funding is not None:
            funding_frames.append(funding)
        else:
            result.funding_missing.append(_missing_entry("month", ym, funding_url, reason))
            _warn(f"{market_id} funding {ym}: monthly archive {reason}; recorded as missing (no daily funding files)")
        print(f"archive {market_id} {ym}: klines={kline_rows} funding={0 if funding is None else len(funding)}")
        time.sleep(args.sleep)

    if not kline_frames:
        raise RuntimeError(f"No archive kline rows found for {market_id}.")
    if not funding_frames:
        raise RuntimeError(f"No archive funding rows found for {market_id}.")

    label = f"{market_id} {INTERVAL} klines"
    raw_ohlcv = _coerce_archive_numeric(pd.concat(kline_frames, ignore_index=True), KLINE_NUMERIC)
    ohlcv = _window(_ohlcv_frame(raw_ohlcv, label), start_ts, end_ts)

    # Published monthly files are occasionally incomplete (e.g. SOLUSDT 2022-02 lacks its last three days): patch
    # absent bars from that day's daily file and record whatever is still missing.
    reported = {entry["period"] for entry in result.ohlcv_missing}
    missing = _missing_bar_times(ohlcv["timestamp"], end_ts)
    patches = []
    for day in sorted(set(missing.normalize())):
        ymd = day.strftime("%Y-%m-%d")
        if ymd in reported:
            continue  # the day's daily file is already recorded as unavailable
        wanted = missing[missing.normalize() == day]
        day_url = _kline_daily_url(base, market_id, ymd)
        if ymd in daily_days:
            daily, why = None, "absent from the daily file"
        else:
            daily, reason = download(day_url, "ohlcv")
            time.sleep(args.sleep)
            if daily is None:
                why = f"absent from the monthly file; daily file {reason}"
            else:
                why = "absent from both the monthly and the daily file"
        added = 0
        if daily is not None:
            patch = _coerce_archive_numeric(daily, KLINE_NUMERIC)
            patch = patch[patch["open_time"].astype("int64").isin([_ms(ts) for ts in wanted])]
            patch = patch.drop_duplicates("open_time")
            added = len(patch)
            if added:
                patches.append(patch)
                result.daily_gap_fills.append({"period": ymd, "url": day_url, "bars_added": added})
        if added < len(wanted):
            remaining = len(wanted) - added
            entry = _missing_entry("day", ymd, day_url, f"{remaining} bar(s) {why}")
            result.ohlcv_missing.append({**entry, "missing_bars": remaining})
            _warn(f"{label} {ymd}: {remaining} bar(s) {why}; recorded as missing")
    if patches:
        raw_ohlcv = pd.concat([raw_ohlcv, *patches], ignore_index=True)
        ohlcv = _window(_ohlcv_frame(raw_ohlcv, label), start_ts, end_ts)
    result.ohlcv_missing.sort(key=lambda entry: entry["period"])
    result.ohlcv = ohlcv

    raw_funding = _coerce_archive_numeric(
        pd.concat(funding_frames, ignore_index=True),
        ["calc_time", "last_funding_rate"],
    )
    funding = _funding_frame(
        raw_funding["calc_time"],
        raw_funding["last_funding_rate"],
        raw_funding["funding_interval_hours"] if "funding_interval_hours" in raw_funding else None,
        label=f"{market_id} funding",
    )
    result.funding = _window(funding, start_ts, end_ts)
    return result


def fetch_ccxt(args) -> FetchResult:
    import ccxt

    fetch_time = _utc_now()
    opts = {"enableRateLimit": True, "timeout": int(args.timeout * 1000)}
    if args.proxy:
        opts["proxies"] = {"http": args.proxy, "https": args.proxy}
    ex = ccxt.binanceusdm(opts)
    market_id = _market_id(args.symbol)
    since = _ms(args.start)
    end_ms = _ms(args.end)
    rows = []
    while since < end_ms:
        batch = ex.fetch_ohlcv(args.symbol, timeframe=INTERVAL, since=since, limit=1000)
        if not batch:
            break
        # ccxt omits the close time; reconstruct it so in-progress bars can be dropped.
        rows.extend([*x[:6], int(x[0]) + INTERVAL_MS - 1] for x in batch)
        since = batch[-1][0] + 1
        time.sleep(ex.rateLimit / 1000)
    ohlcv = _complete_api_klines(rows, args.start, args.end, fetch_time, f"{market_id} klines")

    funding_rows = []
    since = _ms(args.start)
    while since < end_ms:
        batch = ex.fapiPublicGetFundingRate(
            {"symbol": market_id, "startTime": since, "endTime": end_ms - 1, "limit": 1000}
        )
        if not batch:
            break
        funding_rows.extend(batch)
        since = int(batch[-1]["fundingTime"]) + 1
        time.sleep(ex.rateLimit / 1000)
    funding = _funding_frame(
        [int(x["fundingTime"]) for x in funding_rows],
        [float(x["fundingRate"]) for x in funding_rows],
        label=f"{market_id} funding",
    )
    return FetchResult(ohlcv, _window(funding, args.start, args.end), source="api", fetch_time=fetch_time)


# --- outputs ----------------------------------------------------------------------------------------------------


def _run_git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True, timeout=30)
    return proc.stdout.decode("utf-8", errors="replace").strip()


def _git_commit(repo: Path = REPO_ROOT) -> str:
    """Short HEAD commit of the repo, suffixed '-dirty' when the worktree has uncommitted changes."""
    try:
        sha = _run_git(repo, "rev-parse", "--short", "HEAD")
        # --no-optional-locks: never take the index lock (others may be running git in this worktree).
        status = _run_git(repo, "--no-optional-locks", "status", "--porcelain")
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{sha}-dirty" if status else sha


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    # Fixed "\n" line endings keep the CSV bytes (and the manifest sha256) identical across platforms.
    frame.to_csv(path, index=False, lineterminator="\n")


def _manifest_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.name}.manifest.json")


def _build_manifest(args, result: FetchResult, dataset: str, csv_path: Path, frame: pd.DataFrame, commit: str) -> dict:
    archive = result.source == "archive"
    ohlcv = dataset == "ohlcv"
    if archive:
        source_url = args.archive_url
    elif args.method == "direct":
        source_url = args.base_url
    else:
        source_url = "ccxt:binanceusdm"
    manifest = {
        "file": csv_path.name,
        "dataset": dataset,
        "source": result.source,
        "method": args.method,
        "source_url": source_url,
        "symbol": _market_id(args.symbol),
        "interval": INTERVAL if ohlcv else "funding_settlement",
        "requested_start": _iso(args.start),
        "requested_end": _iso(args.end),
        "requested_end_exclusive": True,
        "first_timestamp": _iso(frame["timestamp"].iloc[0]) if len(frame) else None,
        "last_timestamp": _iso(frame["timestamp"].iloc[-1]) if len(frame) else None,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "sha256": _sha256_file(csv_path),
        "missing_periods_tracked": archive,
        "missing_periods": result.ohlcv_missing if ohlcv else result.funding_missing,
    }
    if archive and ohlcv:
        manifest["daily_fallback_months"] = result.daily_fallback_months
        manifest["daily_gap_fills"] = result.daily_gap_fills
    if archive:
        manifest["archive_files_verified_sha256"] = (
            result.ohlcv_files_verified if ohlcv else result.funding_files_verified
        )
    manifest["fetch_time_utc"] = _iso(result.fetch_time)
    manifest["git_commit"] = commit
    return manifest


def _write_manifest(csv_path: Path, manifest: dict) -> Path:
    path = _manifest_path(csv_path)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def _report_missing(label: str, result: FetchResult) -> None:
    if result.daily_fallback_months:
        months = [entry["period"] for entry in result.daily_fallback_months]
        _warn(f"{label} klines: monthly file unavailable, rebuilt from daily files: {_summarize_periods(months)}")
    if result.daily_gap_fills:
        days = [entry["period"] for entry in result.daily_gap_fills]
        bars = sum(entry["bars_added"] for entry in result.daily_gap_fills)
        filled = _summarize_periods(days)
        _warn(f"{label} klines: {bars} bar(s) absent from monthly files, filled from daily files: {filled}")
    if result.ohlcv_missing:
        days = [entry["period"] for entry in result.ohlcv_missing]
        _warn(f"{label} klines: bars MISSING from the archive on {len(days)} day(s): {_summarize_periods(days)}")
    if result.funding_missing:
        months = [entry["period"] for entry in result.funding_missing]
        _warn(f"{label} funding: {len(months)} month(s) MISSING from the archive: {_summarize_periods(months)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fetch Binance USD-M perpetual 4h klines and funding rates.")
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--file-prefix", default="BTCUSDT")
    ap.add_argument("--start", default="2019-09-08")
    ap.add_argument("--end", default="2025-11-24", help="Exclusive: only bars opening before this time are kept.")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--method", choices=["direct", "ccxt", "archive"], default="direct")
    ap.add_argument("--base-url", default="https://fapi.binance.com")
    ap.add_argument("--archive-url", default="https://data.binance.vision")
    ap.add_argument("--proxy", default=None, help="Example: http://127.0.0.1:7890")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args(argv)

    if args.method == "direct":
        result = fetch_direct(args)
    elif args.method == "archive":
        result = fetch_archive(args)
    else:
        result = fetch_ccxt(args)

    label = _market_id(args.symbol)
    if result.ohlcv.empty:
        raise RuntimeError(f"No complete {INTERVAL} klines for {label} in [{args.start}, {args.end}).")
    if result.funding.empty:
        _warn(f"{label}: no funding settlements in [{args.start}, {args.end}); writing an empty funding file")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    commit = _git_commit()
    outputs = [
        ("ohlcv", out / f"{args.file_prefix}_{INTERVAL}.csv", result.ohlcv),
        ("funding", out / f"{args.file_prefix}_funding.csv", result.funding),
    ]
    for dataset, csv_path, frame in outputs:
        _write_csv(frame, csv_path)
        manifest_path = _write_manifest(csv_path, _build_manifest(args, result, dataset, csv_path, frame, commit))
        span = f"{_iso(frame['timestamp'].iloc[0])} .. {_iso(frame['timestamp'].iloc[-1])}" if len(frame) else "empty"
        print(f"wrote {csv_path} ({len(frame)} rows, {span}) and {manifest_path.name}")
    _report_missing(label, result)


if __name__ == "__main__":
    main()
