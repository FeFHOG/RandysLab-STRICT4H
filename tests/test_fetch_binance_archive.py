from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile

import pandas as pd
import pytest

import scripts.fetch_binance as fb
from scripts.fetch_binance import (
    ArchiveChecksumError,
    _coerce_archive_numeric,
    _market_id,
    _month_starts,
    _read_archive_csv,
    fetch_archive,
)

BASE = "https://data.binance.vision"
KLINE_HEADER = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"
)
FETCH_TIME = pd.Timestamp("2026-11-05T12:00:00Z")


def test_market_id_and_month_starts() -> None:
    assert _market_id("AVAX/USDT:USDT") == "AVAXUSDT"
    assert [ts.strftime("%Y-%m") for ts in _month_starts("2024-01-15", "2024-04-01")] == [
        "2024-01",
        "2024-02",
        "2024-03",
    ]


def test_fetch_archive_parses_monthly_zip_payloads(monkeypatch) -> None:
    class Args:
        symbol = "ETH/USDT:USDT"
        start = "2024-01-01"
        end = "2024-02-01"
        proxy = None
        archive_url = "https://data.binance.vision"
        timeout = 1.0
        retries = 1
        sleep = 0.0

    def fake_get_archive_csv(_session, url: str, _timeout: float, _retries: int) -> pd.DataFrame:
        if "klines" in url:
            return pd.DataFrame(
                {
                    "open_time": [1704067200000, 1704081600000],
                    "open": ["100.0", "101.0"],
                    "high": ["102.0", "103.0"],
                    "low": ["99.0", "100.0"],
                    "close": ["101.0", "102.0"],
                    "volume": ["10.0", "11.0"],
                }
            )
        return pd.DataFrame(
            {
                "calc_time": [1704067200000],
                "funding_interval_hours": [8],
                "last_funding_rate": ["0.0001"],
            }
        )

    monkeypatch.setattr("scripts.fetch_binance._get_archive_csv", fake_get_archive_csv)
    monkeypatch.setattr("scripts.fetch_binance.time.sleep", lambda _seconds: None)

    ohlcv, funding = fetch_archive(Args())

    assert list(ohlcv.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert list(funding.columns) == ["timestamp", "funding_rate", "funding_interval_hours"]
    assert len(ohlcv) == 2
    assert len(funding) == 1
    assert float(ohlcv.iloc[0]["close"]) == 101.0
    assert float(funding.iloc[0]["funding_rate"]) == 0.0001
    assert int(funding.iloc[0]["funding_interval_hours"]) == 8


def test_read_archive_csv_accepts_headerless_legacy_files() -> None:
    payload = "1577836800000,1,2,0.5,1.5,10,1577851199999,20,3,4,5,0\n"

    frame = _read_archive_csv(io.StringIO(payload))

    assert list(frame.columns)[:6] == ["open_time", "open", "high", "low", "close", "volume"]
    assert frame.iloc[0]["open_time"] == 1577836800000


def test_coerce_archive_numeric_drops_repeated_header_rows() -> None:
    frame = pd.DataFrame(
        {
            "open_time": ["open_time", "1640995200000"],
            "open": ["open", "1.0"],
            "high": ["high", "2.0"],
            "low": ["low", "0.5"],
            "close": ["close", "1.5"],
            "volume": ["volume", "10.0"],
        }
    )

    cleaned = _coerce_archive_numeric(frame, ["open_time", "open", "high", "low", "close", "volume"])

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["open_time"] == 1640995200000


# --- fake data.binance.vision (download layer patched; no network) ---------------------------------------------


def _ms(ts) -> int:
    return int(pd.Timestamp(ts, tz="UTC").value // 1_000_000)


def monthly_kline_url(ym: str, sym: str = "BTCUSDT") -> str:
    return f"{BASE}/data/futures/um/monthly/klines/{sym}/4h/{sym}-4h-{ym}.zip"


def daily_kline_url(ymd: str, sym: str = "BTCUSDT") -> str:
    return f"{BASE}/data/futures/um/daily/klines/{sym}/4h/{sym}-4h-{ymd}.zip"


def funding_url(ym: str, sym: str = "BTCUSDT") -> str:
    return f"{BASE}/data/futures/um/monthly/fundingRate/{sym}/{sym}-fundingRate-{ym}.zip"


def not_published(granularity: str, period: str, url: str) -> dict:
    """Expected manifest entry for an archive file that returned HTTP 404."""
    return {"granularity": granularity, "period": period, "url": url, "reason": "not published (HTTP 404)"}


def kline_csv(start: str, end: str, header: bool = True, drop=()) -> str:
    """4h bars opening in [start, end] (inclusive, like pd.date_range), minus the open times in ``drop``."""
    dropped = {pd.Timestamp(ts) for ts in drop}
    lines = [KLINE_HEADER] if header else []
    for i, ts in enumerate(pd.date_range(start, end, freq="4h")):
        if ts in dropped:
            continue
        open_ms = _ms(ts)
        close_ms = open_ms + 4 * 3600 * 1000 - 1
        lines.append(f"{open_ms},{100 + i},{101 + i},{99 + i},{100.5 + i},{10 + i},{close_ms},1,1,1,1,0")
    return "\n".join(lines) + "\n"


def funding_csv(rows: list[tuple], with_interval: bool = True) -> str:
    """rows: (timestamp incl. ms, rate, interval_hours)."""
    if with_interval:
        lines = ["calc_time,funding_interval_hours,last_funding_rate"]
        lines += [f"{_ms(ts)},{hours},{rate}" for ts, rate, hours in rows]
    else:
        lines = ["calc_time,last_funding_rate"]
        lines += [f"{_ms(ts)},{rate}" for ts, rate, *_ in rows]
    return "\n".join(lines) + "\n"


def eight_hourly_funding(start: str, end: str) -> list[tuple]:
    """Settlements stamped a few ms late, as Binance does."""
    return [(ts + pd.Timedelta(milliseconds=8), 0.0001, 8) for ts in pd.date_range(start, end, freq="8h")]


class FakeArchive:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.requested: list[str] = []

    def add(self, url: str, csv_text: str, checksum: str | None = None, publish_checksum: bool = True) -> bytes:
        zip_name = url.rsplit("/", 1)[-1]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(zip_name.replace(".zip", ".csv"), csv_text)
        payload = buffer.getvalue()
        self.files[url] = payload
        if publish_checksum:
            digest = checksum or hashlib.sha256(payload).hexdigest()
            self.files[f"{url}.CHECKSUM"] = f"{digest}  {zip_name}\n".encode()
        return payload

    def http_get(self, _session, url: str, _timeout: float, _retries: int) -> bytes | None:
        self.requested.append(url)
        return self.files.get(url)  # None == HTTP 404


@pytest.fixture
def archive(monkeypatch) -> FakeArchive:
    fake = FakeArchive()
    monkeypatch.setattr(fb, "_http_get", fake.http_get)
    monkeypatch.setattr(fb.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(fb, "_utc_now", lambda: FETCH_TIME)
    monkeypatch.setattr(fb, "_git_commit", lambda *_args, **_kwargs: "abc1234-dirty")
    return fake


def archive_args(start: str, end: str, **overrides) -> argparse.Namespace:
    values = {
        "symbol": "BTC/USDT:USDT",
        "file_prefix": "BTCUSDT",
        "start": start,
        "end": end,
        "outdir": "unused",
        "method": "archive",
        "base_url": "https://fapi.binance.com",
        "archive_url": BASE,
        "proxy": None,
        "timeout": 1.0,
        "retries": 1,
        "sleep": 0.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_archive_keeps_only_bars_opening_before_end(archive: FakeArchive) -> None:
    archive.add(monthly_kline_url("2024-02"), kline_csv("2024-02-01", "2024-02-29 20:00"))
    archive.add(funding_url("2024-02"), funding_csv(eight_hourly_funding("2024-02-01", "2024-02-29 16:00")))

    ohlcv, funding = fetch_archive(archive_args("2024-02-10T04:00", "2024-02-11T08:00"))

    end = pd.Timestamp("2024-02-11T08:00", tz="UTC")
    assert ohlcv["timestamp"].iloc[0] == pd.Timestamp("2024-02-10T04:00", tz="UTC")
    # The bar opening exactly at `end` (and everything after) is excluded.
    assert ohlcv["timestamp"].iloc[-1] == pd.Timestamp("2024-02-11T04:00", tz="UTC")
    assert len(ohlcv) == 7
    # 2024-02-11 08:00:00.008 floors to 08:00 == end -> excluded as well.
    expected_funding = pd.date_range("2024-02-10 08:00", "2024-02-11 00:00", freq="8h", tz="UTC")
    assert list(funding["timestamp"]) == list(expected_funding)
    assert (funding["timestamp"] < end).all() and (ohlcv["timestamp"] < end).all()


def test_monthly_kline_404_falls_back_to_daily_files(archive: FakeArchive, capsys) -> None:
    archive.add(monthly_kline_url("2024-02"), kline_csv("2024-02-01", "2024-02-29 20:00"))
    # March: no monthly file; daily files for 03-01 and 03-03 exist, 03-02 is missing too.
    archive.add(daily_kline_url("2024-03-01"), kline_csv("2024-03-01", "2024-03-01 20:00"))
    archive.add(daily_kline_url("2024-03-03"), kline_csv("2024-03-03", "2024-03-03 20:00"))
    archive.add(funding_url("2024-02"), funding_csv(eight_hourly_funding("2024-02-01", "2024-02-29 16:00")))

    result = fetch_archive(archive_args("2024-02-01", "2024-03-04"))

    timestamps = set(result.ohlcv["timestamp"])
    assert len(result.ohlcv) == 29 * 6 + 6 + 6
    assert pd.Timestamp("2024-03-01 20:00", tz="UTC") in timestamps
    assert pd.Timestamp("2024-03-03 20:00", tz="UTC") in timestamps
    assert not any(ts.strftime("%Y-%m-%d") == "2024-03-02" for ts in timestamps)
    assert result.daily_fallback_months == [not_published("month", "2024-03", monthly_kline_url("2024-03"))]
    assert result.ohlcv_missing == [not_published("day", "2024-03-02", daily_kline_url("2024-03-02"))]
    # Funding has no daily archive: the unavailable month is recorded, not silently skipped.
    assert result.funding_missing == [not_published("month", "2024-03", funding_url("2024-03"))]
    assert result.ohlcv_files_verified == 3 and result.funding_files_verified == 1
    # Daily files are only requested for days inside [start, end); none for February (monthly file existed).
    daily_requests = [url for url in archive.requested if "/daily/" in url and not url.endswith(".CHECKSUM")]
    assert daily_requests == [daily_kline_url(f"2024-03-0{d}") for d in (1, 2, 3)]
    err = capsys.readouterr().err
    assert "WARNING" in err and "falling back to daily files" in err
    assert "2024-03-02" in err and "funding 2024-03" in err


def test_monthly_fallback_respects_window_start(archive: FakeArchive) -> None:
    # Like BTCUSDT in 2019: no monthly file; daily files start late in the month.
    archive.add(daily_kline_url("2019-12-31"), kline_csv("2019-12-31", "2019-12-31 20:00", header=False))
    archive.add(funding_url("2020-01"), funding_csv(eight_hourly_funding("2020-01-01", "2020-01-01 16:00")))
    archive.add(monthly_kline_url("2020-01"), kline_csv("2020-01-01", "2020-01-31 20:00", header=False))

    result = fetch_archive(archive_args("2019-12-28", "2020-01-02"))

    assert [entry["period"] for entry in result.ohlcv_missing] == ["2019-12-28", "2019-12-29", "2019-12-30"]
    assert [entry["period"] for entry in result.funding_missing] == ["2019-12"]
    assert result.ohlcv["timestamp"].iloc[0] == pd.Timestamp("2019-12-31", tz="UTC")
    assert result.ohlcv["timestamp"].iloc[-1] == pd.Timestamp("2020-01-01 20:00", tz="UTC")
    assert result.ohlcv["timestamp"].diff().dropna().eq(pd.Timedelta(hours=4)).all()


def test_holes_in_monthly_file_are_patched_from_daily_files(archive: FakeArchive, capsys) -> None:
    whole_day = list(pd.date_range("2024-02-10", periods=6, freq="4h"))
    tail_day = list(pd.date_range("2024-02-29", periods=6, freq="4h"))
    hole = pd.Timestamp("2024-02-20 08:00")
    archive.add(
        monthly_kline_url("2024-02"), kline_csv("2024-02-01", "2024-02-29 20:00", drop=[*whole_day, hole, *tail_day])
    )
    archive.add(daily_kline_url("2024-02-10"), kline_csv("2024-02-10", "2024-02-10 20:00"))
    archive.add(daily_kline_url("2024-02-20"), kline_csv("2024-02-20", "2024-02-20 20:00", drop=[hole]))
    # 2024-02-29: daily file not published either
    archive.add(funding_url("2024-02"), funding_csv(eight_hourly_funding("2024-02-01", "2024-02-29 16:00")))

    result = fetch_archive(archive_args("2024-02-01", "2024-03-01"))

    timestamps = set(result.ohlcv["timestamp"])
    assert all(ts.tz_localize("UTC") in timestamps for ts in whole_day)
    assert hole.tz_localize("UTC") not in timestamps
    assert len(result.ohlcv) == 29 * 6 - 1 - 6
    assert result.daily_gap_fills == [{"period": "2024-02-10", "url": daily_kline_url("2024-02-10"), "bars_added": 6}]
    assert result.ohlcv_missing == [
        {
            "granularity": "day",
            "period": "2024-02-20",
            "url": daily_kline_url("2024-02-20"),
            "reason": "1 bar(s) absent from both the monthly and the daily file",
            "missing_bars": 1,
        },
        {
            "granularity": "day",
            "period": "2024-02-29",
            "url": daily_kline_url("2024-02-29"),
            "reason": "6 bar(s) absent from the monthly file; daily file not published (HTTP 404)",
            "missing_bars": 6,
        },
    ]
    err = capsys.readouterr().err
    assert "2024-02-20: 1 bar(s)" in err and "2024-02-29: 6 bar(s)" in err
    assert "duplicate" not in err  # only the absent bars are taken from the daily file


def test_incomplete_daily_file_in_fallback_month_is_recorded(archive: FakeArchive) -> None:
    hole = pd.Timestamp("2024-03-01 12:00")
    archive.add(daily_kline_url("2024-03-01"), kline_csv("2024-03-01", "2024-03-01 20:00", drop=[hole]))
    archive.add(funding_url("2024-03"), funding_csv(eight_hourly_funding("2024-03-01", "2024-03-01 16:00")))

    result = fetch_archive(archive_args("2024-03-01", "2024-03-02"))

    assert len(result.ohlcv) == 5
    assert result.daily_gap_fills == []
    assert result.ohlcv_missing == [
        {
            "granularity": "day",
            "period": "2024-03-01",
            "url": daily_kline_url("2024-03-01"),
            "reason": "1 bar(s) absent from the daily file",
            "missing_bars": 1,
        }
    ]
    assert archive.requested.count(daily_kline_url("2024-03-01")) == 1


@pytest.mark.parametrize("problem", ["corrupted_download", "checksum_not_published", "checksum_names_other_file"])
def test_archive_checksum_problems_raise(archive: FakeArchive, problem: str) -> None:
    url = monthly_kline_url("2024-02")
    archive.add(funding_url("2024-02"), funding_csv(eight_hourly_funding("2024-02-01", "2024-02-29 16:00")))
    if problem == "corrupted_download":
        payload = archive.add(url, kline_csv("2024-02-01", "2024-02-29 20:00"))
        archive.files[url] = payload[:-1] + bytes([payload[-1] ^ 0xFF])
    elif problem == "checksum_not_published":
        archive.add(url, kline_csv("2024-02-01", "2024-02-29 20:00"), publish_checksum=False)
    else:
        payload = archive.add(url, kline_csv("2024-02-01", "2024-02-29 20:00"))
        digest = hashlib.sha256(payload).hexdigest()
        archive.files[f"{url}.CHECKSUM"] = f"{digest}  BTCUSDT-4h-2024-01.zip\n".encode()

    with pytest.raises(ArchiveChecksumError):
        fetch_archive(archive_args("2024-02-01", "2024-03-01"))


def test_checksum_mismatch_message_names_file_and_digests(archive: FakeArchive) -> None:
    url = monthly_kline_url("2024-02")
    archive.add(url, kline_csv("2024-02-01", "2024-02-29 20:00"), checksum="0" * 64)
    archive.add(funding_url("2024-02"), funding_csv(eight_hourly_funding("2024-02-01", "2024-02-29 16:00")))

    with pytest.raises(ArchiveChecksumError, match="SHA-256 mismatch") as excinfo:
        fetch_archive(archive_args("2024-02-01", "2024-03-01"))

    assert url in str(excinfo.value) and "0" * 64 in str(excinfo.value)


def test_archive_funding_floors_timestamps_and_uses_interval_column(archive: FakeArchive) -> None:
    archive.add(monthly_kline_url("2024-02"), kline_csv("2024-02-01", "2024-02-29 20:00"))
    rows = [
        ("2024-02-01 00:00:00.008", 0.0001, 8),
        ("2024-02-01 08:00:00.003", 0.0002, 8),
        ("2024-02-01 16:00:00.001", 0.0003, 8),
        # switch to a 4h schedule; the file's own column says so
        ("2024-02-01 20:00:00.007", 0.0004, 4),
        ("2024-02-02 00:00:00.002", 0.0005, 4),
    ]
    archive.add(funding_url("2024-02"), funding_csv(rows))

    _, funding = fetch_archive(archive_args("2024-02-01", "2024-02-03"))

    assert [ts.strftime("%Y-%m-%d %H:%M:%S.%f") for ts in funding["timestamp"]] == [
        "2024-02-01 00:00:00.000000",
        "2024-02-01 08:00:00.000000",
        "2024-02-01 16:00:00.000000",
        "2024-02-01 20:00:00.000000",
        "2024-02-02 00:00:00.000000",
    ]
    assert funding["funding_interval_hours"].tolist() == [8, 8, 8, 4, 4]
    assert funding["funding_rate"].tolist() == [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]


def test_archive_funding_interval_column_takes_precedence_over_gaps(archive: FakeArchive) -> None:
    archive.add(monthly_kline_url("2024-02"), kline_csv("2024-02-01", "2024-02-29 20:00"))
    rows = [("2024-02-01 00:00:00.008", 0.0001, 1), ("2024-02-01 08:00:00.003", 0.0002, 1)]
    archive.add(funding_url("2024-02"), funding_csv(rows))

    _, funding = fetch_archive(archive_args("2024-02-01", "2024-02-03"))

    assert funding["funding_interval_hours"].tolist() == [1, 1]


def test_archive_funding_without_interval_column_infers_from_gaps(archive: FakeArchive) -> None:
    archive.add(monthly_kline_url("2024-02"), kline_csv("2024-02-01", "2024-02-29 20:00"))
    rows = [
        ("2024-02-01 00:00:00.008", 0.0001),
        ("2024-02-01 08:00:00.003", 0.0002),
        ("2024-02-01 16:00:00.001", 0.0003),
        ("2024-02-01 20:00:00.007", 0.0004),
        ("2024-02-01 21:00:00.002", 0.0005),
    ]
    archive.add(funding_url("2024-02"), funding_csv(rows, with_interval=False))

    _, funding = fetch_archive(archive_args("2024-02-01", "2024-02-03"))

    # gap to the previous settlement; the first row borrows the next gap
    assert funding["funding_interval_hours"].tolist() == [8, 8, 8, 4, 1]
    assert (funding["timestamp"].dt.second == 0).all() and (funding["timestamp"].dt.microsecond == 0).all()


def test_main_writes_csvs_and_manifests(archive: FakeArchive, tmp_path) -> None:
    from randyslab.data import read_funding, read_ohlcv

    archive.add(monthly_kline_url("2024-02"), kline_csv("2024-02-01", "2024-02-29 20:00"))
    archive.add(daily_kline_url("2024-03-01"), kline_csv("2024-03-01", "2024-03-01 20:00"))
    archive.add(funding_url("2024-02"), funding_csv(eight_hourly_funding("2024-02-01", "2024-02-29 16:00")))
    outdir = tmp_path / "archive_out"

    fb.main(
        [
            "--method", "archive",
            "--symbol", "BTC/USDT:USDT",
            "--file-prefix", "BTCUSDT",
            "--start", "2024-02-01",
            "--end", "2024-03-03",
            "--outdir", str(outdir),
            "--sleep", "0",
        ]
    )  # fmt: skip

    ohlcv_csv = outdir / "BTCUSDT_4h.csv"
    funding_csv_path = outdir / "BTCUSDT_funding.csv"
    manifest = json.loads((outdir / "BTCUSDT_4h.csv.manifest.json").read_text(encoding="utf-8"))
    funding_manifest = json.loads((outdir / "BTCUSDT_funding.csv.manifest.json").read_text(encoding="utf-8"))

    assert ohlcv_csv.read_bytes().splitlines()[:2] == [
        b"timestamp,open,high,low,close,volume",
        b"2024-02-01 00:00:00+00:00,100.0,101.0,99.0,100.5,10.0",
    ]
    assert b"\r\n" not in ohlcv_csv.read_bytes()
    assert funding_csv_path.read_bytes().splitlines()[:2] == [
        b"timestamp,funding_rate,funding_interval_hours",
        b"2024-02-01 00:00:00+00:00,0.0001,8",
    ]
    assert len(read_ohlcv(ohlcv_csv)) == 29 * 6 + 6
    assert len(read_funding(funding_csv_path)) == 29 * 3

    assert manifest["file"] == "BTCUSDT_4h.csv"
    assert manifest["dataset"] == "ohlcv"
    assert manifest["source"] == "archive"
    assert manifest["symbol"] == "BTCUSDT"
    assert manifest["interval"] == "4h"
    assert manifest["requested_start"] == "2024-02-01T00:00:00Z"
    assert manifest["requested_end"] == "2024-03-03T00:00:00Z"
    assert manifest["first_timestamp"] == "2024-02-01T00:00:00Z"
    assert manifest["last_timestamp"] == "2024-03-01T20:00:00Z"
    assert manifest["rows"] == 29 * 6 + 6
    assert manifest["columns"] == ["timestamp", "open", "high", "low", "close", "volume"]
    assert manifest["sha256"] == hashlib.sha256(ohlcv_csv.read_bytes()).hexdigest()
    assert manifest["missing_periods_tracked"] is True
    assert manifest["missing_periods"] == [not_published("day", "2024-03-02", daily_kline_url("2024-03-02"))]
    assert [entry["period"] for entry in manifest["daily_fallback_months"]] == ["2024-03"]
    assert manifest["daily_gap_fills"] == []
    assert manifest["archive_files_verified_sha256"] == 2
    assert manifest["fetch_time_utc"] == "2026-11-05T12:00:00Z"
    assert manifest["git_commit"] == "abc1234-dirty"

    assert funding_manifest["file"] == "BTCUSDT_funding.csv"
    assert funding_manifest["dataset"] == "funding"
    assert funding_manifest["source"] == "archive"
    assert funding_manifest["first_timestamp"] == "2024-02-01T00:00:00Z"
    assert funding_manifest["last_timestamp"] == "2024-02-29T16:00:00Z"
    assert funding_manifest["rows"] == 29 * 3
    assert funding_manifest["columns"] == ["timestamp", "funding_rate", "funding_interval_hours"]
    assert funding_manifest["sha256"] == hashlib.sha256(funding_csv_path.read_bytes()).hexdigest()
    assert funding_manifest["missing_periods"] == [not_published("month", "2024-03", funding_url("2024-03"))]
    assert "daily_fallback_months" not in funding_manifest
    assert funding_manifest["git_commit"] == "abc1234-dirty"
