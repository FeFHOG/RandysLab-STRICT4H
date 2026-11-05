from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import types

import pandas as pd
import pytest
import requests

import scripts.fetch_binance as fb
from scripts.fetch_binance import (
    ArchiveChecksumError,
    _expected_sha256,
    _funding_frame,
    _http_get,
    _infer_funding_interval_hours,
    _summarize_periods,
    fetch_ccxt,
    fetch_direct,
)

H4_MS = 4 * 3600 * 1000


def _ms(ts) -> int:
    return int(pd.Timestamp(ts, tz="UTC").value // 1_000_000)


def api_kline(ts) -> list:
    open_ms = _ms(ts)
    return [open_ms, "100.0", "101.0", "99.0", "100.5", "10.0", open_ms + H4_MS - 1, "1", 1, "1", "1", "0"]


def api_args(start: str, end: str, method: str = "direct", **overrides) -> argparse.Namespace:
    values = {
        "symbol": "BTC/USDT:USDT",
        "file_prefix": "BTCUSDT",
        "start": start,
        "end": end,
        "outdir": "unused",
        "method": method,
        "base_url": "https://fapi.binance.com",
        "archive_url": "https://data.binance.vision",
        "proxy": None,
        "timeout": 1.0,
        "retries": 1,
        "sleep": 0.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class FakeBinanceApi:
    """Exchange state at fetch time. Deliberately ignores endTime to model a server returning the end bar."""

    def __init__(self, last_open: str, funding_times: list[str]) -> None:
        self.klines = [api_kline(ts) for ts in pd.date_range("2024-01-01", last_open, freq="4h")]
        self.funding = [{"fundingTime": _ms(ts), "fundingRate": "0.0001"} for ts in funding_times]
        self.calls: list[tuple[str, dict]] = []

    def get_json(self, _session, url: str, params: dict, _timeout: float, _retries: int) -> list:
        self.calls.append((url, dict(params)))
        if url.endswith("/fapi/v1/klines"):
            return [row for row in self.klines if row[0] >= params["startTime"]][: params["limit"]]
        return [row for row in self.funding if row["fundingTime"] >= params["startTime"]][: params["limit"]]


@pytest.fixture
def no_sleep(monkeypatch) -> None:
    monkeypatch.setattr(fb.time, "sleep", lambda _seconds: None)


FUNDING_TIMES = [
    "2024-01-01 00:00:00.008",
    "2024-01-01 08:00:00.003",
    "2024-01-01 16:00:00.001",
    "2024-01-02 00:00:00.009",
    "2024-01-02 04:00:00.004",
    "2024-01-02 08:00:00.002",
]


def test_direct_drops_in_progress_bar(monkeypatch, no_sleep) -> None:
    # Fetching at 13:00 while the 12:00 bar is still open; requested end lies in the future.
    monkeypatch.setattr(fb, "_utc_now", lambda: pd.Timestamp("2024-01-02T13:00:00Z"))
    api = FakeBinanceApi(last_open="2024-01-02 12:00", funding_times=FUNDING_TIMES)
    monkeypatch.setattr(fb, "_get_json", api.get_json)

    result = fetch_direct(api_args("2024-01-01", "2024-01-03"))

    assert result.source == "api"
    assert result.ohlcv["timestamp"].iloc[-1] == pd.Timestamp("2024-01-02 08:00", tz="UTC")
    assert len(result.ohlcv) == 9
    assert result.ohlcv["timestamp"].diff().dropna().eq(pd.Timedelta(hours=4)).all()


def test_direct_excludes_bar_opening_at_end(monkeypatch, no_sleep) -> None:
    monkeypatch.setattr(fb, "_utc_now", lambda: pd.Timestamp("2024-02-01T00:00:00Z"))
    api = FakeBinanceApi(last_open="2024-01-03 20:00", funding_times=FUNDING_TIMES)
    monkeypatch.setattr(fb, "_get_json", api.get_json)

    ohlcv, funding = fetch_direct(api_args("2024-01-01", "2024-01-02T08:00"))

    end_ms = _ms("2024-01-02T08:00")
    assert ohlcv["timestamp"].iloc[-1] == pd.Timestamp("2024-01-02 04:00", tz="UTC")
    kline_calls = [params for url, params in api.calls if url.endswith("/fapi/v1/klines")]
    assert all(params["endTime"] == end_ms - 1 for params in kline_calls)
    # funding: floored to the minute, 08:00:00.002 floors onto `end` and is excluded
    assert [ts.strftime("%Y-%m-%d %H:%M:%S.%f") for ts in funding["timestamp"]] == [
        "2024-01-01 00:00:00.000000",
        "2024-01-01 08:00:00.000000",
        "2024-01-01 16:00:00.000000",
        "2024-01-02 00:00:00.000000",
        "2024-01-02 04:00:00.000000",
    ]
    assert funding["funding_interval_hours"].tolist() == [8, 8, 8, 8, 4]


def test_ccxt_drops_in_progress_bar_and_bars_after_end(monkeypatch, no_sleep) -> None:
    monkeypatch.setattr(fb, "_utc_now", lambda: pd.Timestamp("2024-01-02T13:00:00Z"))
    bars = [[_ms(ts), 1.0, 2.0, 0.5, 1.5, 10.0] for ts in pd.date_range("2024-01-01", "2024-01-02 12:00", freq="4h")]
    funding_rows = [{"fundingTime": str(_ms(ts)), "fundingRate": "0.0001"} for ts in FUNDING_TIMES]

    class FakeExchange:
        rateLimit = 0

        def __init__(self, _opts) -> None:
            pass

        def fetch_ohlcv(self, _symbol, timeframe, since, limit):
            assert timeframe == "4h"
            return [bar for bar in bars if bar[0] >= since][:limit]

        def fapiPublicGetFundingRate(self, params):  # ccxt's generated method name
            return [row for row in funding_rows if int(row["fundingTime"]) >= params["startTime"]]

    monkeypatch.setitem(sys.modules, "ccxt", types.SimpleNamespace(binanceusdm=FakeExchange))

    in_progress = fetch_ccxt(api_args("2024-01-01", "2024-01-03", method="ccxt"))
    assert in_progress.ohlcv["timestamp"].iloc[-1] == pd.Timestamp("2024-01-02 08:00", tz="UTC")

    bounded = fetch_ccxt(api_args("2024-01-01", "2024-01-01T12:00", method="ccxt"))
    assert bounded.ohlcv["timestamp"].iloc[-1] == pd.Timestamp("2024-01-01 08:00", tz="UTC")
    assert bounded.funding["timestamp"].iloc[-1] == pd.Timestamp("2024-01-01 08:00", tz="UTC")


def test_main_direct_writes_api_manifest(monkeypatch, no_sleep, tmp_path) -> None:
    monkeypatch.setattr(fb, "_utc_now", lambda: pd.Timestamp("2024-01-02T13:00:00Z"))
    monkeypatch.setattr(fb, "_git_commit", lambda *_args, **_kwargs: "abc1234")
    api = FakeBinanceApi(last_open="2024-01-02 12:00", funding_times=FUNDING_TIMES)
    monkeypatch.setattr(fb, "_get_json", api.get_json)

    fb.main(["--start", "2024-01-01", "--end", "2024-01-03", "--outdir", str(tmp_path), "--sleep", "0"])

    manifest = json.loads((tmp_path / "BTCUSDT_4h.csv.manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == "api" and manifest["method"] == "direct"
    assert manifest["last_timestamp"] == "2024-01-02T08:00:00Z"
    assert manifest["missing_periods_tracked"] is False and manifest["missing_periods"] == []
    assert "daily_fallback_months" not in manifest
    assert manifest["fetch_time_utc"] == "2024-01-02T13:00:00Z"
    funding = pd.read_csv(tmp_path / "BTCUSDT_funding.csv")
    assert list(funding.columns) == ["timestamp", "funding_rate", "funding_interval_hours"]
    assert funding["timestamp"].iloc[0] == "2024-01-01 00:00:00+00:00"


def test_funding_frame_floors_to_minute_and_infers_intervals() -> None:
    times = [
        "2024-01-01 00:00:00.008",
        "2024-01-01 08:00:00.003",
        "2024-01-01 15:59:59.999",  # early stamp floors to 15:59 -> 7.98h rounds to 8
        "2024-01-01 20:00:00.001",
        "2024-01-01 21:00:00.005",
    ]

    frame = _funding_frame([_ms(t) for t in times], [0.1, 0.2, 0.3, 0.4, 0.5])

    assert list(frame.columns) == ["timestamp", "funding_rate", "funding_interval_hours"]
    assert [ts.strftime("%H:%M:%S.%f") for ts in frame["timestamp"]] == [
        "00:00:00.000000",
        "08:00:00.000000",
        "15:59:00.000000",
        "20:00:00.000000",
        "21:00:00.000000",
    ]
    assert frame["funding_interval_hours"].tolist() == [8, 8, 8, 4, 1]
    assert str(frame["funding_interval_hours"].dtype) == "int64"


def test_funding_frame_prefers_file_interval_and_fills_gaps_by_inference() -> None:
    times = ["2024-01-01 00:00", "2024-01-01 08:00", "2024-01-01 12:00", "2024-01-01 16:00"]

    frame = _funding_frame([_ms(t) for t in times], [0.0] * 4, interval_hours=["8", None, "4", ""])

    assert frame["funding_interval_hours"].tolist() == [8, 8, 4, 4]


def test_funding_frame_sorts_and_drops_duplicate_settlements(capsys) -> None:
    times = ["2024-01-01 08:00:00.003", "2024-01-01 00:00:00.008", "2024-01-01 08:00:00.009"]

    frame = _funding_frame([_ms(t) for t in times], [0.2, 0.1, 0.3])

    assert frame["funding_rate"].tolist() == [0.1, 0.2]
    assert "duplicate" in capsys.readouterr().err


def test_infer_funding_interval_edge_cases() -> None:
    assert _infer_funding_interval_hours(pd.Series([], dtype="datetime64[us, UTC]")).tolist() == []
    single = pd.Series([pd.Timestamp("2024-01-01", tz="UTC")])
    assert _infer_funding_interval_hours(single).tolist() == [8]
    pair = pd.Series(pd.to_datetime(["2024-01-01 00:00", "2024-01-01 04:00"], utc=True))
    assert _infer_funding_interval_hours(pair).tolist() == [4, 4]


def test_expected_sha256_parsing() -> None:
    url = "https://data.binance.vision/x/BTCUSDT-4h-2024-01.zip"
    digest = "AB" * 32
    assert _expected_sha256(f"{digest}  BTCUSDT-4h-2024-01.zip\n", url) == digest.lower()
    assert _expected_sha256(f"{digest} *BTCUSDT-4h-2024-01.zip", url) == digest.lower()
    assert _expected_sha256(digest, url) == digest.lower()
    with pytest.raises(ArchiveChecksumError):
        _expected_sha256("not-a-digest  BTCUSDT-4h-2024-01.zip", url)
    with pytest.raises(ArchiveChecksumError):
        _expected_sha256(f"{digest}  ETHUSDT-4h-2024-01.zip", url)


class FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def get(self, _url, timeout):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_http_get_404_is_missing_without_retry(no_sleep) -> None:
    session = FakeSession([FakeResponse(404)])
    assert _http_get(session, "https://example.invalid/a.zip", 1.0, 5) is None
    assert session.calls == 1


def test_http_get_retries_transient_failures(no_sleep) -> None:
    session = FakeSession([requests.ConnectionError("reset"), FakeResponse(503), FakeResponse(200, b"payload")])
    assert _http_get(session, "https://example.invalid/a.zip", 1.0, 5) == b"payload"
    assert session.calls == 3


def test_http_get_raises_after_exhausting_retries(no_sleep) -> None:
    session = FakeSession([FakeResponse(500), FakeResponse(500)])
    with pytest.raises(RuntimeError, match="after 2 attempts"):
        _http_get(session, "https://example.invalid/a.zip", 1.0, 2)


def test_summarize_periods_compresses_runs() -> None:
    assert _summarize_periods(["2019-09-08", "2019-09-09", "2019-09-10", "2019-09-12"]) == (
        "2019-09-08..2019-09-10, 2019-09-12"
    )
    assert _summarize_periods(["2019-11", "2019-12", "2020-01", "2020-03"]) == "2019-11..2020-01, 2020-03"
    assert _summarize_periods([]) == ""


def test_git_commit_marks_dirty_worktree(monkeypatch) -> None:
    def fake_run(cmd, **_kwargs):
        stdout = b"abc1234\n" if "rev-parse" in cmd else b" M scripts/fetch_binance.py\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(fb.subprocess, "run", fake_run)
    assert fb._git_commit() == "abc1234-dirty"


def test_git_commit_clean_and_unavailable(monkeypatch) -> None:
    def clean_run(cmd, **_kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=b"abc1234\n" if "rev-parse" in cmd else b"", stderr=b"")

    monkeypatch.setattr(fb.subprocess, "run", clean_run)
    assert fb._git_commit() == "abc1234"

    def missing_git(cmd, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(fb.subprocess, "run", missing_git)
    assert fb._git_commit() == "unknown"


def test_git_commit_against_this_repo() -> None:
    assert re.fullmatch(r"[0-9a-f]{4,40}(-dirty)?|unknown", fb._git_commit())
