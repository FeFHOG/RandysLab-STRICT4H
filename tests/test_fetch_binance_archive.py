from __future__ import annotations

import io

import pandas as pd

from scripts.fetch_binance import _coerce_archive_numeric, _market_id, _month_starts, _read_archive_csv, fetch_archive


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
    assert list(funding.columns) == ["timestamp", "funding_rate"]
    assert len(ohlcv) == 2
    assert len(funding) == 1
    assert float(ohlcv.iloc[0]["close"]) == 101.0
    assert float(funding.iloc[0]["funding_rate"]) == 0.0001


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
