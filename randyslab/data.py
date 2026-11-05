from __future__ import annotations

from pathlib import Path

import pandas as pd

from .accounting import funding_feature, funding_settled_per_bar, normalize_funding_events

REQUIRED_OHLCV = {"timestamp", "open", "high", "low", "close", "volume"}


def read_ohlcv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_OHLCV - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV CSV missing columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    return df.set_index("timestamp")


def read_funding(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df or "funding_rate" not in df:
        raise ValueError("Funding CSV must contain timestamp,funding_rate")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    columns = ["funding_rate"] + (["funding_interval_hours"] if "funding_interval_hours" in df.columns else [])
    return df.set_index("timestamp")[columns]


def align_funding_to_ohlcv(ohlcv: pd.DataFrame, funding: pd.DataFrame) -> pd.Series:
    """Funding feature per bar: the last rate settled at or before the bar open, as an 8h-equivalent rate."""
    return funding_feature(ohlcv.index, normalize_funding_events(funding))


def load_market(
    ohlcv_path: str | Path,
    funding_path: str | Path,
    bar_hours: float = 4.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """OHLCV with a per-bar ``funding_settled`` cost column, plus the funding feature series.

    The engine charges funding from ``funding_settled`` (exact settlements), while signals only see the feature.
    """
    ohlcv = read_ohlcv(ohlcv_path)
    events = normalize_funding_events(read_funding(funding_path))
    ohlcv["funding_settled"] = funding_settled_per_bar(ohlcv.index, events, bar_hours)
    return ohlcv, funding_feature(ohlcv.index, events)


def slice_window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = df
    if start:
        out = out[out.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        out = out[out.index < pd.Timestamp(end, tz="UTC")]
    return out.copy()
