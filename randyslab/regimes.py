"""Labelled market-stress regimes. Report results with and without them, so one event cannot carry a verdict."""

from __future__ import annotations

import pandas as pd

REGIMES = {
    # 2025-10-10/11: the largest liquidation cascade on record (> $19B) after the China-tariff announcement.
    "liquidation_cascade_2025_10_10": ("2025-10-10T00:00:00Z", "2025-10-12T00:00:00Z"),
    # 2026-10-29 20:05 UTC: Binance hot-wallet theft (~$6.3B); withdrawals halted, then reopened in phases.
    "binance_hot_wallet_incident_2026_10_29": ("2026-10-29T20:00:00Z", "2026-11-06T00:00:00Z"),
}


def regime_labels(index: pd.DatetimeIndex) -> pd.Series:
    """Regime name per bar ("" outside every labelled regime)."""
    labels = pd.Series("", index=index, dtype=object)
    for name, (start, end) in REGIMES.items():
        labels[(index >= pd.Timestamp(start)) & (index < pd.Timestamp(end))] = name
    return labels


def outside_regimes(index: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask of bars outside every labelled regime."""
    return regime_labels(index).eq("")
