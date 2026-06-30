from __future__ import annotations

import pandas as pd

import randyslab
from randyslab.config import CostProfile
from randyslab.engine import run_strict4h


def test_randyslab_imports() -> None:
    assert randyslab.__version__ == "0.1.0"


def test_first_bar_turnover_starts_at_zero() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
    ohlcv = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1000.0, 1000.0, 1000.0, 1000.0],
        },
        index=idx,
    )
    funding = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)
    ledger = run_strict4h(
        ohlcv,
        funding,
        {"strategy": "ma_cross", "ma_fast": 1, "ma_slow": 2, "min_hold_bars": 1},
        CostProfile(),
    )

    assert ledger["delta_exposure"].iloc[0] == 0.0
