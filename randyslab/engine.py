from __future__ import annotations

import pandas as pd

from .accounting import build_ledger, infer_bar_hours, legacy_funding_settled
from .config import CostProfile
from .signals import DEFAULT_THETA, make_signal


def run_strict4h(
    ohlcv: pd.DataFrame,
    funding_rate: pd.Series,
    theta: dict,
    costs: CostProfile,
    execution_delay_bars: int = 1,
) -> pd.DataFrame:
    p = DEFAULT_THETA | theta
    signal = make_signal(ohlcv, funding_rate, p)
    return run_strict4h_signal(
        ohlcv,
        funding_rate,
        signal,
        p,
        costs,
        execution_delay_bars=execution_delay_bars,
    )


def run_strict4h_signal(
    ohlcv: pd.DataFrame,
    funding_rate: pd.Series,
    signal: pd.Series,
    theta: dict,
    costs: CostProfile,
    execution_delay_bars: int = 1,
) -> pd.DataFrame:
    """Strict next-bar ledger for a {-1, 0, 1} signal; position size is ``signal * max_exposure_abs``.

    Funding is charged from ``ohlcv["funding_settled"]`` (see ``randyslab.data.load_market``). Frames without that
    column fall back to the legacy pro-rata approximation.
    """
    p = DEFAULT_THETA | theta
    signal = signal.reindex(ohlcv.index).fillna(0).astype("int8")
    size = abs(float(p.get("max_exposure_abs", 1.0) or 1.0))
    exposure = signal.shift(execution_delay_bars).fillna(0.0).astype(float) * size

    funding = funding_rate.reindex(ohlcv.index).fillna(0.0)
    if "funding_settled" in ohlcv.columns:
        settled = ohlcv["funding_settled"]
    else:
        settled = legacy_funding_settled(funding, infer_bar_hours(ohlcv.index))
    ledger = build_ledger(ohlcv["close"], exposure, settled, costs)

    return pd.concat(
        [
            pd.DataFrame(
                {
                    "open": ohlcv["open"],
                    "high": ohlcv["high"],
                    "low": ohlcv["low"],
                    "close": ohlcv["close"].astype(float),
                    "funding_rate": funding,
                    "signal": signal,
                },
                index=ohlcv.index,
            ),
            ledger,
        ],
        axis=1,
    )
