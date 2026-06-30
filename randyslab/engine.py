from __future__ import annotations

import numpy as np
import pandas as pd

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
    exposure = signal.shift(execution_delay_bars).fillna(0.0).astype(float)
    cap = float(p.get("max_exposure_abs", 1.0) or 1.0)
    exposure = exposure.clip(lower=-abs(cap), upper=abs(cap))

    close = ohlcv["close"].astype(float)
    r_mkt = close.pct_change().fillna(0.0)
    r_raw = exposure * r_mkt

    delta = exposure.diff().fillna(0.0)
    turnover = delta.abs()
    fee_bps = costs.taker_bps * costs.fee_multiplier
    slip_bps = costs.slippage_bps * costs.slippage_multiplier
    c_fee = turnover * fee_bps / 10_000.0
    c_slip = turnover * slip_bps / 10_000.0

    # Funding rates are 8h observations; STRICT4H accrues half per 4h bar.
    c_fund = exposure * funding_rate.reindex(ohlcv.index).fillna(0.0) * 0.5 * costs.funding_multiplier
    r_net = r_raw - c_fee - c_slip - c_fund

    return pd.DataFrame(
        {
            "open": ohlcv["open"],
            "high": ohlcv["high"],
            "low": ohlcv["low"],
            "close": close,
            "funding_rate": funding_rate.reindex(ohlcv.index).fillna(0.0),
            "signal": signal,
            "exposure": exposure,
            "delta_exposure": delta,
            "r_mkt": r_mkt,
            "r_raw": r_raw,
            "c_fee": c_fee,
            "c_slip": c_slip,
            "c_fund": c_fund,
            "r_net": r_net.replace([np.inf, -np.inf], 0.0).fillna(0.0),
        },
        index=ohlcv.index,
    )
