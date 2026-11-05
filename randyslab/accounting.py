"""Shared strict accounting core for the Randy stack.

RandysLab judges and baselines and QuantumRandy research backtests all build their ledgers here, so fee, slippage and
funding semantics live in one place (QuantumRandy docs/RANDY_STACK_TARGET_ARCHITECTURE.md assigns them to RandysLab).

Bar conventions (bars indexed by open time T, covering [T, T + bar_hours)):

- ``exposure[T]`` is the position held during bar T; callers shift signals so it was decided at an earlier bar close.
- ``r_mkt[T] = close[T] / close[previous bar] - 1``.
- Trading costs are charged on ``|exposure[T] - exposure[previous bar]|``; the first bar pays its entry from flat.
- A funding settlement at time ``tau`` is paid by the position held at ``tau``. Rebalancing happens just after a bar
  closes, so a settlement exactly on the boundary ``T + bar_hours`` belongs to bar T: bar T pays the sum of rates
  settled in ``(T, T + bar_hours]``. Longs pay positive rates. Settlement intervals may be 8h, 4h or 1h and may change
  over time (Binance switches symbols automatically), so costs are summed per event rather than smoothed.
- The funding *feature* exposed to signals is the last rate settled at or before the bar's open time, so it never
  uses information from inside the bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

STANDARD_FUNDING_INTERVAL_HOURS = 8.0


@dataclass(frozen=True)
class CostRates:
    fee_bps: float
    slippage_bps: float
    funding_multiplier: float = 1.0

    @classmethod
    def from_costs(cls, costs: Any) -> CostRates:
        """Accept a RandysLab ``CostProfile``, a QuantumRandy ``CostConfig``, or any object with the same fields."""
        if isinstance(costs, CostRates):
            return costs
        return cls(
            fee_bps=float(getattr(costs, "taker_bps", 0.0)) * float(getattr(costs, "fee_multiplier", 1.0)),
            slippage_bps=float(getattr(costs, "slippage_bps", 0.0)) * float(getattr(costs, "slippage_multiplier", 1.0)),
            funding_multiplier=float(getattr(costs, "funding_multiplier", 1.0)),
        )


def normalize_funding_events(funding: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Return settlement events indexed by minute-floored UTC time with ``funding_rate`` and ``interval_hours``.

    Binance stamps settlements a few milliseconds late (``08:00:00.008``); flooring to the minute keeps them on the
    settlement grid. Without a ``funding_interval_hours`` column the interval is inferred from the gap to the previous
    settlement, capped at 8h so that missing data is not mistaken for a longer interval.
    """
    frame = funding.rename("funding_rate").to_frame() if isinstance(funding, pd.Series) else funding.copy()
    if "timestamp" in frame.columns:
        frame = frame.set_index("timestamp")
    if "funding_rate" not in frame.columns:
        raise ValueError("funding events need a funding_rate column")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).floor("min")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    rates = pd.to_numeric(frame["funding_rate"], errors="coerce").fillna(0.0).astype(float)

    gaps = pd.Series(frame.index, index=frame.index).diff().dt.total_seconds() / 3600.0
    inferred = gaps.round().clip(lower=1.0, upper=STANDARD_FUNDING_INTERVAL_HOURS).bfill()
    inferred = inferred.fillna(STANDARD_FUNDING_INTERVAL_HOURS)
    if "funding_interval_hours" in frame.columns:
        declared = pd.to_numeric(frame["funding_interval_hours"], errors="coerce")
        interval = declared.where(declared > 0).fillna(inferred)
    else:
        interval = inferred
    return pd.DataFrame({"funding_rate": rates, "interval_hours": interval.astype(float)}, index=frame.index)


def funding_feature(
    bar_index: pd.DatetimeIndex,
    events: pd.DataFrame,
    *,
    as_8h_equivalent: bool = True,
) -> pd.Series:
    """Last settled rate known at each bar's open time, by default rescaled to an 8h-equivalent rate.

    The rescaling keeps the feature comparable when a symbol switches between 8h, 4h and 1h settlement.
    """
    rate = events["funding_rate"]
    if as_8h_equivalent:
        rate = rate * STANDARD_FUNDING_INTERVAL_HOURS / events["interval_hours"]
    aligned = rate.reindex(bar_index.union(rate.index)).sort_index().ffill().reindex(bar_index)
    return aligned.fillna(0.0).astype(float).rename("funding_rate")


def funding_settled_per_bar(bar_index: pd.DatetimeIndex, events: pd.DataFrame, bar_hours: float) -> pd.Series:
    """Sum of funding rates settled while each bar's position is held, i.e. in ``(T, T + bar_hours]``.

    Settlements that fall inside a bar missing from ``bar_index`` are dropped; check data gaps separately.
    """
    if events.empty:
        return pd.Series(0.0, index=bar_index, name="funding_settled")
    step = pd.Timedelta(hours=float(bar_hours))
    owner = events.index.ceil(step) - step
    settled = events["funding_rate"].groupby(owner).sum()
    return settled.reindex(bar_index, fill_value=0.0).astype(float).rename("funding_settled")


def legacy_funding_settled(funding_rate: pd.Series, bar_hours: float = 4.0) -> pd.Series:
    """Pre-2026-11 approximation: accrue the last 8h rate pro rata every bar.

    Only for frames that carry no settlement events (synthetic test data, old artifacts). It undercharges whenever the
    real settlement interval is shorter than 8h.
    """
    return (funding_rate.fillna(0.0).astype(float) * float(bar_hours) / STANDARD_FUNDING_INTERVAL_HOURS).rename(
        "funding_settled"
    )


def build_ledger(
    close: pd.Series,
    exposure: pd.Series,
    funding_settled: pd.Series,
    costs: Any,
) -> pd.DataFrame:
    """Bar-level ledger: market return, gross strategy return, fee, slippage, funding and net return."""
    rates = CostRates.from_costs(costs)
    close = close.astype(float)
    exposure = exposure.reindex(close.index).fillna(0.0).astype(float)
    r_mkt = close.pct_change().fillna(0.0)
    r_raw = exposure * r_mkt
    delta = exposure.diff()
    if len(delta):
        delta.iloc[0] = exposure.iloc[0]
    turnover = delta.abs()
    c_fee = turnover * rates.fee_bps / 10_000.0
    c_slip = turnover * rates.slippage_bps / 10_000.0
    settled = funding_settled.reindex(close.index).fillna(0.0).astype(float)
    c_fund = exposure * settled * rates.funding_multiplier
    r_net = (r_raw - c_fee - c_slip - c_fund).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return pd.DataFrame(
        {
            "exposure": exposure,
            "delta_exposure": delta,
            "r_mkt": r_mkt,
            "r_raw": r_raw,
            "funding_settled": settled,
            "c_fee": c_fee,
            "c_slip": c_slip,
            "c_fund": c_fund,
            "r_net": r_net,
        },
        index=close.index,
    )


def infer_bar_hours(index: pd.Index, default: float = 4.0) -> float:
    """Median bar spacing in hours, or ``default`` when it cannot be inferred."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return float(default)
    spacing = pd.Series(index).diff().dropna().median()
    hours = spacing.total_seconds() / 3600.0 if pd.notna(spacing) else 0.0
    return float(hours) if hours > 0 else float(default)
