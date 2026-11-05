from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from randyslab.accounting import (
    CostRates,
    build_ledger,
    funding_feature,
    funding_settled_per_bar,
    legacy_funding_settled,
    normalize_funding_events,
)
from randyslab.config import CostProfile
from randyslab.engine import run_strict4h_signal
from randyslab.metrics import monthly_geometric_mean, summarize

H = 4.0


def _bars(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="4h", tz="UTC")


def _events(stamps: list[str], rates: list[float], intervals: list[float] | None = None) -> pd.DataFrame:
    frame = pd.DataFrame({"funding_rate": rates}, index=pd.to_datetime(stamps, utc=True))
    if intervals is not None:
        frame["funding_interval_hours"] = intervals
    return normalize_funding_events(frame)


def _reference_ledger(
    close: pd.Series,
    exposure: pd.Series,
    events: pd.DataFrame,
    fee_bps: float,
    slippage_bps: float,
    funding_multiplier: float,
) -> pd.Series:
    """Independent bar-by-bar oracle: settlements are matched to bars by scanning timestamps, not by ceil arithmetic."""
    step = pd.Timedelta(hours=H)
    settlements = list(zip(events.index, events["funding_rate"], strict=True))
    out = []
    previous_exposure = 0.0
    for i, bar_open in enumerate(close.index):
        held = float(exposure.iloc[i])
        market = 0.0 if i == 0 else float(close.iloc[i]) / float(close.iloc[i - 1]) - 1.0
        trading = abs(held - previous_exposure) * (fee_bps + slippage_bps) / 10_000.0
        funding = sum(rate for tau, rate in settlements if bar_open < tau <= bar_open + step)
        out.append(held * market - trading - held * funding * funding_multiplier)
        previous_exposure = held
    return pd.Series(out, index=close.index)


def test_eight_hour_settlements_belong_to_the_bar_ending_on_them():
    index = _bars("2024-01-01 00:00", 6)
    events = _events(["2024-01-01 00:00", "2024-01-01 08:00", "2024-01-01 16:00", "2024-01-02 00:00"], [1e-4] * 4)
    settled = funding_settled_per_bar(index, events, H)
    # Bar 04:00 holds through 08:00, bar 12:00 through 16:00, bar 20:00 through 00:00 the next day.
    assert settled.tolist() == pytest.approx([0.0, 1e-4, 0.0, 1e-4, 0.0, 1e-4])


def test_millisecond_settlement_stamps_stay_on_their_bar():
    index = _bars("2024-01-01 00:00", 3)
    events = _events(["2024-01-01 08:00:00.008"], [2e-4])
    assert events.index[0] == pd.Timestamp("2024-01-01 08:00", tz="UTC")
    assert funding_settled_per_bar(index, events, H).tolist() == pytest.approx([0.0, 2e-4, 0.0])


def test_hourly_settlements_are_summed_per_bar_and_feature_is_8h_equivalent():
    index = _bars("2024-01-01 00:00", 2)
    stamps = [f"2024-01-01 {hour:02d}:00" for hour in range(1, 9)]
    events = _events(stamps, [1e-4] * 8)
    assert events["interval_hours"].tolist() == pytest.approx([1.0] * 8)
    assert funding_settled_per_bar(index, events, H).tolist() == pytest.approx([4e-4, 4e-4])
    feature = funding_feature(_bars("2024-01-01 04:00", 1), events)
    assert feature.iloc[0] == pytest.approx(8e-4)


def test_declared_interval_column_wins_and_gaps_are_not_longer_intervals():
    declared = _events(["2024-01-01 00:00", "2024-01-01 04:00"], [1e-4, 1e-4], intervals=[4, 4])
    assert declared["interval_hours"].tolist() == [4.0, 4.0]
    gapped = _events(["2024-01-01 00:00", "2024-01-02 00:00"], [1e-4, 1e-4])
    assert gapped["interval_hours"].tolist() == [8.0, 8.0]


def test_funding_feature_never_uses_settlements_after_the_bar_open():
    index = _bars("2024-01-01 00:00", 12)
    stamps = pd.date_range("2024-01-01 00:00", periods=6, freq="8h", tz="UTC")
    events = normalize_funding_events(pd.Series(np.arange(1, 7) * 1e-4, index=stamps, name="funding_rate"))
    full = funding_feature(index, events)
    for cut in range(1, len(index)):
        visible = events[events.index <= index[cut]]
        truncated = funding_feature(index[: cut + 1], visible)
        assert truncated.tolist() == pytest.approx(full.iloc[: cut + 1].tolist())


def test_ledger_matches_independent_reference_engine():
    rng = np.random.default_rng(7)
    index = _bars("2024-01-01 00:00", 300)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(index)))), index=index)
    exposure = pd.Series(rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], len(index)), index=index)
    hourly = pd.date_range("2024-01-20 00:00", "2024-01-25 00:00", freq="1h", tz="UTC")
    eight = pd.date_range("2024-01-01 00:00", "2024-02-20 00:00", freq="8h", tz="UTC")
    stamps = eight[(eight < hourly[0]) | (eight > hourly[-1])].union(hourly)
    raw = pd.DataFrame({"funding_rate": rng.normal(1e-4, 3e-4, len(stamps))}, index=stamps)
    raw.index = raw.index + pd.to_timedelta(rng.integers(0, 50, len(stamps)), unit="ms")
    events = normalize_funding_events(raw)
    costs = CostProfile(taker_bps=4.0, slippage_bps=2.0, funding_multiplier=1.5)

    ledger = build_ledger(close, exposure, funding_settled_per_bar(index, events, H), costs)
    expected = _reference_ledger(close, exposure, events, 4.0, 2.0, 1.5)

    assert ledger["r_net"].to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-15)
    covered = (events.index > index[0]) & (events.index <= index[-1] + pd.Timedelta(hours=H))
    assert ledger["funding_settled"].sum() == pytest.approx(events.loc[covered, "funding_rate"].sum())


def test_sign_conventions_and_first_bar_entry():
    index = _bars("2024-01-01 04:00", 2)
    close = pd.Series([100.0, 100.0], index=index)
    settled = pd.Series([1e-3, 1e-3], index=index)
    costs = CostRates(fee_bps=4.0, slippage_bps=1.0)
    long = build_ledger(close, pd.Series([1.0, 1.0], index=index), settled, costs)
    short = build_ledger(close, pd.Series([-1.0, -1.0], index=index), settled, costs)
    assert long["c_fund"].tolist() == pytest.approx([1e-3, 1e-3])
    assert short["c_fund"].tolist() == pytest.approx([-1e-3, -1e-3])
    assert long["delta_exposure"].tolist() == [1.0, 0.0]
    assert long["c_fee"].iloc[0] == pytest.approx(4e-4)


def test_two_hour_interval_crash_is_not_undercharged():
    index = _bars("2022-11-09 00:00", 6)
    stamps = pd.date_range("2022-11-09 02:00", "2022-11-10 00:00", freq="2h", tz="UTC")
    events = normalize_funding_events(pd.Series(-0.02, index=stamps, name="funding_rate"))
    exposure = pd.Series(-1.0, index=index)
    close = pd.Series(10.0, index=index)
    exact = build_ledger(close, exposure, funding_settled_per_bar(index, events, H), CostRates(0.0, 0.0))
    legacy_feature = funding_feature(index, events, as_8h_equivalent=False)
    legacy = build_ledger(close, exposure, legacy_funding_settled(legacy_feature, H), CostRates(0.0, 0.0))
    assert exact["c_fund"].sum() == pytest.approx(0.02 * len(stamps))
    assert legacy["c_fund"].sum() < 0.5 * exact["c_fund"].sum()


def test_cost_rates_accept_both_projects_configs():
    assert CostRates.from_costs(CostProfile(taker_bps=4.0, slippage_bps=2.0, fee_multiplier=2.0)).fee_bps == 8.0
    qr_style = SimpleNamespace(taker_bps=5.0, slippage_bps=1.0, funding_multiplier=0.5)
    rates = CostRates.from_costs(qr_style)
    assert (rates.fee_bps, rates.slippage_bps, rates.funding_multiplier) == (5.0, 1.0, 0.5)


def test_engine_uses_settled_funding_and_scales_position_size():
    index = _bars("2024-01-01 00:00", 4)
    ohlcv = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0, "funding_settled": 1e-3},
        index=index,
    )
    signal = pd.Series([1, 1, 1, 1], index=index)
    ledger = run_strict4h_signal(ohlcv, pd.Series(0.0, index=index), signal, {"max_exposure_abs": 1.5}, CostProfile())
    assert ledger["exposure"].tolist() == [0.0, 1.5, 1.5, 1.5]
    assert ledger["c_fund"].tolist() == pytest.approx([0.0, 1.5e-3, 1.5e-3, 1.5e-3])


def test_monthly_geometric_mean_ignores_partial_months_and_summary_reports_final_equity():
    index = pd.date_range("2024-01-31 00:00", "2024-03-01 20:00", freq="4h", tz="UTC")
    returns = pd.Series(0.0, index=index)
    returns[index.month == 1] = 0.05
    assert monthly_geometric_mean(returns) == pytest.approx(0.0)
    ledger = pd.DataFrame(
        {"r_net": returns, "delta_exposure": 0.0, "c_fee": 0.0, "c_slip": 0.0, "c_fund": 0.0}, index=index
    )
    assert summarize(ledger)["final_equity"] == pytest.approx(1.05**6)
