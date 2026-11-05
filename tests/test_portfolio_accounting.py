from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from randyslab.accounting import CostRates, build_ledger, build_portfolio_ledger
from randyslab.config import CostProfile


def _panel(n: int = 120, assets: int = 4, seed: int = 9) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    columns = [f"A{i}USDT" for i in range(assets)]
    paths = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.03, (n, assets)), axis=0))
    close = pd.DataFrame(paths, index=index, columns=columns)
    settled = pd.DataFrame(rng.normal(3e-4, 5e-4, (n, assets)), index=index, columns=columns)
    return close, settled


def _reference(weights, close, settled, fee_bps, slip_by_asset, multiplier) -> pd.Series:
    """Per-asset, per-bar loop written independently of the vectorized ledger."""
    out = []
    previous = dict.fromkeys(close.columns, 0.0)
    for i, bar in enumerate(close.index):
        total = 0.0
        for asset in close.columns:
            weight = float(weights.at[bar, asset])
            price = close.at[bar, asset]
            last = close.at[close.index[i - 1], asset] if i > 0 else np.nan
            market = price / last - 1.0 if i > 0 and pd.notna(price) and pd.notna(last) else 0.0
            trade = abs(weight - previous[asset]) * (fee_bps + slip_by_asset[asset]) / 10_000.0
            funding = weight * float(settled.at[bar, asset]) * multiplier
            total += weight * market - trade - funding
            previous[asset] = weight
        out.append(total)
    return pd.Series(out, index=close.index)


def test_single_asset_book_matches_the_single_asset_ledger():
    close, settled = _panel(assets=1)
    rng = np.random.default_rng(1)
    exposure = pd.Series(rng.choice([-1.0, 0.0, 0.5, 1.0], len(close)), index=close.index)
    costs = CostProfile(taker_bps=4.0, slippage_bps=3.0, funding_multiplier=1.2)
    single = build_ledger(close.iloc[:, 0], exposure, settled.iloc[:, 0], costs)
    book = build_portfolio_ledger(exposure.to_frame(close.columns[0]), close, settled, costs)
    assert book["r_net"].to_numpy() == pytest.approx(single["r_net"].to_numpy(), abs=1e-15)


def test_book_matches_reference_loop_with_listings_delistings_and_per_asset_slippage():
    close, settled = _panel()
    close.iloc[:30, 1] = np.nan  # listed on day 30
    close.iloc[80:, 2] = np.nan  # delisted after day 79
    rng = np.random.default_rng(2)
    weights = pd.DataFrame(rng.uniform(-0.3, 0.3, close.shape), index=close.index, columns=close.columns)
    weights = weights.where(close.notna(), 0.0)
    weights.iloc[30, 1] = 0.0  # no position on the listing bar itself (no prior close)
    slippage = pd.Series({"A0USDT": 1.0, "A1USDT": 8.0, "A2USDT": 5.0, "A3USDT": 2.0})
    costs = CostRates(fee_bps=4.0, slippage_bps=2.0, funding_multiplier=1.0)

    book = build_portfolio_ledger(weights, close, settled, costs, slippage_bps=slippage)
    expected = _reference(weights, close, settled, 4.0, slippage.to_dict(), 1.0)
    assert book["r_net"].to_numpy() == pytest.approx(expected.to_numpy(), abs=1e-14)
    # Closing the delisted asset is a trade on the first unpriced bar.
    assert book["turnover"].iloc[80] >= abs(weights.iloc[79, 2])


def test_weight_without_a_price_is_rejected():
    close, settled = _panel()
    close.iloc[50:, 3] = np.nan
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    weights.iloc[60, 3] = 0.1
    with pytest.raises(ValueError, match="A3USDT"):
        build_portfolio_ledger(weights, close, settled, CostProfile())


def test_dollar_neutral_book_reports_exposures():
    close, _ = _panel()
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    weights[["A0USDT", "A1USDT"]] = 0.25
    weights[["A2USDT", "A3USDT"]] = -0.25
    book = build_portfolio_ledger(weights, close, None, CostProfile())
    assert book["net_exposure"].abs().max() == pytest.approx(0.0)
    assert book["gross_exposure"].eq(1.0).all()
    assert book["positions"].eq(4).all()
    assert book["turnover"].iloc[0] == pytest.approx(1.0)
    assert book["turnover"].iloc[1:].eq(0.0).all()
