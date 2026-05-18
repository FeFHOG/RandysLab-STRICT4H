from __future__ import annotations

import numpy as np
import pandas as pd


BARS_PER_YEAR_4H = 365.0 * 24.0 / 4.0


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    return initial * (1.0 + returns.fillna(0.0)).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(-dd.min())


def cagr(returns: pd.Series, bars_per_year: float = BARS_PER_YEAR_4H) -> float:
    if returns.empty:
        return 0.0
    total = float((1.0 + returns.fillna(0.0)).prod())
    years = len(returns) / bars_per_year
    if years <= 0 or total <= 0:
        return -1.0
    return total ** (1.0 / years) - 1.0


def sharpe(returns: pd.Series, risk_free_rate: float = 0.03, bars_per_year: float = BARS_PER_YEAR_4H) -> float:
    r = returns.fillna(0.0)
    excess = r - risk_free_rate / bars_per_year
    std = float(excess.std(ddof=0))
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(bars_per_year))


def monthly_geometric_mean(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    monthly = monthly.dropna().clip(lower=-0.999999)
    if monthly.empty:
        return 0.0
    return float(np.exp(np.log1p(monthly).mean()) - 1.0)


def summarize(ledger: pd.DataFrame, risk_free_rate: float = 0.03) -> dict:
    r = ledger["r_net"]
    switches = int((ledger["delta_exposure"].abs() > 0).sum())
    return {
        "bars": int(len(ledger)),
        "cagr": cagr(r),
        "monthly_geom": monthly_geometric_mean(r),
        "sharpe": sharpe(r, risk_free_rate=risk_free_rate),
        "max_dd": max_drawdown(r),
        "trades": switches,
        "switch_density": switches / max(len(ledger), 1),
        "fee_total": float(ledger["c_fee"].sum()),
        "slippage_total": float(ledger["c_slip"].sum()),
        "funding_total": float(ledger["c_fund"].sum()),
        "net_total": float(r.sum()),
    }
