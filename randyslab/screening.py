from __future__ import annotations

import pandas as pd

from .config import CostProfile, Screening, Windows
from .engine import run_strict4h
from .metrics import max_drawdown, monthly_geometric_mean

STAGE2_WINDOWS = ("train_val", "training")


def stage2_window(windows: Windows, window: str = "train_val") -> tuple[str, str]:
    """Stage II screens Stage I survivors on Training+Validation, as in the AutoQuant paper.

    ``training`` reproduces the 2026-06-30 regression in which Stage II only re-scored Stage I's own window.
    """
    if window == "train_val":
        return windows.training_start, windows.validation_end
    if window == "training":
        return windows.training_start, windows.training_end
    raise ValueError(f"window must be one of {STAGE2_WINDOWS}")


def scenario_grid(base: CostProfile, screening: Screening) -> list[CostProfile]:
    out = []
    for taker in screening.scenario_taker_bps:
        for fund_mult in screening.scenario_funding_mult:
            out.append(
                CostProfile(
                    taker_bps=float(taker),
                    slippage_bps=base.slippage_bps,
                    funding_multiplier=float(fund_mult),
                    fee_multiplier=base.fee_multiplier,
                    slippage_multiplier=base.slippage_multiplier,
                )
            )
    return out


def robust_summary(ohlcv, funding, candidates: list[dict], base_costs: CostProfile, screening: Screening) -> pd.DataFrame:
    rows = []
    scenarios = scenario_grid(base_costs, screening)
    for idx, theta in enumerate(candidates):
        metrics = []
        for scen_id, costs in enumerate(scenarios):
            ledger = run_strict4h(ohlcv, funding, theta, costs)
            metrics.append(
                {
                    "scenario": scen_id,
                    "monthly_true": monthly_geometric_mean(ledger["r_net"]),
                    "max_dd": max_drawdown(ledger["r_net"]),
                    "switch_density": float((ledger["delta_exposure"].abs() > 0).sum() / max(len(ledger), 1)),
                }
            )
        m = pd.DataFrame(metrics)
        rows.append(
            {
                "candidate_id": idx,
                "theta": theta,
                "mean_monthly_true": float(m["monthly_true"].mean()),
                "min_monthly_true": float(m["monthly_true"].min()),
                "maxDD_mean": float(m["max_dd"].mean()),
                "switch_density_mean": float(m["switch_density"].mean()),
            }
        )
    return pd.DataFrame(rows)


def select_stable(summary: pd.DataFrame, screening: Screening) -> pd.DataFrame:
    passed = summary[
        (summary["mean_monthly_true"] >= screening.min_mean_monthly_true)
        & (summary["min_monthly_true"] >= screening.min_monthly_true)
        & (summary["maxDD_mean"] <= screening.max_dd_mean)
        & (summary["switch_density_mean"] <= screening.max_switch_density_mean)
    ].copy()
    return passed.sort_values(["mean_monthly_true", "maxDD_mean"], ascending=[False, True]).head(screening.stable_k)
