from __future__ import annotations

import pandas as pd

from randyslab.drawdown_diagnostics import (
    build_drawdown_episode_summary,
    label_drawdown_failure,
    render_drawdown_diagnostics_report,
)


def test_build_drawdown_episode_summary_identifies_peak_trough_and_recovery() -> None:
    idx = pd.date_range("2024-01-01", periods=8, freq="4h", tz="UTC")
    ledger = pd.DataFrame(
        {
            "close": [100, 110, 121, 90, 80, 92, 122, 124],
            "exposure": [0, 1, 1, 1, 1, 0, 1, 1],
            "r_mkt": [0.0, 0.1, 0.1, -0.2562, -0.1111, 0.15, 0.3261, 0.0164],
            "r_net": [0.0, 0.1, 0.1, -0.2562, -0.1111, 0.0, 0.55, 0.0164],
            "c_fee": 0.0,
            "c_slip": 0.0,
            "c_fund": 0.0,
        },
        index=idx,
    )

    episodes = build_drawdown_episode_summary(
        ledger,
        metadata={
            "asset": "SOLUSDT",
            "window": "validation",
            "variant_id": "thr_0p5_long_flat_cap_1p0_none",
        },
        min_drawdown=0.05,
        top_n=1,
    )

    assert len(episodes) == 1
    row = episodes.iloc[0]
    assert row["asset"] == "SOLUSDT"
    assert row["peak_timestamp"] == "2024-01-01T08:00:00+00:00"
    assert row["trough_timestamp"] == "2024-01-01T16:00:00+00:00"
    assert row["recovery_timestamp"] == "2024-01-02T00:00:00+00:00"
    assert round(row["max_drawdown"], 4) == 0.3388
    assert "crash_period_drawdown" in row["failure_labels"]


def test_label_drawdown_failure_marks_btc_sol_avax_validation_and_extreme_drawdown() -> None:
    labels = label_drawdown_failure(
        {
            "asset": "BTCUSDT",
            "window": "validation",
            "max_drawdown": 0.82,
            "market_return_peak_to_trough": -0.31,
        }
    )

    assert labels == [
        "btc_weakness",
        "validation_weakness",
        "crash_period_drawdown",
        "extreme_row_drawdown",
    ]


def test_render_drawdown_diagnostics_report_handles_row_summary_values() -> None:
    payload = {
        "generated_at": "2026-07-03T00:00:00+00:00",
        "row_count": 1,
        "episode_count": 1,
        "thresholds": [0.5],
        "signal_modes": ["long_flat"],
    }
    row_summary = pd.DataFrame(
        [
            {
                "asset": "SOLUSDT",
                "window": "validation",
                "variant_id": "thr_0p5_long_flat_cap_1p0_none",
                "sharpe": 0.5,
                "max_dd": 0.9,
                "max_episode_drawdown": 0.9,
                "episode_failure_labels": "sol_avax_concentration|validation_weakness",
            }
        ]
    )
    episodes = pd.DataFrame(
        [
            {
                "asset": "SOLUSDT",
                "window": "validation",
                "variant_id": "thr_0p5_long_flat_cap_1p0_none",
                "max_drawdown": 0.9,
                "peak_timestamp": "2021-01-01T00:00:00+00:00",
                "trough_timestamp": "2021-05-01T00:00:00+00:00",
                "recovery_timestamp": "",
                "failure_labels": "sol_avax_concentration|validation_weakness",
            }
        ]
    )

    report = render_drawdown_diagnostics_report(payload, row_summary, episodes)

    assert "SOLUSDT" in report
    assert "thr_0p5_long_flat_cap_1p0_none" in report
