from __future__ import annotations

import json

import pandas as pd

from randyslab.config import CostProfile
from randyslab.factor_candidate_review import (
    ReviewRules,
    build_factor_candidate_review,
    review_factor_candidate_sensitivity,
)
from randyslab.factor_candidate_sensitivity import run_factor_candidate_sensitivity
from randyslab.formula_candidates import (
    evaluate_candidate_factor,
    evaluate_formula,
    judge_factor_candidate,
    run_factor_candidate_judging,
)
from randyslab.metrics import sharpe


def _write_market_fixture(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    idx = pd.date_range("2024-01-01", periods=220, freq="4h", tz="UTC")
    close = pd.Series(range(100, 320), index=idx, dtype=float)
    volume = pd.Series([1000 + (i % 20) * 25 for i in range(len(idx))], index=idx, dtype=float)
    pd.DataFrame(
        {
            "timestamp": idx,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": volume,
        }
    ).to_csv(data_dir / "BTCUSDT_4h.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=110, freq="8h", tz="UTC"),
            "funding_rate": 0.0001,
        }
    ).to_csv(data_dir / "BTCUSDT_funding.csv", index=False)


def _write_config(tmp_path) -> None:
    (tmp_path / "strict4h.yaml").write_text(
        """
symbol: BTC/USDT:USDT
exchange: binanceusdm
risk_free_rate: 0.0
paths:
  ohlcv_csv: data/BTCUSDT_4h.csv
  funding_csv: data/BTCUSDT_funding.csv
costs:
  taker_bps: 4.0
  slippage_bps: 2.0
  funding_multiplier: 1.0
windows:
  training_start: "2024-01-01"
  training_end: "2024-01-20"
  validation_start: "2024-01-20"
  validation_end: "2024-02-01"
  long_start: "2024-01-01"
  long_end: "2024-02-10"
  blind_start: "2024-02-01"
  blind_end: "2024-02-10"
""".strip(),
        encoding="utf-8",
    )


def test_evaluate_formula_supports_selector_candidate_shapes() -> None:
    idx = pd.date_range("2024-01-01", periods=160, freq="4h", tz="UTC")
    ohlcv = pd.DataFrame(
        {
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": pd.Series(range(100, 260), index=idx, dtype=float),
            "volume": pd.Series(range(1000, 1160), index=idx, dtype=float),
        },
        index=idx,
    )
    funding = pd.Series(0.0, index=idx)

    volume_factor = evaluate_formula("zscore(ema(volume,24),96)", ohlcv, funding)
    vol_factor = evaluate_formula("zscore(std(close,48),120)", ohlcv, funding)

    assert volume_factor.index.equals(idx)
    assert vol_factor.index.equals(idx)
    assert volume_factor.dropna().empty is False
    assert vol_factor.dropna().empty is True


def test_evaluate_formula_supports_quantumrandy_core_regime_proxy_shapes() -> None:
    idx = pd.date_range("2024-01-01", periods=180, freq="4h", tz="UTC")
    close = pd.Series([100 + i + (i % 7) * 0.3 for i in range(len(idx))], index=idx, dtype=float)
    ohlcv = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": pd.Series([1000 + (i % 13) * 50 for i in range(len(idx))], index=idx, dtype=float),
        },
        index=idx,
    )
    funding = pd.Series([0.0001 * ((i % 5) - 2) for i in range(len(idx))], index=idx, dtype=float)

    formulas = [
        "zscore(corr(funding_rate,ret(close,12),48),72)",
        "zscore(div(sub(high,low),close),72)",
        "zscore(rsi(close,24),72)",
    ]

    for formula in formulas:
        factor = evaluate_formula(formula, ohlcv, funding)
        assert factor.index.equals(idx)
        assert factor.dropna().empty is False


def test_evaluate_candidate_factor_supports_research_only_component_mean() -> None:
    idx = pd.date_range("2024-01-01", periods=180, freq="4h", tz="UTC")
    ohlcv = pd.DataFrame(
        {
            "open": 1.0,
            "high": 2.0,
            "low": 0.5,
            "close": pd.Series(range(100, 280), index=idx, dtype=float),
            "volume": pd.Series(range(1000, 1180), index=idx, dtype=float),
        },
        index=idx,
    )
    funding = pd.Series(0.0, index=idx)
    candidate = {
        "candidate_id": "combo_volume",
        "formula": "equal_weight_mean(zscore(ema(volume,24),96),zscore(ema(volume,48),120))",
        "combination_method": "mean",
        "component_formulas": [
            "zscore(ema(volume,24),96)",
            "zscore(ema(volume,48),120)",
        ],
    }

    combined = evaluate_candidate_factor(candidate, ohlcv, funding)

    assert combined.index.equals(idx)
    assert combined.dropna().empty is False


def test_run_factor_candidate_judging_writes_research_artifacts_and_failure_reasons(tmp_path) -> None:
    _write_market_fixture(tmp_path)
    _write_config(tmp_path)
    candidates = tmp_path / "factor_candidates.jsonl"
    candidates.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "artifact_type": "quantumrandy_factor_candidate_export",
                        "research_only": True,
                        "not_runtime_publish_payload": True,
                        "candidate_id": "qr_volume",
                        "formula": "zscore(ema(volume,24),96)",
                        "formula_family": "volume_liquidity",
                        "parent_factor_id": "qr_parent",
                        "llm_true_improved_count": 5,
                    }
                ),
                json.dumps(
                    {
                        "candidate_id": "qr_bad",
                        "formula": "zscore(unknown_field,24)",
                        "formula_family": "other",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "judge"
    payload = run_factor_candidate_judging(
        config_path=tmp_path / "strict4h.yaml",
        candidate_jsonl=candidates,
        out_dir=out,
        write_ledgers=True,
        signal_mode="long_flat",
        ohlcv_csv=tmp_path / "data" / "BTCUSDT_4h.csv",
        funding_csv=tmp_path / "data" / "BTCUSDT_funding.csv",
        symbol="BTC/USDT:USDT",
    )

    assert payload["artifact_type"] == "randyslab_factor_candidate_judging"
    assert payload["role"] == "strict_factor_candidate_judge"
    assert payload["safety"]["research_only"] is True
    assert payload["safety"]["not_runtime_publish_payload"] is True
    assert payload["execution"]["matching"] == "next_bar_t_plus_1"
    assert payload["execution"]["signal_mode"] == "long_flat"
    assert payload["data"]["ohlcv_csv"].endswith("BTCUSDT_4h.csv")
    assert payload["data"]["funding_csv"].endswith("BTCUSDT_funding.csv")
    assert payload["candidate_count"] == 2
    assert payload["completed_count"] == 1
    assert payload["failed_count"] == 1
    assert (out / "factor_candidate_judging_summary.csv").exists()
    assert (out / "factor_candidate_judging_summary.json").exists()
    assert (out / "factor_candidate_judging_report.md").exists()
    assert (out / "qr_volume_ledger.csv").exists()

    summary = pd.read_csv(out / "factor_candidate_judging_summary.csv")
    by_id = {row["candidate_id"]: row for row in summary.to_dict(orient="records")}
    assert by_id["qr_volume"]["status"] == "completed"
    assert by_id["qr_volume"]["bars"] == 220
    assert by_id["qr_bad"]["status"] == "failed"
    assert "formula_error" in str(by_id["qr_bad"]["failure_reasons"])
    report = (out / "factor_candidate_judging_report.md").read_text(encoding="utf-8")
    assert "not a runtime publish payload" in report


def test_judge_factor_candidate_applies_research_exposure_and_volatility_controls() -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="4h", tz="UTC")
    close = pd.Series(range(100, 140), index=idx, dtype=float)
    ohlcv = pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0,
        },
        index=idx,
    )
    funding = pd.Series(0.0, index=idx)
    candidate = {
        "candidate_id": "ret_long",
        "formula": "ret(close,1)",
        "formula_family": "test",
    }

    capped = judge_factor_candidate(
        candidate,
        ohlcv=ohlcv,
        funding=funding,
        costs=CostProfile(taker_bps=0.0, slippage_bps=0.0, funding_multiplier=0.0),
        risk_free_rate=0.0,
        signal_mode="long_flat",
        max_exposure_abs=0.25,
    )
    filtered = judge_factor_candidate(
        candidate,
        ohlcv=ohlcv,
        funding=funding,
        costs=CostProfile(taker_bps=0.0, slippage_bps=0.0, funding_multiplier=0.0),
        risk_free_rate=0.0,
        signal_mode="long_flat",
        max_exposure_abs=0.25,
        volatility_cap_formula="std(close,2)",
        volatility_cap_max=0.0,
    )

    assert capped["status"] == "completed"
    assert capped["ledger"]["exposure"].abs().max() <= 0.25
    assert capped["nonzero_signal_bars"] > 0
    assert filtered["status"] == "completed"
    assert filtered["nonzero_signal_bars"] == 0
    assert "no_nonzero_signal" in filtered["failure_reasons"]


def test_sharpe_returns_zero_for_near_constant_flat_returns() -> None:
    idx = pd.date_range("2024-01-01", periods=80, freq="4h", tz="UTC")
    returns = pd.Series(0.0, index=idx)

    assert sharpe(returns, risk_free_rate=0.03) == 0.0


def test_judge_factor_candidate_applies_research_drawdown_stop_cooldown() -> None:
    idx = pd.date_range("2024-01-01", periods=16, freq="4h", tz="UTC")
    close = pd.Series(
        [100, 108, 116, 124, 132, 140, 112, 101, 98, 99, 100, 102, 104, 106, 108, 110],
        index=idx,
        dtype=float,
    )
    ohlcv = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0,
        },
        index=idx,
    )
    funding = pd.Series(0.0, index=idx)
    candidate = {
        "candidate_id": "always_long",
        "formula": "ret(close,1)",
        "formula_family": "test",
    }

    stopped = judge_factor_candidate(
        candidate,
        ohlcv=ohlcv,
        funding=funding,
        costs=CostProfile(taker_bps=0.0, slippage_bps=0.0, funding_multiplier=0.0),
        risk_free_rate=0.0,
        signal_threshold=-1.0,
        signal_mode="long_flat",
        drawdown_stop_max=0.15,
        drawdown_stop_cooldown_bars=3,
    )

    assert stopped["status"] == "completed"
    assert stopped["drawdown_stop_label"] == "dd_stop_0p15_cd_3"
    cooldown_exposure = stopped["ledger"]["exposure"].iloc[8:11]
    assert (cooldown_exposure == 0.0).any()
    assert stopped["ledger"]["exposure"].iloc[-1] == 1.0


def test_run_factor_candidate_sensitivity_writes_aggregate_research_artifacts(tmp_path) -> None:
    _write_market_fixture(tmp_path)
    _write_config(tmp_path)
    candidates = tmp_path / "factor_candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "candidate_id": "qr_volume",
                "formula": "zscore(ema(volume,24),96)",
                "formula_family": "volume_liquidity",
                "intended_scope": "BTCUSDT_4h",
                "applicability_hypothesis": "BTCUSDT 4h scoped participation research.",
                "out_of_scope_policy": "diagnostic_only",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "sensitivity"
    payload = run_factor_candidate_sensitivity(
        config_path=tmp_path / "strict4h.yaml",
        candidate_jsonl=candidates,
        assets=[
            {
                "symbol": "BTCUSDT",
                "ohlcv_csv": str(tmp_path / "data" / "BTCUSDT_4h.csv"),
                "funding_csv": str(tmp_path / "data" / "BTCUSDT_funding.csv"),
            }
        ],
        windows=["all", "training"],
        thresholds=[0.0, 0.5],
        signal_modes=["long_short", "long_flat"],
        out_dir=out,
    )

    assert payload["artifact_type"] == "randyslab_factor_candidate_sensitivity"
    assert payload["safety"]["research_only"] is True
    assert payload["run_count"] == 8
    assert payload["candidate_row_count"] == 8
    assert (out / "factor_candidate_sensitivity_detail.csv").exists()
    assert (out / "factor_candidate_sensitivity_summary.csv").exists()
    assert (out / "factor_candidate_sensitivity_summary.json").exists()
    assert (out / "factor_candidate_sensitivity_report.md").exists()
    detail = pd.read_csv(out / "factor_candidate_sensitivity_detail.csv")
    assert set(detail["window"]) == {"all", "training"}
    assert set(detail["threshold"]) == {0.0, 0.5}
    assert set(detail["signal_mode"]) == {"long_short", "long_flat"}
    assert set(detail["intended_scope"]) == {"BTCUSDT_4h"}
    assert set(detail["applicability_hypothesis"]) == {"BTCUSDT 4h scoped participation research."}
    assert set(detail["out_of_scope_policy"]) == {"diagnostic_only"}
    summary = pd.read_csv(out / "factor_candidate_sensitivity_summary.csv")
    assert summary.iloc[0]["candidate_id"] == "qr_volume"
    assert summary.iloc[0]["completed_rows"] == 8
    report = (out / "factor_candidate_sensitivity_report.md").read_text(encoding="utf-8")
    assert "not a runtime publish payload" in report
    assert "Threshold sweeps are diagnostic evidence" in report


def test_run_factor_candidate_sensitivity_writes_drawdown_variant_columns(tmp_path) -> None:
    _write_market_fixture(tmp_path)
    _write_config(tmp_path)
    candidates = tmp_path / "factor_candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "candidate_id": "qr_volume",
                "formula": "zscore(ema(volume,24),96)",
                "formula_family": "volume_liquidity",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "drawdown_sensitivity"
    payload = run_factor_candidate_sensitivity(
        config_path=tmp_path / "strict4h.yaml",
        candidate_jsonl=candidates,
        assets=[
            {
                "symbol": "BTCUSDT",
                "ohlcv_csv": str(tmp_path / "data" / "BTCUSDT_4h.csv"),
                "funding_csv": str(tmp_path / "data" / "BTCUSDT_funding.csv"),
            }
        ],
        windows=["all"],
        thresholds=[0.5],
        signal_modes=["long_flat"],
        exposure_caps=[1.0, 0.5],
        volatility_caps=[
            {"label": "none", "formula": "", "max_value": None},
            {"label": "low_vol_only", "formula": "std(close,2)", "max_value": 0.0},
        ],
        drawdown_stops=[
            {"label": "none", "max_drawdown": None, "cooldown_bars": 0},
            {"label": "dd_stop_10", "max_drawdown": 0.1, "cooldown_bars": 6},
        ],
        out_dir=out,
    )

    assert payload["run_count"] == 8
    assert payload["variant_count"] == 8
    assert payload["exposure_caps"] == [1.0, 0.5]
    detail = pd.read_csv(out / "factor_candidate_sensitivity_detail.csv")
    assert set(detail["exposure_cap"]) == {1.0, 0.5}
    assert set(detail["volatility_cap_label"]) == {"none", "low_vol_only"}
    assert set(detail["drawdown_stop_label"]) == {"none", "dd_stop_10"}
    assert set(detail["drawdown_stop_cooldown_bars"]) == {0, 6}
    assert detail["variant_id"].nunique() == 8
    report = (out / "factor_candidate_sensitivity_report.md").read_text(encoding="utf-8")
    assert "Exposure caps" in report
    assert "Volatility caps" in report
    assert "Drawdown stops" in report


def test_review_factor_candidate_sensitivity_blocks_conservative_failures(tmp_path) -> None:
    detail = pd.DataFrame(
        [
            {
                "candidate_id": "steady",
                "formula": "zscore(ema(volume,48),120)",
                "asset": asset,
                "window": window,
                "threshold": 0.5,
                "status": "completed",
                "sharpe": 0.8,
                "cagr": 0.2,
                "max_dd": 0.2,
            }
            for asset in ["BTC", "ETH", "SOL"]
            for window in ["training", "validation", "blind"]
        ]
        + [
            {
                "candidate_id": "fragile",
                "formula": "zscore(std(close,48),144)",
                "asset": "SOL",
                "window": window,
                "threshold": 1.0,
                "status": "completed",
                "sharpe": -0.2 if window == "validation" else 1.1,
                "cagr": 0.1,
                "max_dd": 0.9,
            }
            for window in ["training", "validation", "blind"]
        ]
    )

    review = build_factor_candidate_review(
        detail,
        ReviewRules(
            min_completed_rows=3,
            max_mean_max_dd=0.5,
            max_worst_max_dd=0.8,
        ),
    )
    by_id = {row["candidate_id"]: row for row in review.to_dict(orient="records")}
    assert by_id["steady"]["review_verdict"] == "research_watchlist"
    assert by_id["fragile"]["review_verdict"] == "blocked_by_conservative_rules"
    assert "weak_validation_window" in by_id["fragile"]["failure_reasons"]
    assert "positive_evidence_asset_concentrated" in by_id["fragile"]["failure_reasons"]
    assert "extreme_row_drawdown" in by_id["fragile"]["failure_reasons"]

    detail_csv = tmp_path / "detail.csv"
    detail.to_csv(detail_csv, index=False)
    out = tmp_path / "review"
    payload = review_factor_candidate_sensitivity(
        detail_csv=detail_csv,
        out_dir=out,
        rules=ReviewRules(min_completed_rows=3),
    )
    assert payload["artifact_type"] == "randyslab_factor_candidate_review"
    assert payload["safety"]["does_not_admit_factors"] is True
    assert (out / "factor_candidate_review.csv").exists()
    assert (out / "factor_candidate_review_summary.json").exists()
    assert (out / "factor_candidate_review_report.md").exists()
    report = (out / "factor_candidate_review_report.md").read_text(encoding="utf-8")
    assert "not a runtime publish payload" in report
    assert "research_watchlist" in report


def test_review_factor_candidate_sensitivity_reviews_drawdown_variants_separately() -> None:
    detail = pd.DataFrame(
        [
            {
                "candidate_id": "combo",
                "formula": "equal_weight_mean(zscore(ema(volume,48),120),zscore(std(close,48),144))",
                "variant_id": variant_id,
                "exposure_cap": exposure_cap,
                "volatility_cap_label": "none",
                "drawdown_stop_label": drawdown_stop_label,
                "drawdown_stop_max": drawdown_stop_max,
                "drawdown_stop_cooldown_bars": 0,
                "asset": asset,
                "window": window,
                "threshold": 0.5,
                "signal_mode": "long_flat",
                "status": "completed",
                "sharpe": sharpe,
                "cagr": 0.2,
                "max_dd": max_dd,
            }
            for variant_id, exposure_cap, drawdown_stop_label, drawdown_stop_max, sharpe, max_dd in [
                ("full_risk", 1.0, "none", "", 0.7, 0.9),
                ("half_risk", 0.5, "none", "", 0.7, 0.2),
                ("stopped_full_risk", 1.0, "dd_stop_10", 0.1, 0.7, 0.2),
            ]
            for asset in ["BTC", "ETH", "SOL"]
            for window in ["training", "validation", "blind"]
        ]
    )

    review = build_factor_candidate_review(
        detail,
        ReviewRules(
            min_completed_rows=3,
            max_positive_asset_share=1.0,
            max_mean_max_dd=0.45,
            max_worst_max_dd=0.8,
        ),
    )

    assert len(review) == 3
    by_variant = {row["variant_id"]: row for row in review.to_dict(orient="records")}
    assert by_variant["full_risk"]["review_verdict"] == "blocked_by_conservative_rules"
    assert "extreme_row_drawdown" in by_variant["full_risk"]["failure_reasons"]
    assert by_variant["half_risk"]["review_verdict"] == "research_watchlist"
    assert by_variant["stopped_full_risk"]["review_verdict"] == "research_watchlist"
    assert by_variant["stopped_full_risk"]["drawdown_stop_label"] == "dd_stop_10"


def test_review_factor_candidate_sensitivity_allows_declared_single_asset_scope(tmp_path) -> None:
    detail = pd.DataFrame(
        [
            {
                "candidate_id": "btc_scoped",
                "formula": "zscore(ema(volume,48),120)",
                "intended_scope": "BTCUSDT_4h",
                "applicability_hypothesis": "BTCUSDT 4h scoped participation research.",
                "out_of_scope_policy": "diagnostic_only",
                "asset": "BTCUSDT",
                "window": window,
                "threshold": 0.5,
                "status": "completed",
                "sharpe": 0.6,
                "cagr": 0.2,
                "max_dd": 0.2,
            }
            for window in ["training", "validation", "blind"]
        ]
    )

    review = build_factor_candidate_review(
        detail,
        ReviewRules(min_completed_rows=3, scope_mode="declared"),
    )

    row = review.iloc[0].to_dict()
    assert row["review_verdict"] == "research_watchlist"
    assert row["intended_scope"] == "BTCUSDT_4h"
    assert row["applicability_hypothesis"] == "BTCUSDT 4h scoped participation research."
    assert row["scope_mode"] == "declared"
    assert row["scope_asset_count"] == 1
    assert "too_few_positive_assets" not in row["failure_reasons"]
    assert "positive_evidence_asset_concentrated" not in row["failure_reasons"]

    detail_csv = tmp_path / "declared_scope_detail.csv"
    detail.to_csv(detail_csv, index=False)
    out = tmp_path / "declared_scope_review"
    review_factor_candidate_sensitivity(
        detail_csv=detail_csv,
        out_dir=out,
        rules=ReviewRules(min_completed_rows=3, scope_mode="declared"),
    )

    csv = pd.read_csv(out / "factor_candidate_review.csv")
    assert csv.iloc[0]["applicability_hypothesis"] == "BTCUSDT 4h scoped participation research."
    report = (out / "factor_candidate_review_report.md").read_text(encoding="utf-8")
    assert "BTCUSDT_4h" in report
    assert "BTCUSDT 4h scoped participation research." in report
    assert "diagnostic_only" in report


def test_review_factor_candidate_sensitivity_declared_scope_caps_completed_row_floor_to_group_size() -> None:
    detail = pd.DataFrame(
        [
            {
                "candidate_id": "btc_scoped",
                "formula": "zscore(ema(volume,48),120)",
                "variant_id": "thr_0p5_long_flat_cap_1p0_none",
                "exposure_cap": 1.0,
                "volatility_cap_label": "none",
                "intended_scope": "BTCUSDT_4h",
                "applicability_hypothesis": "BTCUSDT 4h scoped participation research.",
                "out_of_scope_policy": "diagnostic_only",
                "asset": "BTCUSDT",
                "window": window,
                "threshold": 0.5,
                "status": "completed",
                "sharpe": 0.7,
                "cagr": 0.2,
                "max_dd": 0.2,
            }
            for window in ["all", "training", "validation", "long", "blind"]
        ]
    )

    review = build_factor_candidate_review(detail, ReviewRules(scope_mode="declared"))

    row = review.iloc[0].to_dict()
    assert row["completed_rows"] == 5
    assert row["effective_min_completed_rows"] == 5
    assert row["review_verdict"] == "research_watchlist"
    assert "too_few_completed_rows" not in row["failure_reasons"]
