from __future__ import annotations

import json

import pandas as pd

from randyslab.factor_candidate_robustness import (
    StressScenario,
    build_default_watchlist_stress_scenarios,
    build_selector_v082_crash_hypothesis_scenarios,
    build_robustness_summaries,
    parse_variant_specs,
    run_watchlist_robustness_gauntlet,
)


def _write_market_fixture(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    idx = pd.date_range("2024-01-01", periods=260, freq="4h", tz="UTC")
    close = pd.Series([100 + i * 0.5 for i in range(len(idx))], index=idx, dtype=float)
    volume = pd.Series([1000 + (i % 30) * 20 for i in range(len(idx))], index=idx, dtype=float)
    pd.DataFrame(
        {
            "timestamp": idx,
            "open": close - 0.25,
            "high": close + 0.75,
            "low": close - 0.75,
            "close": close,
            "volume": volume,
        }
    ).to_csv(data_dir / "BTCUSDT_4h.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=130, freq="8h", tz="UTC"),
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
  training_end: "2024-01-16"
  validation_start: "2024-01-16"
  validation_end: "2024-01-26"
  long_start: "2024-01-01"
  long_end: "2024-02-13"
  blind_start: "2024-01-26"
  blind_end: "2024-02-13"
""".strip(),
        encoding="utf-8",
    )


def test_parse_variant_specs_uses_review_metadata() -> None:
    review = pd.DataFrame(
        [
            {
                "variant_id": "thr_0p5_long_flat_cap_0p5_none_dd_stop_35_cd_42",
                "threshold": 0.5,
                "signal_mode": "long_flat",
                "exposure_cap": 0.5,
                "volatility_cap_label": "none",
                "volatility_cap_formula": "",
                "volatility_cap_max": "",
                "drawdown_stop_label": "dd_stop_35_cd_42",
                "drawdown_stop_max": 0.35,
                "drawdown_stop_cooldown_bars": 42,
                "mitigation_family": "drawdown_stop_cooldown",
            }
        ]
    )

    variants = parse_variant_specs(
        review,
        ["thr_0p5_long_flat_cap_0p5_none_dd_stop_35_cd_42"],
    )

    assert len(variants) == 1
    assert variants[0].threshold == 0.5
    assert variants[0].signal_mode == "long_flat"
    assert variants[0].exposure_cap == 0.5
    assert variants[0].drawdown_stop_max == 0.35
    assert variants[0].drawdown_stop_cooldown_bars == 42


def test_default_stress_scenarios_cover_required_families() -> None:
    scenarios = build_default_watchlist_stress_scenarios(
        [
            {"symbol": "BTCUSDT", "ohlcv_csv": "data/BTCUSDT_4h.csv", "funding_csv": "data/BTCUSDT_funding.csv"},
            {"symbol": "ETHUSDT", "ohlcv_csv": "data/ETHUSDT_4h.csv", "funding_csv": "data/ETHUSDT_funding.csv"},
            {"symbol": "SOLUSDT", "ohlcv_csv": "data/SOLUSDT_4h.csv", "funding_csv": "data/SOLUSDT_funding.csv"},
        ]
    )

    families = {scenario.stress_family for scenario in scenarios}

    assert "higher_costs" in families
    assert "funding" in families
    assert "combined_harsh_costs" in families
    assert "crash_period" in families
    assert "asset_exclusion" in families
    assert "validation_blind" in families
    assert any(scenario.asset_exclusions == ["BTCUSDT"] for scenario in scenarios)


def test_selector_v082_crash_hypothesis_scenarios_add_pair_and_core_asset_stresses() -> None:
    scenarios = build_selector_v082_crash_hypothesis_scenarios(
        [
            {"symbol": "BTCUSDT", "ohlcv_csv": "data/BTCUSDT_4h.csv", "funding_csv": "data/BTCUSDT_funding.csv"},
            {"symbol": "ETHUSDT", "ohlcv_csv": "data/ETHUSDT_4h.csv", "funding_csv": "data/ETHUSDT_funding.csv"},
            {"symbol": "SOLUSDT", "ohlcv_csv": "data/SOLUSDT_4h.csv", "funding_csv": "data/SOLUSDT_funding.csv"},
            {"symbol": "BNBUSDT", "ohlcv_csv": "data/BNBUSDT_4h.csv", "funding_csv": "data/BNBUSDT_funding.csv"},
            {"symbol": "AVAXUSDT", "ohlcv_csv": "data/AVAXUSDT_4h.csv", "funding_csv": "data/AVAXUSDT_funding.csv"},
        ]
    )

    by_id = {scenario.scenario_id: scenario for scenario in scenarios}

    assert len(by_id) == len(scenarios)
    assert "exclude_sol_avax" in by_id
    assert by_id["exclude_sol_avax"].asset_exclusions == ["SOLUSDT", "AVAXUSDT"]
    assert by_id["validation_exclude_sol_avax"].windows == ["validation"]
    assert by_id["btc_eth_only_all_windows"].asset_exclusions == ["SOLUSDT", "BNBUSDT", "AVAXUSDT"]
    assert by_id["btc_eth_only_validation"].windows == ["validation"]
    assert by_id["crash_2020_btc_eth_only"].custom_start == "2020-02-01"
    assert by_id["crash_2020_btc_eth_only"].custom_end == "2020-05-01"


def test_build_robustness_summaries_assigns_required_fragility_labels() -> None:
    detail = pd.DataFrame(
        [
            {
                "scenario_id": "base_all_windows",
                "stress_family": "baseline",
                "variant_id": "v",
                "asset": "BTCUSDT",
                "window": "validation",
                "status": "completed",
                "sharpe": -0.4,
                "max_dd": 0.35,
            },
            {
                "scenario_id": "base_all_windows",
                "stress_family": "baseline",
                "variant_id": "v",
                "asset": "SOLUSDT",
                "window": "blind",
                "status": "completed",
                "sharpe": 0.1,
                "max_dd": 0.55,
            },
            {
                "scenario_id": "fee_slip_2x",
                "stress_family": "higher_costs",
                "variant_id": "v",
                "asset": "ETHUSDT",
                "window": "training",
                "status": "completed",
                "sharpe": 0.05,
                "max_dd": 0.40,
            },
            {
                "scenario_id": "funding_2x",
                "stress_family": "funding",
                "variant_id": "v",
                "asset": "ETHUSDT",
                "window": "training",
                "status": "completed",
                "sharpe": 0.02,
                "max_dd": 0.40,
            },
            {
                "scenario_id": "crash_2021_2022",
                "stress_family": "crash_period",
                "variant_id": "v",
                "asset": "AVAXUSDT",
                "window": "crash_2021_2022",
                "status": "completed",
                "sharpe": -0.1,
                "max_dd": 0.72,
            },
            {
                "scenario_id": "exclude_solusdt",
                "stress_family": "asset_exclusion",
                "variant_id": "v",
                "asset": "BTCUSDT",
                "window": "training",
                "status": "completed",
                "sharpe": 0.01,
                "max_dd": 0.30,
            },
        ]
    )

    scenario_summary, variant_ranking = build_robustness_summaries(detail)
    labels = set("|".join(scenario_summary["robustness_labels"].fillna("")).split("|"))
    final_labels = set(variant_ranking.iloc[0]["robustness_labels"].split("|"))

    assert "btc_weakness" in labels
    assert "sol_avax_concentration" in labels
    assert "validation_weakness" in labels
    assert "blind_weakness" in labels
    assert "crash_period_drawdown" in labels
    assert "fee_fragility" in labels
    assert "funding_fragility" in labels
    assert "asset_exclusion_fragility" in labels
    assert final_labels >= {
        "btc_weakness",
        "crash_period_drawdown",
        "fee_fragility",
        "funding_fragility",
        "asset_exclusion_fragility",
    }
    assert variant_ranking.iloc[0]["conservative_verdict"] == "blocked_pending_new_hypotheses"


def test_build_robustness_summaries_uses_declared_scope_for_hard_gate() -> None:
    detail = pd.DataFrame(
        [
            {
                "candidate_id": "qr_scoped",
                "formula": "zscore(close,24)",
                "scenario_id": "base_all_windows",
                "stress_family": "baseline",
                "variant_id": "v",
                "intended_scope": "BTCUSDT_4h",
                "asset": "BTCUSDT",
                "window": "validation",
                "status": "completed",
                "sharpe": 0.75,
                "max_dd": 0.20,
            },
            {
                "candidate_id": "qr_scoped",
                "formula": "zscore(close,24)",
                "scenario_id": "base_all_windows",
                "stress_family": "baseline",
                "variant_id": "v",
                "intended_scope": "BTCUSDT_4h",
                "asset": "BTCUSDT",
                "window": "blind",
                "status": "completed",
                "sharpe": 0.80,
                "max_dd": 0.22,
            },
            {
                "candidate_id": "qr_scoped",
                "formula": "zscore(close,24)",
                "scenario_id": "base_all_windows",
                "stress_family": "baseline",
                "variant_id": "v",
                "intended_scope": "BTCUSDT_4h",
                "asset": "SOLUSDT",
                "window": "validation",
                "status": "completed",
                "sharpe": -1.20,
                "max_dd": 0.65,
            },
            {
                "candidate_id": "qr_scoped",
                "formula": "zscore(close,24)",
                "scenario_id": "exclude_btcusdt",
                "stress_family": "asset_exclusion",
                "variant_id": "v",
                "intended_scope": "BTCUSDT_4h",
                "asset_exclusions": "BTCUSDT",
                "asset": "SOLUSDT",
                "window": "validation",
                "status": "completed",
                "sharpe": -1.00,
                "max_dd": 0.70,
            },
        ]
    )

    scenario_summary, variant_ranking = build_robustness_summaries(detail)
    by_scenario = scenario_summary.set_index("scenario_id")
    ranking = variant_ranking.iloc[0]

    assert by_scenario.loc["base_all_windows", "conservative_verdict"] == "research_watchlist"
    assert by_scenario.loc["base_all_windows", "diagnostic_verdict"] == "blocked_by_robustness_stress"
    assert bool(by_scenario.loc["base_all_windows", "scope_hard_gate"]) is True
    assert by_scenario.loc["exclude_btcusdt", "conservative_verdict"] == "diagnostic_only_out_of_scope"
    assert bool(by_scenario.loc["exclude_btcusdt", "scope_hard_gate"]) is False
    assert ranking["conservative_verdict"] == "research_watchlist"
    assert ranking["stress_survival_count"] == 1
    assert ranking["stress_count"] == 1
    assert ranking["diagnostic_scenario_count"] == 1
    assert ranking["failure_reasons"] == ""
    assert "asset_exclusion_fragility" in ranking["robustness_labels"].split("|")


def test_run_watchlist_robustness_gauntlet_writes_research_artifacts(tmp_path) -> None:
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
    review = pd.DataFrame(
        [
            {
                "variant_id": "thr_0p5_long_flat_cap_0p5_none_none",
                "threshold": 0.5,
                "signal_mode": "long_flat",
                "exposure_cap": 0.5,
                "volatility_cap_label": "none",
                "volatility_cap_formula": "",
                "volatility_cap_max": "",
                "drawdown_stop_label": "none",
                "drawdown_stop_max": "",
                "drawdown_stop_cooldown_bars": 0,
                "mitigation_family": "capped_long_flat",
            }
        ]
    )
    variants = parse_variant_specs(review, ["thr_0p5_long_flat_cap_0p5_none_none"])

    payload = run_watchlist_robustness_gauntlet(
        config_path=tmp_path / "strict4h.yaml",
        candidate_jsonl=candidates,
        assets=[
            {
                "symbol": "BTCUSDT",
                "ohlcv_csv": str(tmp_path / "data" / "BTCUSDT_4h.csv"),
                "funding_csv": str(tmp_path / "data" / "BTCUSDT_funding.csv"),
            }
        ],
        variants=variants,
        scenarios=[
            StressScenario(
                scenario_id="validation_only",
                stress_family="validation_blind",
                label="Validation only",
                windows=["validation"],
            )
        ],
        out_dir=tmp_path / "robustness",
    )

    assert payload["artifact_type"] == "randyslab_watchlist_robustness_gauntlet"
    assert payload["safety"]["research_only"] is True
    assert payload["detail_row_count"] == 1
    assert (tmp_path / "robustness" / "watchlist_robustness_detail.csv").exists()
    assert (tmp_path / "robustness" / "watchlist_robustness_scenario_summary.csv").exists()
    assert (tmp_path / "robustness" / "watchlist_robustness_variant_ranking.csv").exists()
    assert (tmp_path / "robustness" / "watchlist_robustness_report.md").exists()


def test_run_watchlist_robustness_gauntlet_skips_empty_custom_windows(tmp_path) -> None:
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
    review = pd.DataFrame(
        [
            {
                "variant_id": "thr_0p5_long_flat_cap_0p5_none_none",
                "threshold": 0.5,
                "signal_mode": "long_flat",
                "exposure_cap": 0.5,
                "volatility_cap_label": "none",
                "volatility_cap_formula": "",
                "volatility_cap_max": "",
                "drawdown_stop_label": "none",
                "drawdown_stop_max": "",
                "drawdown_stop_cooldown_bars": 0,
                "mitigation_family": "capped_long_flat",
            }
        ]
    )
    variants = parse_variant_specs(review, ["thr_0p5_long_flat_cap_0p5_none_none"])

    payload = run_watchlist_robustness_gauntlet(
        config_path=tmp_path / "strict4h.yaml",
        candidate_jsonl=candidates,
        assets=[
            {
                "symbol": "BTCUSDT",
                "ohlcv_csv": str(tmp_path / "data" / "BTCUSDT_4h.csv"),
                "funding_csv": str(tmp_path / "data" / "BTCUSDT_funding.csv"),
            }
        ],
        variants=variants,
        scenarios=[
            StressScenario(
                scenario_id="pre_data_crash",
                stress_family="crash_period",
                label="Pre-data crash",
                windows=[],
                custom_start="2023-01-01",
                custom_end="2023-02-01",
                custom_window_label="pre_data_crash",
            )
        ],
        out_dir=tmp_path / "empty_custom",
    )

    assert payload["detail_row_count"] == 0
