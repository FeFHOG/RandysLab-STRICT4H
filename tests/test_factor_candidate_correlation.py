from __future__ import annotations

import json

import pandas as pd

from randyslab.factor_candidate_correlation import run_factor_candidate_correlation_review


def _write_market_fixture(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    idx = pd.date_range("2024-01-01", periods=220, freq="4h", tz="UTC")
    close = pd.Series([100 + i + (i % 9) * 0.2 for i in range(len(idx))], index=idx, dtype=float)
    volume = pd.Series([1000 + (i % 17) * 20 for i in range(len(idx))], index=idx, dtype=float)
    pd.DataFrame(
        {
            "timestamp": idx,
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": volume,
        }
    ).to_csv(data_dir / "BTCUSDT_4h.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=110, freq="8h", tz="UTC"),
            "funding_rate": [0.0001 * ((i % 5) - 2) for i in range(110)],
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


def test_run_factor_candidate_correlation_review_writes_redundancy_artifacts(tmp_path) -> None:
    _write_market_fixture(tmp_path)
    _write_config(tmp_path)
    candidates = tmp_path / "factor_candidates.jsonl"
    records = [
        {
            "candidate_id": "liq",
            "formula": "zscore(div(volume,sma(volume,12)),24)",
            "formula_family": "liquidity_participation",
            "intended_scope": "BTCUSDT_4h",
        },
        {
            "candidate_id": "trend",
            "formula": "zscore(ret(close,3),24)",
            "formula_family": "price_trend",
            "intended_scope": "BTCUSDT_4h",
        },
        {
            "candidate_id": "bundle",
            "formula": "equal_weight_mean(zscore(div(volume,sma(volume,12)),24),zscore(ret(close,3),24))",
            "formula_family": "scoped_equal_weight_bundle",
            "component_candidate_ids": ["liq", "trend"],
            "component_formulas": [
                "zscore(div(volume,sma(volume,12)),24)",
                "zscore(ret(close,3),24)",
            ],
            "combination_method": "equal_weight_mean",
            "intended_scope": "BTCUSDT_4h",
        },
    ]
    candidates.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    out = tmp_path / "correlation"
    payload = run_factor_candidate_correlation_review(
        config_path=tmp_path / "strict4h.yaml",
        candidate_jsonl=candidates,
        out_dir=out,
        ohlcv_csv=tmp_path / "data" / "BTCUSDT_4h.csv",
        funding_csv=tmp_path / "data" / "BTCUSDT_funding.csv",
        symbol="BTCUSDT",
    )

    assert payload["artifact_type"] == "randyslab_factor_candidate_correlation_review"
    assert payload["safety"]["research_only"] is True
    assert payload["candidate_count"] == 3
    assert payload["bundle_count"] == 1
    assert (out / "factor_candidate_pairwise_correlation.csv").exists()
    assert (out / "factor_candidate_bundle_redundancy.csv").exists()
    assert (out / "factor_candidate_correlation_summary.json").exists()
    assert (out / "factor_candidate_correlation_report.md").exists()

    pairwise = pd.read_csv(out / "factor_candidate_pairwise_correlation.csv")
    assert {"left_candidate_id", "right_candidate_id", "corr", "abs_corr", "overlap_count"}.issubset(pairwise.columns)
    assert len(pairwise) >= 1

    bundle = pd.read_csv(out / "factor_candidate_bundle_redundancy.csv")
    assert bundle.iloc[0]["bundle_candidate_id"] == "bundle"
    assert bundle.iloc[0]["component_count"] == 2
    assert bundle.iloc[0]["redundancy_verdict"] in {
        "diversified_enough_for_research",
        "redundant_research_memory_only",
    }

    report = (out / "factor_candidate_correlation_report.md").read_text(encoding="utf-8")
    assert "not factor admission" in report
    assert "BTCUSDT" in report
