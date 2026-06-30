from __future__ import annotations

import json

import pandas as pd

from randyslab.baseline_export import run_baseline_export


def test_run_baseline_export_writes_quantumrandy_comparison_artifacts(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    idx = pd.date_range("2024-01-01", periods=80, freq="4h", tz="UTC")
    close = pd.Series(range(100, 180), index=idx, dtype=float)
    pd.DataFrame(
        {
            "timestamp": idx,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0,
        }
    ).to_csv(data_dir / "BTCUSDT_4h.csv", index=False)
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=40, freq="8h", tz="UTC"),
            "funding_rate": 0.0001,
        }
    ).to_csv(data_dir / "BTCUSDT_funding.csv", index=False)

    config = tmp_path / "strict4h.yaml"
    config.write_text(
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
""".strip(),
        encoding="utf-8",
    )
    strategy = tmp_path / "ma_cross.json"
    strategy.write_text(
        json.dumps({"strategy": "ma_cross", "ma_fast": 2, "ma_slow": 4, "min_hold_bars": 1}),
        encoding="utf-8",
    )

    out_dir = tmp_path / "export"
    payload = run_baseline_export(
        config_path=config,
        strategy_paths=[strategy],
        out_dir=out_dir,
        write_ledgers=True,
    )

    assert payload["artifact_type"] == "randyslab_baseline_export"
    assert payload["role"] == "baseline_control_group"
    assert payload["safety"]["not_runtime_publish_payload"] is True
    assert payload["strategies"][0]["strategy_id"] == "ma_cross"
    assert (out_dir / "baseline_summary.json").exists()
    assert (out_dir / "baseline_summary.csv").exists()
    assert (out_dir / "baseline_report.md").exists()
    assert (out_dir / "ma_cross_ledger.csv").exists()
    assert "must not be auto-promoted" in (out_dir / "baseline_report.md").read_text(encoding="utf-8")
