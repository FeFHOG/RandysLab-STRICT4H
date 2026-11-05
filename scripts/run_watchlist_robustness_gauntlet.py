from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randyslab import trial_ledger
from randyslab.factor_candidate_robustness import (
    build_default_watchlist_stress_scenarios,
    parse_variant_specs,
    run_watchlist_robustness_gauntlet,
)

DEFAULT_VARIANTS = [
    "thr_0p5_long_flat_cap_0p5_none_dd_stop_35_cd_42",
    "thr_0p5_long_flat_cap_0p5_none_none",
    "thr_0p5_long_flat_cap_1p0_rvz_lte_1p5_none",
    "thr_0p5_long_flat_cap_1p0_none_none",
    "thr_0p5_long_short_cap_0p75_none_none",
    "thr_0p5_long_flat_cap_1p0_none_dd_stop_35_cd_42",
    "thr_0p5_long_short_cap_1p0_none_none",
]


def main() -> None:
    trial_ledger.configure_default(Path(__file__).resolve().parents[1], context={"entry_point": Path(__file__).stem})
    parser = argparse.ArgumentParser(
        description="Run research-only robustness stresses for selected selector v0.8.2 watchlist variants."
    )
    parser.add_argument("--config", default="configs/strict4h.yaml")
    parser.add_argument(
        "--candidates",
        default="reports/factor_candidate_combination_exports/"
        "selector_v082_participation_realized_vol_drawdown_reduction/factor_candidate_combo.jsonl",
    )
    parser.add_argument(
        "--review-csv",
        default="reports/factor_candidate_review/selector_v082_combo_drawdown_deep_mitigation/"
        "deep_mitigation_variant_ranking.csv",
    )
    parser.add_argument(
        "--out",
        default="reports/factor_candidate_robustness/selector_v082_combo_watchlist_robustness_gauntlet",
    )
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        help="Asset spec as SYMBOL:OHLCV_CSV:FUNDING_CSV. Repeat for multiple assets.",
    )
    parser.add_argument(
        "--variant-id",
        action="append",
        default=[],
        help="Variant id to stress. Repeat for multiple variants. Defaults to the required watchlist cohort.",
    )
    args = parser.parse_args()

    assets = [_parse_asset(value) for value in args.asset] or _default_assets()
    review = pd.read_csv(args.review_csv)
    variant_ids = args.variant_id or DEFAULT_VARIANTS
    variants = parse_variant_specs(review, variant_ids)
    scenarios = build_default_watchlist_stress_scenarios(assets)
    payload = run_watchlist_robustness_gauntlet(
        config_path=args.config,
        candidate_jsonl=args.candidates,
        assets=assets,
        variants=variants,
        scenarios=scenarios,
        out_dir=args.out,
    )
    print(
        json.dumps(
            {
                "artifact_type": payload["artifact_type"],
                "detail_row_count": payload["detail_row_count"],
                "scenario_summary_count": payload["scenario_summary_count"],
                "variant_count": payload["variant_count"],
                "out": str(Path(args.out).resolve()),
            },
            indent=2,
        )
    )


def _parse_asset(raw: str) -> dict[str, str]:
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError("--asset must be SYMBOL:OHLCV_CSV:FUNDING_CSV")
    return {"symbol": parts[0], "ohlcv_csv": parts[1], "funding_csv": parts[2]}


def _default_assets() -> list[dict[str, str]]:
    return [
        {"symbol": "BTCUSDT", "ohlcv_csv": "data/BTCUSDT_4h.csv", "funding_csv": "data/BTCUSDT_funding.csv"},
        {"symbol": "ETHUSDT", "ohlcv_csv": "data/ETHUSDT_4h.csv", "funding_csv": "data/ETHUSDT_funding.csv"},
        {"symbol": "SOLUSDT", "ohlcv_csv": "data/SOLUSDT_4h.csv", "funding_csv": "data/SOLUSDT_funding.csv"},
        {"symbol": "BNBUSDT", "ohlcv_csv": "data/BNBUSDT_4h.csv", "funding_csv": "data/BNBUSDT_funding.csv"},
        {"symbol": "AVAXUSDT", "ohlcv_csv": "data/AVAXUSDT_4h.csv", "funding_csv": "data/AVAXUSDT_funding.csv"},
    ]


if __name__ == "__main__":
    main()
