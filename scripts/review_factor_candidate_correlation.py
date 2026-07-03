from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randyslab.factor_candidate_correlation import run_factor_candidate_correlation_review


def main() -> None:
    parser = argparse.ArgumentParser(description="Review factor candidate correlation and bundle redundancy.")
    parser.add_argument("--config", default="configs/strict4h.yaml")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", default="reports/factor_candidate_correlation")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--ohlcv-csv", default="data/BTCUSDT_4h.csv")
    parser.add_argument("--funding-csv", default="data/BTCUSDT_funding.csv")
    parser.add_argument("--high-corr-threshold", type=float, default=0.80)
    args = parser.parse_args()
    payload = run_factor_candidate_correlation_review(
        config_path=args.config,
        candidate_jsonl=args.candidates,
        out_dir=args.out,
        ohlcv_csv=args.ohlcv_csv,
        funding_csv=args.funding_csv,
        symbol=args.symbol,
        high_corr_threshold=args.high_corr_threshold,
    )
    print(
        json.dumps(
            {
                "artifact_type": payload["artifact_type"],
                "candidate_count": payload["candidate_count"],
                "bundle_count": payload["bundle_count"],
                "high_corr_pair_count": payload["high_corr_pair_count"],
                "out": str(Path(args.out).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
