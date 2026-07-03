from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randyslab.factor_candidate_review import ReviewRules, review_factor_candidate_sensitivity


def main() -> None:
    default_rules = ReviewRules()
    parser = argparse.ArgumentParser(
        description="Conservatively review strict factor-candidate sensitivity evidence."
    )
    parser.add_argument("--detail-csv", required=True, help="factor_candidate_sensitivity_detail.csv path.")
    parser.add_argument("--out", default="reports/factor_candidate_review")
    parser.add_argument("--min-mean-sharpe", type=float, default=default_rules.min_mean_sharpe)
    parser.add_argument("--min-median-sharpe", type=float, default=default_rules.min_median_sharpe)
    parser.add_argument("--min-positive-row-share", type=float, default=default_rules.min_positive_row_share)
    parser.add_argument(
        "--scope-mode",
        choices=["multi_asset", "declared"],
        default=default_rules.scope_mode,
        help="Use declared for candidates with explicit intended_scope metadata.",
    )
    parser.add_argument("--max-mean-max-dd", type=float, default=default_rules.max_mean_max_dd)
    parser.add_argument("--max-worst-max-dd", type=float, default=default_rules.max_worst_max_dd)
    args = parser.parse_args()

    payload = review_factor_candidate_sensitivity(
        detail_csv=args.detail_csv,
        out_dir=args.out,
        rules=ReviewRules(
            min_mean_sharpe=args.min_mean_sharpe,
            min_median_sharpe=args.min_median_sharpe,
            min_positive_row_share=args.min_positive_row_share,
            scope_mode=args.scope_mode,
            max_mean_max_dd=args.max_mean_max_dd,
            max_worst_max_dd=args.max_worst_max_dd,
        ),
    )
    print(
        json.dumps(
            {
                "artifact_type": payload["artifact_type"],
                "candidate_count": payload["candidate_count"],
                "verdict_counts": payload["verdict_counts"],
                "out": str(Path(args.out).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
