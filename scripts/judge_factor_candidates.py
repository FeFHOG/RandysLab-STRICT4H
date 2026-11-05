from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randyslab import trial_ledger
from randyslab.formula_candidates import run_factor_candidate_judging


def main() -> None:
    trial_ledger.configure_default(Path(__file__).resolve().parents[1], context={"entry_point": Path(__file__).stem})
    parser = argparse.ArgumentParser(
        description="Strictly judge QuantumRandy research-only factor candidate exports."
    )
    parser.add_argument("--config", default="configs/strict4h.yaml")
    parser.add_argument(
        "--candidates",
        required=True,
        help="QuantumRandy factor_candidates.jsonl export.",
    )
    parser.add_argument("--window", choices=["training", "validation", "long", "blind", "all"], default="all")
    parser.add_argument("--out", default="reports/factor_candidate_judging")
    parser.add_argument("--signal-threshold", type=float, default=0.0)
    parser.add_argument("--signal-mode", choices=["long_short", "long_flat", "short_flat"], default="long_short")
    parser.add_argument("--ohlcv-csv", default=None, help="Override OHLCV CSV path from the config.")
    parser.add_argument("--funding-csv", default=None, help="Override funding CSV path from the config.")
    parser.add_argument("--symbol", default=None, help="Override symbol label in the output artifact.")
    parser.add_argument("--no-ledgers", action="store_true", help="Skip per-candidate ledger CSV output.")
    args = parser.parse_args()

    payload = run_factor_candidate_judging(
        config_path=args.config,
        candidate_jsonl=args.candidates,
        out_dir=args.out,
        window=args.window,
        signal_threshold=args.signal_threshold,
        signal_mode=args.signal_mode,
        write_ledgers=not args.no_ledgers,
        ohlcv_csv=args.ohlcv_csv,
        funding_csv=args.funding_csv,
        symbol=args.symbol,
    )
    print(
        json.dumps(
            {
                "artifact_type": payload["artifact_type"],
                "candidate_count": payload["candidate_count"],
                "completed_count": payload["completed_count"],
                "failed_count": payload["failed_count"],
                "out": str(Path(args.out).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
