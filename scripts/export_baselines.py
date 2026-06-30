from __future__ import annotations

# ruff: noqa: I001

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randyslab.baseline_export import run_baseline_export


DEFAULT_STRATEGIES = [
    "configs/strategies/ma_cross.json",
    "configs/strategies/bb_breakout.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export RandysLab baseline strategies as QuantumRandy comparison artifacts."
    )
    parser.add_argument("--config", default="configs/strict4h.yaml")
    parser.add_argument("--strategy", action="append", dest="strategies", help="Strategy theta JSON path.")
    parser.add_argument("--window", choices=["training", "validation", "long", "blind", "all"], default="all")
    parser.add_argument("--out", default="reports/quantumrandy_baselines")
    parser.add_argument("--no-ledgers", action="store_true", help="Skip per-strategy ledger CSV files.")
    args = parser.parse_args()

    payload = run_baseline_export(
        config_path=args.config,
        strategy_paths=args.strategies or DEFAULT_STRATEGIES,
        window=args.window,
        out_dir=args.out,
        write_ledgers=not args.no_ledgers,
    )
    print(json.dumps({"out_dir": args.out, "strategies": len(payload["strategies"])}, indent=2))


if __name__ == "__main__":
    main()
