from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randyslab.config import CostProfile, load_settings
from randyslab.data import load_market, slice_window
from randyslab.engine import run_strict4h
from randyslab.metrics import summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/strict4h.yaml")
    ap.add_argument("--theta", default=None, help="JSON string or path to theta JSON")
    ap.add_argument("--window", choices=["training", "validation", "long", "blind", "all"], default="all")
    ap.add_argument("--out", default="reports/backtest")
    args = ap.parse_args()

    settings = load_settings(args.config)
    ohlcv, funding = load_market(settings.paths.ohlcv_csv, settings.paths.funding_csv, settings.bar_hours)

    if args.window != "all":
        start = getattr(settings.windows, f"{args.window}_start")
        end = getattr(settings.windows, f"{args.window}_end")
        ohlcv = slice_window(ohlcv, start, end)
        funding = funding.reindex(ohlcv.index)

    theta = {}
    if args.theta:
        maybe_path = Path(args.theta)
        theta = json.loads(maybe_path.read_text(encoding="utf-8") if maybe_path.exists() else args.theta)

    ledger = run_strict4h(ohlcv, funding, theta, settings.costs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(out / "ledger.csv", index_label="timestamp")
    (out / "summary.json").write_text(json.dumps(summarize(ledger, settings.risk_free_rate), indent=2), encoding="utf-8")
    (out / "theta.json").write_text(json.dumps(theta, indent=2), encoding="utf-8")
    print(json.dumps(summarize(ledger, settings.risk_free_rate), indent=2))


if __name__ == "__main__":
    main()

