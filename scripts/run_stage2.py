from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randyslab.config import load_settings
from randyslab.data import load_market, slice_window
from randyslab.screening import robust_summary, select_stable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/strict4h.yaml")
    ap.add_argument("--candidates", default="reports/stage1/top_candidates.json")
    ap.add_argument("--out", default="reports/stage2")
    args = ap.parse_args()

    s = load_settings(args.config)
    ohlcv_all, funding_all = load_market(s.paths.ohlcv_csv, s.paths.funding_csv, s.bar_hours)
    ohlcv = slice_window(ohlcv_all, s.windows.training_start, s.windows.training_end)
    funding = funding_all.reindex(ohlcv.index)
    candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))

    summary = robust_summary(ohlcv, funding, candidates, s.costs, s.screening)
    stable = select_stable(summary, s.screening)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary.assign(theta=summary["theta"].map(json.dumps)).to_csv(out / "robust_summary.csv", index=False)
    stable.assign(theta=stable["theta"].map(json.dumps)).to_csv(out / "stable_candidates.csv", index=False)
    selected = stable.iloc[0]["theta"] if len(stable) else {}
    (out / "selected_theta.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(stable.drop(columns=["theta"]).to_string(index=False) if len(stable) else "No stable candidates passed.")


if __name__ == "__main__":
    main()

