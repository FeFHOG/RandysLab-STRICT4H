from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoquant.config import load_settings
from autoquant.data import align_funding_to_ohlcv, read_funding, read_ohlcv, slice_window
from autoquant.screening import robust_summary, select_stable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/strict4h.yaml")
    ap.add_argument("--candidates", default="reports/stage1/top_candidates.json")
    ap.add_argument("--out", default="reports/stage2")
    args = ap.parse_args()

    s = load_settings(args.config)
    ohlcv_all = read_ohlcv(s.paths.ohlcv_csv)
    funding_all = align_funding_to_ohlcv(ohlcv_all, read_funding(s.paths.funding_csv))
    ohlcv = slice_window(ohlcv_all, s.windows.training_start, s.windows.validation_end)
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
