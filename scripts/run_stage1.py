from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoquant.config import load_settings
from autoquant.data import align_funding_to_ohlcv, read_funding, read_ohlcv, slice_window
from autoquant.optimize import search


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/strict4h.yaml")
    ap.add_argument("--out", default="reports/stage1")
    args = ap.parse_args()

    s = load_settings(args.config)
    ohlcv_all = read_ohlcv(s.paths.ohlcv_csv)
    funding_all = align_funding_to_ohlcv(ohlcv_all, read_funding(s.paths.funding_csv))
    ohlcv = slice_window(ohlcv_all, s.windows.training_start, s.windows.training_end)
    funding = funding_all.reindex(ohlcv.index)

    results = search(ohlcv, funding, s.costs, s.n_trials, s.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    serial = results.copy()
    serial["theta"] = serial["theta"].map(json.dumps)
    serial.to_csv(out / "trials.csv", index=False)
    candidates = results.head(s.screening.top_k_stage1)["theta"].tolist()
    (out / "top_candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    print(results.head(10)[["trial", "score", "monthly_geom", "sharpe", "max_dd", "trades"]].to_string(index=False))


if __name__ == "__main__":
    main()
