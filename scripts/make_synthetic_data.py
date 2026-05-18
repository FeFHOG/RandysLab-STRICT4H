from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--start", default="2019-09-08")
    ap.add_argument("--end", default="2025-11-24")
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    idx = pd.date_range(args.start, args.end, freq="4h", tz="UTC", inclusive="left")
    drift = 0.00008
    vol = 0.025
    close = 10_000 * np.exp(np.cumsum(drift + rng.normal(0, vol, len(idx))))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.01, len(idx)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.01, len(idx)))
    volume = rng.lognormal(10, 0.5, len(idx))
    funding_idx = pd.date_range(args.start, args.end, freq="8h", tz="UTC", inclusive="left")
    funding = rng.normal(0.00005, 0.0002, len(funding_idx))

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": idx, "open": open_, "high": high, "low": low, "close": close, "volume": volume}).to_csv(
        out / f"{args.symbol}_4h.csv", index=False
    )
    pd.DataFrame({"timestamp": funding_idx, "funding_rate": funding}).to_csv(out / f"{args.symbol}_funding.csv", index=False)


if __name__ == "__main__":
    main()
