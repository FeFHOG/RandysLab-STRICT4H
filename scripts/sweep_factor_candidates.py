from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from randyslab.factor_candidate_sensitivity import run_factor_candidate_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep QuantumRandy factor candidates across strict RandysLab assets, windows, and thresholds."
    )
    parser.add_argument("--config", default="configs/strict4h.yaml")
    parser.add_argument("--candidates", required=True, help="QuantumRandy factor_candidates.jsonl export.")
    parser.add_argument("--out", default="reports/factor_candidate_sensitivity")
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        help="Asset spec as SYMBOL:OHLCV_CSV:FUNDING_CSV. Repeat for multiple assets.",
    )
    parser.add_argument(
        "--window",
        action="append",
        default=[],
        choices=["training", "validation", "long", "blind", "all"],
        help="Window to include. Repeat for multiple windows. Defaults to training, validation, blind.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        default=[],
        help="Absolute factor threshold. Repeat for multiple thresholds. Defaults to 0.0, 0.5, 1.0.",
    )
    parser.add_argument(
        "--signal-mode",
        action="append",
        choices=["long_short", "long_flat", "short_flat"],
        default=[],
        help="Signal mode to include. Repeat for multiple modes. Defaults to long_short.",
    )
    parser.add_argument(
        "--exposure-cap",
        action="append",
        type=float,
        default=[],
        help="Absolute exposure cap to include. Repeat for multiple caps. Defaults to 1.0.",
    )
    parser.add_argument(
        "--volatility-cap",
        action="append",
        default=[],
        help="Volatility cap as LABEL:FORMULA:MAX_VALUE. Repeat for multiple caps.",
    )
    parser.add_argument(
        "--drawdown-stop",
        action="append",
        default=[],
        help="Research drawdown stop as LABEL:MAX_DRAWDOWN:COOLDOWN_BARS. Repeat for multiple stops.",
    )
    args = parser.parse_args()

    payload = run_factor_candidate_sensitivity(
        config_path=args.config,
        candidate_jsonl=args.candidates,
        assets=[_parse_asset(value) for value in args.asset] or [_default_asset()],
        out_dir=args.out,
        windows=args.window or None,
        thresholds=args.threshold or None,
        signal_modes=args.signal_mode or None,
        exposure_caps=args.exposure_cap or None,
        volatility_caps=[_parse_volatility_cap(value) for value in args.volatility_cap] or None,
        drawdown_stops=[_parse_drawdown_stop(value) for value in args.drawdown_stop] or None,
    )
    print(
        json.dumps(
            {
                "artifact_type": payload["artifact_type"],
                "run_count": payload["run_count"],
                "candidate_row_count": payload["candidate_row_count"],
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


def _default_asset() -> dict[str, str]:
    return {
        "symbol": "BTC/USDT:USDT",
        "ohlcv_csv": "data/BTCUSDT_4h.csv",
        "funding_csv": "data/BTCUSDT_funding.csv",
    }


def _parse_volatility_cap(raw: str) -> dict[str, object]:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise ValueError("--volatility-cap must be LABEL:FORMULA:MAX_VALUE")
    label, formula, max_value = parts
    parsed_max = None if not formula.strip() or max_value.strip().lower() in {"", "none", "null"} else float(max_value)
    return {"label": label, "formula": formula, "max_value": parsed_max}


def _parse_drawdown_stop(raw: str) -> dict[str, object]:
    parts = raw.split(":", 2)
    if len(parts) != 3:
        raise ValueError("--drawdown-stop must be LABEL:MAX_DRAWDOWN:COOLDOWN_BARS")
    label, max_drawdown, cooldown_bars = parts
    parsed_max = None if max_drawdown.strip().lower() in {"", "none", "null"} else float(max_drawdown)
    return {"label": label, "max_drawdown": parsed_max, "cooldown_bars": int(cooldown_bars)}


if __name__ == "__main__":
    main()
