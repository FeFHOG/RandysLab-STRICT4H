from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


STRATEGIES = {
    "ma_cross": ROOT / "configs" / "strategies" / "ma_cross.json",
    "bb_breakout": ROOT / "configs" / "strategies" / "bb_breakout.json",
}


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    for name, theta_path in STRATEGIES.items():
        out_dir = ROOT / "reports" / name
        run(
            [
                sys.executable,
                "scripts/run_backtest.py",
                "--theta",
                str(theta_path),
                "--window",
                "all",
                "--out",
                str(out_dir),
            ]
        )
        run(
            [
                sys.executable,
                "scripts/make_dashboard.py",
                "--theta",
                str(theta_path),
                "--window",
                "all",
                "--out",
                str(ROOT / "reports" / f"dashboard_{name}.html"),
            ]
        )
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        print(
            f"{name}: CAGR={summary['cagr']:.4f}, "
            f"Sharpe={summary['sharpe']:.4f}, "
            f"MaxDD={summary['max_dd']:.4f}, "
            f"Trades={summary['trades']}"
        )


if __name__ == "__main__":
    main()

