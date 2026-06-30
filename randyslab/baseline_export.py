from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_settings
from .data import align_funding_to_ohlcv, read_funding, read_ohlcv, slice_window
from .engine import run_strict4h
from .metrics import equity_curve, summarize


def run_baseline_export(
    *,
    config_path: str | Path,
    strategy_paths: list[str | Path],
    window: str = "all",
    out_dir: str | Path,
    write_ledgers: bool = True,
) -> dict[str, Any]:
    """Export RandysLab baseline backtests as QuantumRandy comparison artifacts."""
    if not strategy_paths:
        raise ValueError("At least one strategy path is required")
    if window not in {"training", "validation", "long", "blind", "all"}:
        raise ValueError("window must be one of training, validation, long, blind, or all")

    config_path = Path(config_path).resolve()
    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings(config_path)
    ohlcv = read_ohlcv(_resolve_path(base_dir, settings.paths.ohlcv_csv))
    funding = align_funding_to_ohlcv(ohlcv, read_funding(_resolve_path(base_dir, settings.paths.funding_csv)))
    window_bounds = _window_bounds(settings, window)
    if window != "all":
        ohlcv = slice_window(ohlcv, window_bounds["start"], window_bounds["end"])
        funding = funding.reindex(ohlcv.index)

    rows: list[dict[str, Any]] = []
    strategy_exports: list[dict[str, Any]] = []
    for strategy_path in strategy_paths:
        strategy_path = Path(strategy_path)
        if not strategy_path.is_absolute():
            strategy_path = (base_dir / strategy_path).resolve()
        theta = json.loads(strategy_path.read_text(encoding="utf-8"))
        strategy_id = _strategy_id(strategy_path, theta)
        ledger = run_strict4h(ohlcv, funding, theta, settings.costs)
        metrics = summarize(ledger, settings.risk_free_rate)
        export_row = {
            "strategy_id": strategy_id,
            "strategy_path": _display_path(strategy_path, base_dir),
            "strategy": str(theta.get("strategy", "randyslab")),
            **metrics,
            "final_equity": float(equity_curve(ledger["r_net"]).iloc[-1]) if len(ledger) else 1.0,
        }
        rows.append(export_row)

        ledger_file = None
        if write_ledgers:
            ledger_file = f"{strategy_id}_ledger.csv"
            ledger.to_csv(out_dir / ledger_file, index_label="timestamp")
        strategy_exports.append(
            {
                "strategy_id": strategy_id,
                "theta": theta,
                "metrics": metrics,
                "final_equity": export_row["final_equity"],
                "ledger_file": ledger_file,
            }
        )

    summary_csv = out_dir / "baseline_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    payload = {
        "artifact_type": "randyslab_baseline_export",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "role": "baseline_control_group",
        "safety": {
            "paper_only": True,
            "no_live_orders": True,
            "no_exchange_trading_keys": True,
            "not_runtime_publish_payload": True,
        },
        "config_path": _display_path(config_path, base_dir),
        "symbol": settings.symbol,
        "exchange": settings.exchange,
        "bar_hours": settings.bar_hours,
        "window": {"name": window, **window_bounds},
        "costs": asdict(settings.costs),
        "strategies": strategy_exports,
        "artifacts": {
            "summary_csv": summary_csv.name,
            "report_md": "baseline_report.md",
        },
    }
    summary_json = out_dir / "baseline_summary.json"
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (out_dir / "baseline_report.md").write_text(render_baseline_report(payload, rows), encoding="utf-8")
    return payload


def render_baseline_report(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# RandysLab Baseline Export",
        "",
        "This artifact is a traditional-strategy control group for QuantumRandy runtime and paper observation.",
        "It is not a runtime publish payload and must not be auto-promoted into QuantumRandy.",
        "",
        "## Run",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Symbol: `{payload['symbol']}`",
        f"- Exchange: `{payload['exchange']}`",
        f"- Window: `{payload['window']['name']}`",
        f"- Bar hours: `{payload['bar_hours']}`",
        "",
        "## Strategy Summary",
        "",
        "| Strategy | Sharpe | CAGR | Max DD | Trades | Net Total | Final Equity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {strategy_id} | {sharpe:.4f} | {cagr:.4f} | {max_dd:.4f} | {trades:.0f} | "
            "{net_total:.4f} | {final_equity:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Safety Boundary",
            "",
            "- Paper/backtest comparison only.",
            "- No exchange trading credentials.",
            "- No live order placement.",
            "- No automatic QuantumRandy runtime publishing.",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_path(base_dir: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _display_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _strategy_id(strategy_path: Path, theta: dict[str, Any]) -> str:
    raw = str(theta.get("strategy_id") or strategy_path.stem).strip().lower()
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in raw)


def _window_bounds(settings: Any, window: str) -> dict[str, str | None]:
    if window == "all":
        return {"start": None, "end": None}
    return {
        "start": getattr(settings.windows, f"{window}_start"),
        "end": getattr(settings.windows, f"{window}_end"),
    }
