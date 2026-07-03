from __future__ import annotations

import ast
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_settings
from .data import align_funding_to_ohlcv, read_funding, read_ohlcv, slice_window
from .engine import run_strict4h_signal
from .metrics import equity_curve, summarize

SUPPORTED_FUNCTIONS = {
    "zscore",
    "ema",
    "std",
    "sma",
    "delta",
    "ret",
    "neg",
    "add",
    "sub",
    "mul",
    "div",
    "abs",
    "log",
    "sqrt",
    "sign",
    "corr",
    "min",
    "max",
    "rsi",
}
SUPPORTED_FIELDS = {"open", "high", "low", "close", "volume", "funding_rate"}


def run_factor_candidate_judging(
    *,
    config_path: str | Path,
    candidate_jsonl: str | Path,
    out_dir: str | Path,
    window: str = "all",
    signal_threshold: float = 0.0,
    signal_mode: str = "long_short",
    max_exposure_abs: float = 1.0,
    volatility_cap_formula: str | None = None,
    volatility_cap_max: float | None = None,
    drawdown_stop_label: str | None = None,
    drawdown_stop_max: float | None = None,
    drawdown_stop_cooldown_bars: int = 0,
    write_ledgers: bool = True,
    ohlcv_csv: str | Path | None = None,
    funding_csv: str | Path | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    if window not in {"training", "validation", "long", "blind", "all"}:
        raise ValueError("window must be one of training, validation, long, blind, or all")

    config_path = Path(config_path).resolve()
    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    candidate_jsonl = _resolve_path(base_dir, candidate_jsonl)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    settings = load_settings(config_path)
    ohlcv_path = _resolve_path(base_dir, ohlcv_csv or settings.paths.ohlcv_csv)
    funding_path = _resolve_path(base_dir, funding_csv or settings.paths.funding_csv)
    ohlcv = read_ohlcv(ohlcv_path)
    funding = align_funding_to_ohlcv(ohlcv, read_funding(funding_path))
    window_bounds = _window_bounds(settings, window)
    if window != "all":
        ohlcv = slice_window(ohlcv, window_bounds["start"], window_bounds["end"])
        funding = funding.reindex(ohlcv.index)

    candidates = load_factor_candidate_jsonl(candidate_jsonl)
    rows: list[dict[str, Any]] = []
    candidate_exports: list[dict[str, Any]] = []
    for candidate in candidates:
        result = judge_factor_candidate(
            candidate,
            ohlcv=ohlcv,
            funding=funding,
            costs=settings.costs,
            risk_free_rate=settings.risk_free_rate,
            signal_threshold=signal_threshold,
            signal_mode=signal_mode,
            max_exposure_abs=max_exposure_abs,
            volatility_cap_formula=volatility_cap_formula,
            volatility_cap_max=volatility_cap_max,
            drawdown_stop_label=drawdown_stop_label,
            drawdown_stop_max=drawdown_stop_max,
            drawdown_stop_cooldown_bars=drawdown_stop_cooldown_bars,
        )
        ledger_file = None
        if result["status"] == "completed" and write_ledgers:
            ledger_file = f"{_safe_id(result['candidate_id'])}_ledger.csv"
            result["ledger"].to_csv(out / ledger_file, index_label="timestamp")

        row = {
            key: value
            for key, value in result.items()
            if key not in {"ledger", "candidate_record"}
        }
        row["ledger_file"] = ledger_file or ""
        rows.append(row)
        candidate_exports.append(
            {
                "candidate_id": result["candidate_id"],
                "formula": result["formula"],
                "status": result["status"],
                "failure_reasons": result["failure_reasons"],
                "metrics": result.get("metrics", {}),
                "ledger_file": ledger_file,
                "source_record": candidate,
            }
        )

    summary_csv = out / "factor_candidate_judging_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    payload = {
        "artifact_type": "randyslab_factor_candidate_judging",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "role": "strict_factor_candidate_judge",
        "safety": {
            "research_only": True,
            "paper_only": True,
            "no_live_orders": True,
            "no_exchange_trading_keys": True,
            "not_runtime_publish_payload": True,
            "does_not_update_quantumrandy_runtime": True,
        },
        "config_path": _display_path(config_path, base_dir),
        "candidate_jsonl": _display_path(candidate_jsonl, base_dir),
        "symbol": symbol or settings.symbol,
        "exchange": settings.exchange,
        "bar_hours": settings.bar_hours,
        "window": {"name": window, **window_bounds},
        "execution": {
            "execution_delay_bars": 1,
            "matching": "next_bar_t_plus_1",
            "signal_threshold": signal_threshold,
            "signal_mode": signal_mode,
            "max_exposure_abs": max_exposure_abs,
            "volatility_cap": {
                "formula": volatility_cap_formula or "",
                "max_value": volatility_cap_max,
                "active": _volatility_cap_active(volatility_cap_formula, volatility_cap_max),
            },
            "drawdown_stop": {
                "label": drawdown_stop_label
                or _drawdown_stop_label(drawdown_stop_max, drawdown_stop_cooldown_bars),
                "max_drawdown": drawdown_stop_max,
                "cooldown_bars": int(drawdown_stop_cooldown_bars or 0),
                "active": _drawdown_stop_active(drawdown_stop_max),
            },
            "signal_rule": _signal_rule(signal_mode),
        },
        "costs": asdict(settings.costs),
        "data": {
            "ohlcv_csv": _display_path(ohlcv_path, base_dir),
            "funding_csv": _display_path(funding_path, base_dir),
        },
        "candidate_count": len(candidates),
        "completed_count": sum(1 for row in rows if row["status"] == "completed"),
        "failed_count": sum(1 for row in rows if row["status"] != "completed"),
        "candidates": candidate_exports,
        "artifacts": {
            "summary_csv": summary_csv.name,
            "report_md": "factor_candidate_judging_report.md",
        },
    }
    (out / "factor_candidate_judging_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (out / "factor_candidate_judging_report.md").write_text(
        render_factor_candidate_judging_report(payload, rows),
        encoding="utf-8",
    )
    return payload


def judge_factor_candidate(
    candidate: dict[str, Any],
    *,
    ohlcv: pd.DataFrame,
    funding: pd.Series,
    costs: Any,
    risk_free_rate: float,
    signal_threshold: float = 0.0,
    signal_mode: str = "long_short",
    max_exposure_abs: float = 1.0,
    volatility_cap_formula: str | None = None,
    volatility_cap_max: float | None = None,
    drawdown_stop_label: str | None = None,
    drawdown_stop_max: float | None = None,
    drawdown_stop_cooldown_bars: int = 0,
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or candidate.get("factor_id") or "unknown")
    formula = str(candidate.get("formula", ""))
    component_formulas = _component_formulas(candidate)
    base = {
        "candidate_id": candidate_id,
        "formula": formula,
        "component_count": len(component_formulas),
        "combination_method": str(candidate.get("combination_method", "")),
        "source_formula_family": str(candidate.get("formula_family", "")),
        "source_parent_factor_id": str(candidate.get("parent_factor_id", "")),
        "source_llm_true_improved_count": int(candidate.get("llm_true_improved_count") or 0),
        "intended_scope": str(candidate.get("intended_scope", "")),
        "applicability_hypothesis": str(candidate.get("applicability_hypothesis", "")),
        "out_of_scope_policy": str(candidate.get("out_of_scope_policy", "")),
        "max_exposure_abs": float(max_exposure_abs),
        "volatility_cap_formula": volatility_cap_formula or "",
        "volatility_cap_max": volatility_cap_max if volatility_cap_max is not None else "",
        "drawdown_stop_label": drawdown_stop_label
        or _drawdown_stop_label(drawdown_stop_max, drawdown_stop_cooldown_bars),
        "drawdown_stop_max": drawdown_stop_max if drawdown_stop_max is not None else "",
        "drawdown_stop_cooldown_bars": int(drawdown_stop_cooldown_bars or 0),
        "status": "failed",
        "failure_reasons": [],
        "candidate_record": candidate,
    }
    if ohlcv.empty:
        return {**base, "failure_reasons": ["empty_window"], "metrics": {}}

    try:
        factor = evaluate_candidate_factor(candidate, ohlcv, funding)
    except (ValueError, TypeError, SyntaxError) as exc:
        return {**base, "failure_reasons": [f"formula_error:{exc}"], "metrics": {}}

    factor = factor.reindex(ohlcv.index).replace([np.inf, -np.inf], np.nan)
    valid = factor.dropna()
    if valid.empty:
        return {**base, "failure_reasons": ["factor_all_nan"], "metrics": {}}

    signal = _factor_to_signal(factor, signal_threshold=signal_threshold, signal_mode=signal_mode)
    try:
        signal = _apply_volatility_cap(
            signal,
            ohlcv,
            funding,
            volatility_cap_formula=volatility_cap_formula,
            volatility_cap_max=volatility_cap_max,
        )
    except (ValueError, TypeError, SyntaxError) as exc:
        return {**base, "failure_reasons": [f"volatility_cap_error:{exc}"], "metrics": {}}

    signal = _apply_drawdown_stop(
        signal,
        ohlcv,
        funding,
        costs=costs,
        max_exposure_abs=max_exposure_abs,
        drawdown_stop_max=drawdown_stop_max,
        drawdown_stop_cooldown_bars=drawdown_stop_cooldown_bars,
    )

    failure_reasons = []
    if int((signal != 0).sum()) == 0:
        failure_reasons.append("no_nonzero_signal")
    if int((signal.diff().fillna(0).abs() > 0).sum()) == 0:
        failure_reasons.append("no_signal_switches")

    ledger = run_strict4h_signal(
        ohlcv,
        funding,
        signal,
        {
            "strategy": "formula_candidate",
            "strategy_id": candidate_id,
            "max_exposure_abs": max_exposure_abs,
            "min_hold_bars": 1,
        },
        costs,
        execution_delay_bars=1,
    )
    metrics = summarize(ledger, risk_free_rate)
    return {
        **base,
        "status": "completed",
        "failure_reasons": failure_reasons,
        "bars": metrics["bars"],
        "nonzero_signal_bars": int((signal != 0).sum()),
        "factor_valid_bars": int(valid.size),
        "factor_mean": float(valid.mean()),
        "factor_std": float(valid.std(ddof=0)),
        "metrics": metrics,
        **metrics,
        "final_equity": float(equity_curve(ledger["r_net"]).iloc[-1]) if len(ledger) else 1.0,
        "ledger": ledger,
    }


def evaluate_formula(formula: str, ohlcv: pd.DataFrame, funding: pd.Series) -> pd.Series:
    tree = ast.parse(formula, mode="eval")
    return _eval_node(tree.body, ohlcv, funding).astype(float)


def evaluate_candidate_factor(candidate: dict[str, Any], ohlcv: pd.DataFrame, funding: pd.Series) -> pd.Series:
    components = _component_formulas(candidate)
    if not components:
        return evaluate_formula(str(candidate.get("formula", "")), ohlcv, funding)
    method = str(candidate.get("combination_method", "mean")).lower()
    if method not in {"mean", "equal_weight_mean"}:
        raise ValueError(f"unsupported combination_method {method}")
    frame = pd.DataFrame(
        {
            f"component_{idx}": evaluate_formula(formula, ohlcv, funding)
            for idx, formula in enumerate(components)
        }
    )
    return frame.mean(axis=1, skipna=False)


def load_factor_candidate_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError(f"candidate JSONL line {line_number} is not an object")
        records.append(payload)
    return records


def render_factor_candidate_judging_report(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# RandysLab Factor Candidate Judging",
        "",
        "This is a strict research backtest artifact for QuantumRandy factor candidates.",
        "It is not a runtime publish payload, portfolio construction step, or live execution plan.",
        "",
        "## Run",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Symbol: `{payload['symbol']}`",
        f"- Exchange: `{payload['exchange']}`",
        f"- Window: `{payload['window']['name']}`",
        f"- Matching: `{payload['execution']['matching']}`",
        f"- Signal rule: `{payload['execution']['signal_rule']}`",
        "",
        "## Candidate Summary",
        "",
        "| Candidate | Status | Sharpe | CAGR | Max DD | Trades | Failures | Formula |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        failures = "|".join(row.get("failure_reasons", [])) or "none"
        report_row = {
            **row,
            "failures": failures,
            "sharpe": float(row.get("sharpe", 0.0) or 0.0),
            "cagr": float(row.get("cagr", 0.0) or 0.0),
            "max_dd": float(row.get("max_dd", 0.0) or 0.0),
            "trades": float(row.get("trades", 0.0) or 0.0),
        }
        lines.append(
            "| {candidate_id} | {status} | {sharpe:.4f} | {cagr:.4f} | {max_dd:.4f} | "
            "{trades:.0f} | `{failures}` | `{formula}` |".format(**report_row)
        )
    lines.extend(
        [
            "",
            "## Safety Boundary",
            "",
            "- Research/backtest comparison only.",
            "- No exchange trading credentials.",
            "- No live order placement.",
            "- No automatic QuantumRandy runtime publishing.",
            "- Failure reasons must be preserved for downstream review.",
            "",
        ]
    )
    return "\n".join(lines)


def _eval_node(node: ast.AST, ohlcv: pd.DataFrame, funding: pd.Series) -> pd.Series | float:
    if isinstance(node, ast.Name):
        return _field(node.id, ohlcv, funding)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _eval_node(node.operand, ohlcv, funding)
        return -value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in SUPPORTED_FUNCTIONS:
            raise ValueError(f"unsupported function {name}")
        args = [_eval_node(arg, ohlcv, funding) for arg in node.args]
        return _call_function(name, args, ohlcv.index)
    raise ValueError(f"unsupported expression node {type(node).__name__}")


def _call_function(name: str, args: list[Any], index: pd.Index) -> pd.Series:
    if name == "add":
        _require_arg_count(name, args, 2)
        return _series(args[0], index) + _series(args[1], index)
    if name == "sub":
        _require_arg_count(name, args, 2)
        return _series(args[0], index) - _series(args[1], index)
    if name == "mul":
        _require_arg_count(name, args, 2)
        return _series(args[0], index) * _series(args[1], index)
    if name == "div":
        _require_arg_count(name, args, 2)
        denom = _series(args[1], index).replace(0, np.nan)
        return _series(args[0], index) / denom
    if name == "neg":
        _require_arg_count(name, args, 1)
        return -_series(args[0], index)
    if name == "abs":
        _require_arg_count(name, args, 1)
        return _series(args[0], index).abs()
    if name == "log":
        _require_arg_count(name, args, 1)
        return np.log(_series(args[0], index).abs().replace(0, np.nan))
    if name == "sqrt":
        _require_arg_count(name, args, 1)
        return np.sqrt(_series(args[0], index).clip(lower=0))
    if name == "sign":
        _require_arg_count(name, args, 1)
        series = _series(args[0], index)
        return pd.Series(np.where(series > 0, 1.0, np.where(series < 0, -1.0, 0.0)), index=index)
    if name == "sma":
        _require_arg_count(name, args, 2)
        return _series(args[0], index).rolling(_window(args[1]), min_periods=_window(args[1])).mean()
    if name == "ema":
        _require_arg_count(name, args, 2)
        return _series(args[0], index).ewm(span=_window(args[1]), adjust=False).mean()
    if name == "std":
        _require_arg_count(name, args, 2)
        window = _window(args[1])
        return _series(args[0], index).rolling(window, min_periods=window).std(ddof=0)
    if name == "min":
        _require_arg_count(name, args, 2)
        window = _window(args[1])
        return _series(args[0], index).rolling(window, min_periods=window).min()
    if name == "max":
        _require_arg_count(name, args, 2)
        window = _window(args[1])
        return _series(args[0], index).rolling(window, min_periods=window).max()
    if name == "delta":
        _require_arg_count(name, args, 2)
        return _series(args[0], index).diff(_window(args[1]))
    if name == "ret":
        _require_arg_count(name, args, 2)
        return _series(args[0], index).pct_change(_window(args[1]))
    if name == "zscore":
        _require_arg_count(name, args, 2)
        series = _series(args[0], index)
        window = _window(args[1])
        mean = series.rolling(window, min_periods=window).mean()
        std = series.rolling(window, min_periods=window).std(ddof=0)
        return (series - mean) / std.replace(0, np.nan)
    if name == "corr":
        _require_arg_count(name, args, 3)
        window = _window(args[2])
        return _series(args[0], index).rolling(window, min_periods=window).corr(_series(args[1], index))
    if name == "rsi":
        _require_arg_count(name, args, 2)
        delta = _series(args[0], index).diff()
        window = _window(args[1])
        gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
        loss = (-delta).clip(lower=0).rolling(window, min_periods=window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))
    raise ValueError(f"unsupported function {name}")


def _field(name: str, ohlcv: pd.DataFrame, funding: pd.Series) -> pd.Series:
    if name not in SUPPORTED_FIELDS:
        raise ValueError(f"unsupported field {name}")
    if name == "funding_rate":
        return funding.reindex(ohlcv.index).fillna(0.0)
    if name not in ohlcv.columns:
        raise ValueError(f"missing OHLCV field {name}")
    return ohlcv[name].astype(float)


def _series(value: Any, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index).astype(float)
    return pd.Series(float(value), index=index)


def _window(value: Any) -> int:
    if isinstance(value, pd.Series):
        raise ValueError("window argument must be numeric")
    window = int(value)
    if window <= 0:
        raise ValueError("window argument must be positive")
    return window


def _require_arg_count(name: str, args: list[Any], expected: int) -> None:
    if len(args) != expected:
        raise ValueError(f"{name} expects {expected} arguments")


def _resolve_path(base_dir: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _display_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _window_bounds(settings: Any, window: str) -> dict[str, str | None]:
    if window == "all":
        return {"start": None, "end": None}
    return {
        "start": getattr(settings.windows, f"{window}_start"),
        "end": getattr(settings.windows, f"{window}_end"),
    }


def _safe_id(raw: str) -> str:
    text = str(raw).strip().lower()
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text) or "candidate"


def _factor_to_signal(factor: pd.Series, *, signal_threshold: float, signal_mode: str) -> pd.Series:
    mode = str(signal_mode).lower()
    signal = pd.Series(0, index=factor.index, dtype="int8")
    if mode == "long_short":
        signal[factor > signal_threshold] = 1
        signal[factor < -signal_threshold] = -1
    elif mode == "long_flat":
        signal[factor > signal_threshold] = 1
    elif mode == "short_flat":
        signal[factor < -signal_threshold] = -1
    else:
        raise ValueError(f"unsupported signal_mode {signal_mode}")
    return signal.fillna(0).astype("int8")


def _volatility_cap_active(volatility_cap_formula: str | None, volatility_cap_max: float | None) -> bool:
    return bool(str(volatility_cap_formula or "").strip()) and volatility_cap_max is not None


def _apply_volatility_cap(
    signal: pd.Series,
    ohlcv: pd.DataFrame,
    funding: pd.Series,
    *,
    volatility_cap_formula: str | None,
    volatility_cap_max: float | None,
) -> pd.Series:
    if not _volatility_cap_active(volatility_cap_formula, volatility_cap_max):
        return signal
    cap_series = evaluate_formula(str(volatility_cap_formula), ohlcv, funding).reindex(signal.index)
    allowed = cap_series.replace([np.inf, -np.inf], np.nan) <= float(volatility_cap_max)
    return signal.where(allowed.fillna(False), 0).astype("int8")


def _drawdown_stop_active(drawdown_stop_max: float | None) -> bool:
    return drawdown_stop_max is not None and float(drawdown_stop_max) > 0.0


def _drawdown_stop_label(drawdown_stop_max: float | None, drawdown_stop_cooldown_bars: int) -> str:
    if not _drawdown_stop_active(drawdown_stop_max):
        return "none"
    return f"dd_stop_{_numeric_id(float(drawdown_stop_max))}_cd_{int(drawdown_stop_cooldown_bars or 0)}"


def _apply_drawdown_stop(
    signal: pd.Series,
    ohlcv: pd.DataFrame,
    funding: pd.Series,
    *,
    costs: Any,
    max_exposure_abs: float,
    drawdown_stop_max: float | None,
    drawdown_stop_cooldown_bars: int,
) -> pd.Series:
    if not _drawdown_stop_active(drawdown_stop_max):
        return signal

    adjusted = signal.reindex(ohlcv.index).fillna(0).astype("int8").copy()
    close = ohlcv["close"].astype(float)
    funding_rate = funding.reindex(ohlcv.index).fillna(0.0).astype(float)
    cap = abs(float(max_exposure_abs or 1.0))
    fee_bps = float(getattr(costs, "taker_bps", 0.0)) * float(getattr(costs, "fee_multiplier", 1.0))
    slip_bps = float(getattr(costs, "slippage_bps", 0.0)) * float(getattr(costs, "slippage_multiplier", 1.0))
    funding_multiplier = float(getattr(costs, "funding_multiplier", 1.0))
    stop_max = float(drawdown_stop_max)
    cooldown_bars = max(int(drawdown_stop_cooldown_bars or 0), 0)

    equity = 1.0
    peak = 1.0
    previous_exposure = 0.0
    cooldown_remaining = 0

    for idx, timestamp in enumerate(ohlcv.index):
        if cooldown_remaining > 0:
            adjusted.iat[idx] = 0
            cooldown_remaining -= 1

        exposure = float(adjusted.iat[idx - 1]) if idx > 0 else 0.0
        exposure = min(max(exposure, -cap), cap)
        if idx == 0:
            market_return = 0.0
        else:
            previous_close = float(close.iat[idx - 1])
            market_return = 0.0 if previous_close == 0.0 else float(close.iat[idx] / previous_close - 1.0)

        turnover = abs(exposure - previous_exposure)
        fee = turnover * fee_bps / 10_000.0
        slippage = turnover * slip_bps / 10_000.0
        funding_cost = exposure * float(funding_rate.loc[timestamp]) * 0.5 * funding_multiplier
        net_return = exposure * market_return - fee - slippage - funding_cost
        equity *= 1.0 + net_return
        peak = max(peak, equity)
        drawdown = 0.0 if peak <= 0 else 1.0 - equity / peak

        if drawdown >= stop_max:
            adjusted.iat[idx] = 0
            cooldown_remaining = cooldown_bars
            peak = max(equity, 1e-12)

        previous_exposure = exposure

    return adjusted.astype("int8")


def _signal_rule(signal_mode: str) -> str:
    mode = str(signal_mode).lower()
    if mode == "long_short":
        return "long if factor > threshold, short if factor < -threshold, otherwise flat"
    if mode == "long_flat":
        return "long if factor > threshold, otherwise flat"
    if mode == "short_flat":
        return "short if factor < -threshold, otherwise flat"
    raise ValueError(f"unsupported signal_mode {signal_mode}")


def _component_formulas(candidate: dict[str, Any]) -> list[str]:
    raw = candidate.get("component_formulas", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("component_formulas must be a list")
    return [str(value) for value in raw if str(value).strip()]


def _numeric_id(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")
