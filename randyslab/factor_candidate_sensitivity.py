from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .formula_candidates import run_factor_candidate_judging

DEFAULT_WINDOWS = ["training", "validation", "blind"]
DEFAULT_THRESHOLDS = [0.0, 0.5, 1.0]
DEFAULT_SIGNAL_MODES = ["long_short"]
DEFAULT_EXPOSURE_CAPS = [1.0]
DEFAULT_VOLATILITY_CAPS = [{"label": "none", "formula": "", "max_value": None}]
DEFAULT_DRAWDOWN_STOPS = [{"label": "none", "max_drawdown": None, "cooldown_bars": 0}]


def run_factor_candidate_sensitivity(
    *,
    config_path: str | Path,
    candidate_jsonl: str | Path,
    assets: list[dict[str, str]],
    out_dir: str | Path,
    windows: list[str] | None = None,
    thresholds: list[float] | None = None,
    signal_modes: list[str] | None = None,
    exposure_caps: list[float] | None = None,
    volatility_caps: list[dict[str, Any]] | None = None,
    drawdown_stops: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not assets:
        raise ValueError("At least one asset spec is required")
    include_variant_metadata = exposure_caps is not None or volatility_caps is not None or drawdown_stops is not None
    include_drawdown_stop_metadata = drawdown_stops is not None
    windows = windows or DEFAULT_WINDOWS
    thresholds = thresholds or DEFAULT_THRESHOLDS
    signal_modes = signal_modes or DEFAULT_SIGNAL_MODES
    exposure_caps = exposure_caps or DEFAULT_EXPOSURE_CAPS
    volatility_caps = _normalize_volatility_caps(volatility_caps)
    drawdown_stops = _normalize_drawdown_stops(drawdown_stops)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    run_exports: list[dict[str, Any]] = []
    for asset in assets:
        label = _asset_label(asset)
        for window in windows:
            for threshold in thresholds:
                for signal_mode in signal_modes:
                    for exposure_cap in exposure_caps:
                        for volatility_cap in volatility_caps:
                            for drawdown_stop in drawdown_stops:
                                variant_id = _variant_id(
                                    threshold=threshold,
                                    signal_mode=signal_mode,
                                    exposure_cap=exposure_cap,
                                    volatility_cap_label=str(volatility_cap["label"]),
                                    drawdown_stop_label=str(drawdown_stop["label"]),
                                    include_drawdown_stop=include_drawdown_stop_metadata,
                                )
                                run_dir = out / "runs" / f"{_safe_id(label)}_{window}_{variant_id}"
                                payload = run_factor_candidate_judging(
                                    config_path=config_path,
                                    candidate_jsonl=candidate_jsonl,
                                    out_dir=run_dir,
                                    window=window,
                                    signal_threshold=threshold,
                                    signal_mode=signal_mode,
                                    max_exposure_abs=exposure_cap,
                                    volatility_cap_formula=volatility_cap["formula"] or None,
                                    volatility_cap_max=volatility_cap["max_value"],
                                    drawdown_stop_label=str(drawdown_stop["label"]),
                                    drawdown_stop_max=drawdown_stop["max_drawdown"],
                                    drawdown_stop_cooldown_bars=int(drawdown_stop["cooldown_bars"]),
                                    write_ledgers=False,
                                    ohlcv_csv=asset.get("ohlcv_csv"),
                                    funding_csv=asset.get("funding_csv"),
                                    symbol=label,
                                )
                                run_exports.append(
                                    {
                                        "asset": label,
                                        "window": window,
                                        "threshold": threshold,
                                        "signal_mode": signal_mode,
                                        "exposure_cap": exposure_cap,
                                        "volatility_cap": volatility_cap,
                                        "drawdown_stop": drawdown_stop,
                                        "variant_id": variant_id,
                                        "out_dir": run_dir.as_posix(),
                                        "completed_count": payload["completed_count"],
                                        "failed_count": payload["failed_count"],
                                    }
                                )
                                rows.extend(
                                    _rows_from_payload(
                                        payload,
                                        asset=label,
                                        window=window,
                                        threshold=threshold,
                                        signal_mode=signal_mode,
                                        exposure_cap=exposure_cap,
                                        volatility_cap=volatility_cap,
                                        drawdown_stop=drawdown_stop,
                                        variant_id=variant_id,
                                        include_variant_metadata=include_variant_metadata,
                                        include_drawdown_stop_metadata=include_drawdown_stop_metadata,
                                    )
                                )

    detail = pd.DataFrame(rows)
    stability = _stability_summary(detail)
    detail_csv = out / "factor_candidate_sensitivity_detail.csv"
    stability_csv = out / "factor_candidate_sensitivity_summary.csv"
    detail.to_csv(detail_csv, index=False)
    stability.to_csv(stability_csv, index=False)

    payload = {
        "artifact_type": "randyslab_factor_candidate_sensitivity",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "role": "strict_factor_candidate_sensitivity",
        "safety": {
            "research_only": True,
            "paper_only": True,
            "no_live_orders": True,
            "no_exchange_trading_keys": True,
            "not_runtime_publish_payload": True,
            "does_not_update_quantumrandy_runtime": True,
        },
        "config_path": str(config_path),
        "candidate_jsonl": str(candidate_jsonl),
        "assets": assets,
        "windows": windows,
        "thresholds": thresholds,
        "signal_modes": signal_modes,
        "exposure_caps": exposure_caps,
        "volatility_caps": volatility_caps,
        "drawdown_stops": drawdown_stops,
        "variant_count": (
            len(thresholds)
            * len(signal_modes)
            * len(exposure_caps)
            * len(volatility_caps)
            * len(drawdown_stops)
            if include_variant_metadata
            else 1
        ),
        "run_count": len(run_exports),
        "candidate_row_count": len(rows),
        "runs": run_exports,
        "artifacts": {
            "detail_csv": detail_csv.name,
            "summary_csv": stability_csv.name,
            "report_md": "factor_candidate_sensitivity_report.md",
        },
    }
    (out / "factor_candidate_sensitivity_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (out / "factor_candidate_sensitivity_report.md").write_text(
        render_factor_candidate_sensitivity_report(payload, stability),
        encoding="utf-8",
    )
    return payload


def render_factor_candidate_sensitivity_report(payload: dict[str, Any], stability: pd.DataFrame) -> str:
    lines = [
        "# RandysLab Factor Candidate Sensitivity",
        "",
        "This is a strict research sensitivity artifact for QuantumRandy factor candidates.",
        "It is not a runtime publish payload, portfolio construction step, or live execution plan.",
        "",
        "## Run",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Assets: `{len(payload['assets'])}`",
        f"- Windows: `{','.join(payload['windows'])}`",
        f"- Thresholds: `{','.join(str(value) for value in payload['thresholds'])}`",
        f"- Signal modes: `{','.join(payload.get('signal_modes', []))}`",
        f"- Exposure caps: `{','.join(str(value) for value in payload.get('exposure_caps', []))}`",
        "- Volatility caps: `"
        f"{','.join(str(item.get('label', 'none')) for item in payload.get('volatility_caps', []))}`",
        "- Drawdown stops: `"
        f"{','.join(str(item.get('label', 'none')) for item in payload.get('drawdown_stops', []))}`",
        f"- Judge runs: `{payload['run_count']}`",
        f"- Candidate rows: `{payload['candidate_row_count']}`",
        "",
        "## Stability Summary",
        "",
        "| Candidate | Variant | Formula | Mean Sharpe | Median Sharpe | Worst Sharpe | Positive Rows | "
        "Completed Rows | Mean Max DD | Best Row |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    if stability.empty:
        lines.append("| none | none | none | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | none |")
    else:
        for row in stability.to_dict(orient="records"):
            best_row = (
                f"{row['best_asset']} {row['best_window']} thr={row['best_threshold']} "
                f"sharpe={float(row['best_sharpe']):.4f}"
            )
            lines.append(
                "| {candidate_id} | `{variant_id}` | `{formula}` | {mean_sharpe:.4f} | {median_sharpe:.4f} | "
                "{worst_sharpe:.4f} | {positive_rows:.0f} | {completed_rows:.0f} | {mean_max_dd:.4f} | "
                "{best_row} |".format(
                    candidate_id=row["candidate_id"],
                    variant_id=row.get("variant_id", "default"),
                    formula=row["formula"],
                    mean_sharpe=float(row["mean_sharpe"]),
                    median_sharpe=float(row["median_sharpe"]),
                    worst_sharpe=float(row["worst_sharpe"]),
                    positive_rows=float(row["positive_rows"]),
                    completed_rows=float(row["completed_rows"]),
                    mean_max_dd=float(row["mean_max_dd"]),
                    best_row=best_row,
                )
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Research/backtest comparison only.",
            "- No exchange trading credentials.",
            "- No live order placement.",
            "- No automatic QuantumRandy runtime publishing.",
            "- Threshold sweeps are diagnostic evidence, not optimized strategy admission.",
            "",
        ]
    )
    return "\n".join(lines)


def _rows_from_payload(
    payload: dict[str, Any],
    *,
    asset: str,
    window: str,
    threshold: float,
    signal_mode: str,
    exposure_cap: float,
    volatility_cap: dict[str, Any],
    drawdown_stop: dict[str, Any],
    variant_id: str,
    include_variant_metadata: bool,
    include_drawdown_stop_metadata: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in payload.get("candidates", []):
        metrics = candidate.get("metrics", {}) or {}
        row = {
            "asset": asset,
            "window": window,
            "threshold": threshold,
            "signal_mode": signal_mode,
            "candidate_id": candidate.get("candidate_id", ""),
            "formula": candidate.get("formula", ""),
            "intended_scope": candidate.get("source_record", {}).get("intended_scope", ""),
            "applicability_hypothesis": candidate.get("source_record", {}).get("applicability_hypothesis", ""),
            "out_of_scope_policy": candidate.get("source_record", {}).get("out_of_scope_policy", ""),
            "status": candidate.get("status", ""),
            "failure_reasons": "|".join(candidate.get("failure_reasons", [])),
            "bars": int(metrics.get("bars", 0) or 0),
            "cagr": float(metrics.get("cagr", 0.0) or 0.0),
            "monthly_geom": float(metrics.get("monthly_geom", 0.0) or 0.0),
            "sharpe": float(metrics.get("sharpe", 0.0) or 0.0),
            "max_dd": float(metrics.get("max_dd", 0.0) or 0.0),
            "trades": int(metrics.get("trades", 0) or 0),
            "switch_density": float(metrics.get("switch_density", 0.0) or 0.0),
            "fee_total": float(metrics.get("fee_total", 0.0) or 0.0),
            "slippage_total": float(metrics.get("slippage_total", 0.0) or 0.0),
            "funding_total": float(metrics.get("funding_total", 0.0) or 0.0),
            "net_total": float(metrics.get("net_total", 0.0) or 0.0),
        }
        if include_variant_metadata:
            row.update(
                {
                    "exposure_cap": exposure_cap,
                    "volatility_cap_label": volatility_cap["label"],
                    "volatility_cap_formula": volatility_cap["formula"],
                    "volatility_cap_max": (
                        volatility_cap["max_value"] if volatility_cap["max_value"] is not None else ""
                    ),
                    "variant_id": variant_id,
                }
            )
            if include_drawdown_stop_metadata:
                row.update(
                    {
                        "drawdown_stop_label": drawdown_stop["label"],
                        "drawdown_stop_max": (
                            drawdown_stop["max_drawdown"] if drawdown_stop["max_drawdown"] is not None else ""
                        ),
                        "drawdown_stop_cooldown_bars": int(drawdown_stop["cooldown_bars"]),
                    }
                )
        rows.append(row)
    return rows


def _stability_summary(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    completed = detail[detail["status"] == "completed"].copy()
    if completed.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = ["candidate_id", "formula"] + [
        column
        for column in [
            "variant_id",
            "exposure_cap",
            "volatility_cap_label",
            "volatility_cap_formula",
            "volatility_cap_max",
            "drawdown_stop_label",
            "drawdown_stop_max",
            "drawdown_stop_cooldown_bars",
        ]
        if column in completed.columns
    ]
    for group_key, group in completed.groupby(group_columns, dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        key_map = dict(zip(group_columns, key_values, strict=False))
        best = group.sort_values(["sharpe", "cagr"], ascending=[False, False]).iloc[0]
        rows.append(
            {
                "candidate_id": key_map["candidate_id"],
                "formula": key_map["formula"],
                "variant_id": key_map.get("variant_id", "default"),
                "exposure_cap": key_map.get("exposure_cap", 1.0),
                "volatility_cap_label": key_map.get("volatility_cap_label", "none"),
                "volatility_cap_formula": key_map.get("volatility_cap_formula", ""),
                "volatility_cap_max": key_map.get("volatility_cap_max", ""),
                "drawdown_stop_label": key_map.get("drawdown_stop_label", "none"),
                "drawdown_stop_max": key_map.get("drawdown_stop_max", ""),
                "drawdown_stop_cooldown_bars": key_map.get("drawdown_stop_cooldown_bars", 0),
                "completed_rows": int(len(group)),
                "positive_rows": int((group["sharpe"] > 0).sum()),
                "mean_sharpe": round(float(group["sharpe"].mean()), 8),
                "median_sharpe": round(float(group["sharpe"].median()), 8),
                "worst_sharpe": round(float(group["sharpe"].min()), 8),
                "best_sharpe": round(float(group["sharpe"].max()), 8),
                "mean_cagr": round(float(group["cagr"].mean()), 8),
                "mean_max_dd": round(float(group["max_dd"].mean()), 8),
                "mean_trades": round(float(group["trades"].mean()), 8),
                "best_asset": best["asset"],
                "best_window": best["window"],
                "best_threshold": best["threshold"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_sharpe", "positive_rows", "worst_sharpe"],
        ascending=[False, False, False],
    )


def _asset_label(asset: dict[str, str]) -> str:
    label = str(asset.get("symbol") or asset.get("label") or "").strip()
    if label:
        return label
    ohlcv = Path(asset.get("ohlcv_csv", "asset"))
    return ohlcv.stem.replace("_4h", "")


def _normalize_volatility_caps(raw: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    values = raw or DEFAULT_VOLATILITY_CAPS
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(values):
        label = str(item.get("label") or "").strip() or f"vol_cap_{idx}"
        formula = str(item.get("formula") or "").strip()
        max_value = item.get("max_value")
        normalized.append(
            {
                "label": label,
                "formula": formula,
                "max_value": float(max_value) if max_value not in {None, ""} else None,
            }
        )
    return normalized


def _normalize_drawdown_stops(raw: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    values = raw or DEFAULT_DRAWDOWN_STOPS
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(values):
        label = str(item.get("label") or "").strip() or f"drawdown_stop_{idx}"
        max_drawdown = item.get("max_drawdown")
        normalized.append(
            {
                "label": label,
                "max_drawdown": float(max_drawdown) if max_drawdown not in {None, ""} else None,
                "cooldown_bars": int(item.get("cooldown_bars") or 0),
            }
        )
    return normalized


def _variant_id(
    *,
    threshold: float,
    signal_mode: str,
    exposure_cap: float,
    volatility_cap_label: str,
    drawdown_stop_label: str = "none",
    include_drawdown_stop: bool = False,
) -> str:
    parts = [
        f"thr_{_threshold_id(threshold)}",
        signal_mode,
        f"cap_{_threshold_id(exposure_cap)}",
        _safe_id(volatility_cap_label),
    ]
    if include_drawdown_stop:
        parts.append(_safe_id(drawdown_stop_label))
    return "_".join(parts)


def _threshold_id(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _safe_id(raw: str) -> str:
    text = str(raw).strip().lower()
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text) or "asset"
