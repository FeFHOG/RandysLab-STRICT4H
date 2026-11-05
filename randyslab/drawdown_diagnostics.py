from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_settings
from .data import load_market, slice_window
from .formula_candidates import (
    judge_factor_candidate,
    load_factor_candidate_jsonl,
)
from .metrics import equity_curve


def build_drawdown_episode_summary(
    ledger: pd.DataFrame,
    *,
    metadata: dict[str, Any] | None = None,
    min_drawdown: float = 0.05,
    top_n: int | None = 3,
) -> pd.DataFrame:
    metadata = metadata or {}
    if ledger.empty:
        return pd.DataFrame()

    equity = equity_curve(ledger["r_net"])
    peak = equity.cummax()
    drawdown = 1.0 - equity / peak.replace(0.0, pd.NA)
    drawdown = drawdown.fillna(0.0).clip(lower=0.0)
    episodes: list[dict[str, Any]] = []
    in_episode = False
    peak_pos = 0
    trough_pos = 0
    max_drawdown = 0.0

    for pos, _timestamp in enumerate(drawdown.index):
        value = float(drawdown.iloc[pos])
        if value <= 0.0:
            if in_episode:
                episodes.append(
                    _episode_row(
                        ledger,
                        metadata=metadata,
                        equity=equity,
                        peak_pos=peak_pos,
                        trough_pos=trough_pos,
                        recovery_pos=pos,
                        max_drawdown=max_drawdown,
                        recovered=True,
                    )
                )
                in_episode = False
                max_drawdown = 0.0
            peak_pos = pos
            continue

        if not in_episode:
            in_episode = True
            trough_pos = pos
            max_drawdown = value

        if value > max_drawdown:
            max_drawdown = value
            trough_pos = pos

    if in_episode:
        episodes.append(
            _episode_row(
                ledger,
                metadata=metadata,
                equity=equity,
                peak_pos=peak_pos,
                trough_pos=trough_pos,
                recovery_pos=len(drawdown.index) - 1,
                max_drawdown=max_drawdown,
                recovered=False,
            )
        )

    rows = [row for row in episodes if float(row["max_drawdown"]) >= float(min_drawdown)]
    rows.sort(key=lambda row: float(row["max_drawdown"]), reverse=True)
    if top_n is not None:
        rows = rows[: int(top_n)]
    return pd.DataFrame(rows)


def label_drawdown_failure(row: dict[str, Any] | pd.Series) -> list[str]:
    asset = str(row.get("asset", "")).upper()
    window = str(row.get("window", "")).lower()
    max_drawdown = float(row.get("max_drawdown", row.get("max_dd", 0.0)) or 0.0)
    market_return = float(row.get("market_return_peak_to_trough", 0.0) or 0.0)

    labels: list[str] = []
    if asset.startswith("BTC"):
        labels.append("btc_weakness")
    if asset.startswith(("SOL", "AVAX")):
        labels.append("sol_avax_concentration")
    if window == "validation":
        labels.append("validation_weakness")
    if market_return <= -0.25 or max_drawdown >= 0.60:
        labels.append("crash_period_drawdown")
    if max_drawdown >= 0.80:
        labels.append("extreme_row_drawdown")
    return labels


def run_factor_drawdown_diagnostics(
    *,
    config_path: str | Path,
    candidate_jsonl: str | Path,
    assets: list[dict[str, str]],
    out_dir: str | Path,
    windows: list[str],
    thresholds: list[float],
    signal_modes: list[str],
    exposure_caps: list[float],
    volatility_caps: list[dict[str, Any]],
    drawdown_stops: list[dict[str, Any]],
    min_episode_drawdown: float = 0.05,
    top_episodes_per_row: int = 3,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    candidate_jsonl = _resolve_path(base_dir, candidate_jsonl)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    settings = load_settings(config_path)
    candidates = load_factor_candidate_jsonl(candidate_jsonl)
    row_summaries: list[dict[str, Any]] = []
    episode_frames: list[pd.DataFrame] = []

    for asset in assets:
        asset_label = _asset_label(asset)
        ohlcv_path = _resolve_path(base_dir, asset.get("ohlcv_csv") or settings.paths.ohlcv_csv)
        funding_path = _resolve_path(base_dir, asset.get("funding_csv") or settings.paths.funding_csv)
        full_ohlcv, full_funding = load_market(ohlcv_path, funding_path, settings.bar_hours)

        for window in windows:
            bounds = _window_bounds(settings, window)
            ohlcv = slice_window(full_ohlcv, bounds["start"], bounds["end"]) if window != "all" else full_ohlcv
            funding = full_funding.reindex(ohlcv.index)
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
                                )
                                for candidate in candidates:
                                    result = judge_factor_candidate(
                                        candidate,
                                        ohlcv=ohlcv,
                                        funding=funding,
                                        costs=settings.costs,
                                        risk_free_rate=settings.risk_free_rate,
                                        signal_threshold=threshold,
                                        signal_mode=signal_mode,
                                        max_exposure_abs=exposure_cap,
                                        volatility_cap_formula=volatility_cap["formula"] or None,
                                        volatility_cap_max=volatility_cap["max_value"],
                                        drawdown_stop_label=str(drawdown_stop["label"]),
                                        drawdown_stop_max=drawdown_stop["max_drawdown"],
                                        drawdown_stop_cooldown_bars=int(drawdown_stop["cooldown_bars"]),
                                    )
                                    row_metadata = {
                                        "asset": asset_label,
                                        "window": window,
                                        "threshold": threshold,
                                        "signal_mode": signal_mode,
                                        "exposure_cap": exposure_cap,
                                        "volatility_cap_label": volatility_cap["label"],
                                        "drawdown_stop_label": drawdown_stop["label"],
                                        "variant_id": variant_id,
                                        "candidate_id": result["candidate_id"],
                                        "formula": result["formula"],
                                    }
                                    row_summary = {
                                        **row_metadata,
                                        "status": result["status"],
                                        "failure_reasons": "|".join(result.get("failure_reasons", [])),
                                    }
                                    if result["status"] == "completed":
                                        ledger = result["ledger"]
                                        episodes = build_drawdown_episode_summary(
                                            ledger,
                                            metadata=row_metadata,
                                            min_drawdown=min_episode_drawdown,
                                            top_n=top_episodes_per_row,
                                        )
                                        if not episodes.empty:
                                            episode_frames.append(episodes)
                                            worst = episodes.sort_values("max_drawdown", ascending=False).iloc[0]
                                            row_summary.update(
                                                {
                                                    "max_episode_drawdown": float(worst["max_drawdown"]),
                                                    "worst_peak_timestamp": worst["peak_timestamp"],
                                                    "worst_trough_timestamp": worst["trough_timestamp"],
                                                    "worst_recovery_timestamp": worst["recovery_timestamp"],
                                                    "episode_failure_labels": worst["failure_labels"],
                                                }
                                            )
                                        else:
                                            row_summary.update(
                                                {
                                                    "max_episode_drawdown": 0.0,
                                                    "worst_peak_timestamp": "",
                                                    "worst_trough_timestamp": "",
                                                    "worst_recovery_timestamp": "",
                                                    "episode_failure_labels": "",
                                                }
                                            )
                                        row_summary.update(result["metrics"])
                                    row_summaries.append(row_summary)

    row_summary = pd.DataFrame(row_summaries)
    episodes = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()
    row_csv = out / "drawdown_row_summary.csv"
    episode_csv = out / "drawdown_episode_summary.csv"
    row_summary.to_csv(row_csv, index=False)
    episodes.to_csv(episode_csv, index=False)
    payload = {
        "artifact_type": "randyslab_factor_drawdown_diagnostics",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "role": "strict_factor_drawdown_episode_diagnostics",
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
        "row_count": int(len(row_summary)),
        "episode_count": int(len(episodes)),
        "artifacts": {
            "row_summary_csv": row_csv.name,
            "episode_summary_csv": episode_csv.name,
            "report_md": "drawdown_episode_report.md",
        },
    }
    (out / "drawdown_diagnostics_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (out / "drawdown_episode_report.md").write_text(
        render_drawdown_diagnostics_report(payload, row_summary, episodes),
        encoding="utf-8",
    )
    return payload


def render_drawdown_diagnostics_report(
    payload: dict[str, Any],
    row_summary: pd.DataFrame,
    episodes: pd.DataFrame,
) -> str:
    lines = [
        "# RandysLab Drawdown Episode Diagnostics",
        "",
        "This is a research-only episode audit for strict factor-candidate sweeps.",
        "It is not a runtime publish payload, factor admission decision, portfolio construction step, or live "
        "execution plan.",
        "",
        "## Run",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Rows audited: `{payload['row_count']}`",
        f"- Episodes captured: `{payload['episode_count']}`",
        f"- Thresholds: `{','.join(str(value) for value in payload['thresholds'])}`",
        f"- Signal modes: `{','.join(payload['signal_modes'])}`",
        "",
        "## Worst Episodes",
        "",
        "| Asset | Window | Variant | Max DD | Peak | Trough | Recovery | Labels |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    if episodes.empty:
        lines.append("| none | none | none | 0.0000 | none | none | none | none |")
    else:
        worst = episodes.sort_values("max_drawdown", ascending=False).head(20)
        for row in worst.to_dict(orient="records"):
            labels = row.get("failure_labels", "") or "none"
            lines.append(
                "| {asset} | {window} | `{variant_id}` | {max_drawdown:.4f} | {peak_timestamp} | "
                "{trough_timestamp} | {recovery_timestamp} | `{labels}` |".format(
                    labels=labels,
                    **row,
                )
            )

    lines.extend(
        [
            "",
            "## Row-Level Worst Drawdowns",
            "",
            "| Asset | Window | Variant | Sharpe | Max DD | Episode DD | Labels |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    if row_summary.empty:
        lines.append("| none | none | none | 0.0000 | 0.0000 | 0.0000 | none |")
    else:
        row_worst = row_summary.sort_values("max_episode_drawdown", ascending=False).head(20)
        for row in row_worst.to_dict(orient="records"):
            labels = row.get("episode_failure_labels", "") or "none"
            report_row = {
                **row,
                "labels": labels,
                "sharpe": float(row.get("sharpe", 0.0) or 0.0),
                "max_dd": float(row.get("max_dd", 0.0) or 0.0),
                "max_episode_drawdown": float(row.get("max_episode_drawdown", 0.0) or 0.0),
            }
            lines.append(
                "| {asset} | {window} | `{variant_id}` | {sharpe:.4f} | {max_dd:.4f} | "
                "{max_episode_drawdown:.4f} | `{labels}` |".format(
                    **report_row,
                )
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Research/backtest diagnostics only.",
            "- No exchange trading credentials.",
            "- No live order placement.",
            "- No automatic QuantumRandy runtime publishing.",
            "- Episode labels are failure-memory hints, not admission decisions.",
            "",
        ]
    )
    return "\n".join(lines)


def _episode_row(
    ledger: pd.DataFrame,
    *,
    metadata: dict[str, Any],
    equity: pd.Series,
    peak_pos: int,
    trough_pos: int,
    recovery_pos: int,
    max_drawdown: float,
    recovered: bool,
) -> dict[str, Any]:
    segment = ledger.iloc[peak_pos : trough_pos + 1]
    row = {
        **metadata,
        "peak_timestamp": _iso(ledger.index[peak_pos]),
        "trough_timestamp": _iso(ledger.index[trough_pos]),
        "recovery_timestamp": _iso(ledger.index[recovery_pos]) if recovered else "",
        "recovered": bool(recovered),
        "max_drawdown": round(float(max_drawdown), 8),
        "duration_bars_peak_to_trough": int(trough_pos - peak_pos),
        "duration_bars_peak_to_recovery": int(recovery_pos - peak_pos) if recovered else "",
        "equity_at_peak": round(float(equity.iloc[peak_pos]), 8),
        "equity_at_trough": round(float(equity.iloc[trough_pos]), 8),
        "net_return_peak_to_trough": round(float(segment["r_net"].sum()), 8),
        "market_return_peak_to_trough": round(float(segment["r_mkt"].sum()), 8),
        "mean_exposure_peak_to_trough": round(float(segment["exposure"].mean()), 8),
        "active_exposure_share_peak_to_trough": round(float((segment["exposure"].abs() > 0).mean()), 8),
        "fee_total_peak_to_trough": round(float(segment.get("c_fee", pd.Series(dtype=float)).sum()), 8),
        "slippage_total_peak_to_trough": round(float(segment.get("c_slip", pd.Series(dtype=float)).sum()), 8),
        "funding_total_peak_to_trough": round(float(segment.get("c_fund", pd.Series(dtype=float)).sum()), 8),
    }
    row["failure_labels"] = "|".join(label_drawdown_failure(row))
    return row


def _resolve_path(base_dir: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _window_bounds(settings: Any, window: str) -> dict[str, str | None]:
    if window == "all":
        return {"start": None, "end": None}
    return {
        "start": getattr(settings.windows, f"{window}_start"),
        "end": getattr(settings.windows, f"{window}_end"),
    }


def _asset_label(asset: dict[str, str]) -> str:
    label = str(asset.get("symbol") or asset.get("label") or "").strip()
    if label:
        return label
    return Path(asset.get("ohlcv_csv", "asset")).stem.replace("_4h", "")


def _variant_id(
    *,
    threshold: float,
    signal_mode: str,
    exposure_cap: float,
    volatility_cap_label: str,
    drawdown_stop_label: str,
) -> str:
    return "_".join(
        [
            f"thr_{_numeric_id(threshold)}",
            signal_mode,
            f"cap_{_numeric_id(exposure_cap)}",
            _safe_id(volatility_cap_label),
            _safe_id(drawdown_stop_label),
        ]
    )


def _safe_id(raw: str) -> str:
    text = str(raw).strip().lower()
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text) or "asset"


def _numeric_id(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _iso(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.isoformat()
