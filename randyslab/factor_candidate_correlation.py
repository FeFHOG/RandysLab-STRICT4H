from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_settings
from .data import align_funding_to_ohlcv, read_funding, read_ohlcv
from .formula_candidates import evaluate_candidate_factor, evaluate_formula, load_factor_candidate_jsonl


def run_factor_candidate_correlation_review(
    *,
    config_path: str | Path,
    candidate_jsonl: str | Path,
    out_dir: str | Path,
    ohlcv_csv: str | Path | None = None,
    funding_csv: str | Path | None = None,
    symbol: str | None = None,
    high_corr_threshold: float = 0.80,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    settings = load_settings(config_path)
    candidate_path = _resolve_path(base_dir, candidate_jsonl)
    ohlcv_path = _resolve_path(base_dir, ohlcv_csv or settings.paths.ohlcv_csv)
    funding_path = _resolve_path(base_dir, funding_csv or settings.paths.funding_csv)
    candidates = load_factor_candidate_jsonl(candidate_path)
    ohlcv = read_ohlcv(ohlcv_path)
    funding = align_funding_to_ohlcv(ohlcv, read_funding(funding_path))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    factor_values = _candidate_factor_values(candidates, ohlcv, funding)
    pairwise = _pairwise_correlations(candidates, factor_values, high_corr_threshold)
    bundles = _bundle_redundancy(candidates, ohlcv, funding, high_corr_threshold)

    pairwise_csv = out / "factor_candidate_pairwise_correlation.csv"
    bundle_csv = out / "factor_candidate_bundle_redundancy.csv"
    pairwise.to_csv(pairwise_csv, index=False)
    bundles.to_csv(bundle_csv, index=False)

    payload = {
        "artifact_type": "randyslab_factor_candidate_correlation_review",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "role": "factor_bundle_correlation_redundancy_review",
        "safety": {
            "research_only": True,
            "paper_only": True,
            "no_live_orders": True,
            "no_exchange_trading_keys": True,
            "not_runtime_publish_payload": True,
            "does_not_update_quantumrandy_runtime": True,
            "does_not_admit_factors": True,
        },
        "symbol": symbol or settings.symbol,
        "candidate_jsonl": _display_path(candidate_path, base_dir),
        "candidate_count": len(candidates),
        "bundle_count": int(sum(1 for candidate in candidates if candidate.get("component_formulas"))),
        "high_corr_threshold": float(high_corr_threshold),
        "pairwise_row_count": int(len(pairwise)),
        "high_corr_pair_count": int(pairwise["abs_corr_ge_threshold"].sum()) if not pairwise.empty else 0,
        "bundle_verdict_counts": bundles["redundancy_verdict"].value_counts().to_dict() if not bundles.empty else {},
        "artifacts": {
            "pairwise_csv": pairwise_csv.name,
            "bundle_csv": bundle_csv.name,
            "report_md": "factor_candidate_correlation_report.md",
        },
    }
    (out / "factor_candidate_correlation_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (out / "factor_candidate_correlation_report.md").write_text(
        render_factor_candidate_correlation_report(payload, pairwise, bundles),
        encoding="utf-8",
    )
    return payload


def render_factor_candidate_correlation_report(
    payload: dict[str, Any],
    pairwise: pd.DataFrame,
    bundles: pd.DataFrame,
) -> str:
    lines = [
        "# RandysLab Factor Candidate Correlation Review",
        "",
        "This is a research-only correlation and redundancy artifact. It is not factor admission, portfolio construction,",
        "runtime publishing, or live execution approval.",
        "",
        "## Run",
        "",
        f"- Symbol: `{payload['symbol']}`",
        f"- Candidate count: `{payload['candidate_count']}`",
        f"- Bundle count: `{payload['bundle_count']}`",
        f"- High-correlation threshold: `{payload['high_corr_threshold']}`",
        f"- High-correlation pairs: `{payload['high_corr_pair_count']}`",
        "",
        "## Bundle Redundancy",
        "",
        "| Bundle | Components | Max Abs Corr | Verdict |",
        "|---|---:|---:|---|",
    ]
    if bundles.empty:
        lines.append("| none | 0 | 0.0000 | none |")
    else:
        for row in bundles.to_dict(orient="records"):
            lines.append(
                "| `{bundle_candidate_id}` | {component_count} | {max_abs_component_corr:.4f} | "
                "`{redundancy_verdict}` |".format(
                    bundle_candidate_id=row["bundle_candidate_id"],
                    component_count=int(row["component_count"]),
                    max_abs_component_corr=float(row["max_abs_component_corr"]),
                    redundancy_verdict=row["redundancy_verdict"],
                )
            )
    lines.extend(
        [
            "",
            "## Pairwise Sample",
            "",
            "| Left | Right | Corr | Abs Corr | Overlap | High Corr |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in pairwise.head(20).to_dict(orient="records"):
        lines.append(
            "| `{left_candidate_id}` | `{right_candidate_id}` | {corr:.4f} | {abs_corr:.4f} | "
            "{overlap_count:.0f} | `{flag}` |".format(
                left_candidate_id=row["left_candidate_id"],
                right_candidate_id=row["right_candidate_id"],
                corr=float(row["corr"]),
                abs_corr=float(row["abs_corr"]),
                overlap_count=float(row["overlap_count"]),
                flag=bool(row["abs_corr_ge_threshold"]),
            )
        )
    return "\n".join(lines) + "\n"


def _candidate_factor_values(
    candidates: list[dict[str, Any]],
    ohlcv: pd.DataFrame,
    funding: pd.Series,
) -> dict[str, pd.Series]:
    values: dict[str, pd.Series] = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id", ""))
        if not candidate_id:
            continue
        try:
            values[candidate_id] = evaluate_candidate_factor(candidate, ohlcv, funding).replace(
                [np.inf, -np.inf],
                np.nan,
            )
        except (ValueError, TypeError, SyntaxError):
            values[candidate_id] = pd.Series(dtype=float)
    return values


def _pairwise_correlations(
    candidates: list[dict[str, Any]],
    values: dict[str, pd.Series],
    high_corr_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_id = {str(candidate.get("candidate_id", "")): candidate for candidate in candidates}
    for left_id, right_id in combinations(values.keys(), 2):
        left = values[left_id]
        right = values[right_id]
        aligned = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
        corr = float(aligned["left"].corr(aligned["right"])) if len(aligned) >= 3 else 0.0
        if pd.isna(corr):
            corr = 0.0
        left_family = str(by_id.get(left_id, {}).get("formula_family", ""))
        right_family = str(by_id.get(right_id, {}).get("formula_family", ""))
        rows.append(
            {
                "left_candidate_id": left_id,
                "right_candidate_id": right_id,
                "left_family": left_family,
                "right_family": right_family,
                "same_family": left_family == right_family,
                "corr": round(corr, 8),
                "abs_corr": round(abs(corr), 8),
                "overlap_count": int(len(aligned)),
                "abs_corr_ge_threshold": abs(corr) >= float(high_corr_threshold),
            }
        )
    return pd.DataFrame(rows)


def _bundle_redundancy(
    candidates: list[dict[str, Any]],
    ohlcv: pd.DataFrame,
    funding: pd.Series,
    high_corr_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        component_formulas = [str(value) for value in candidate.get("component_formulas") or []]
        if not component_formulas:
            continue
        component_values = {
            formula: evaluate_formula(formula, ohlcv, funding).replace([np.inf, -np.inf], np.nan)
            for formula in component_formulas
        }
        correlations = []
        overlaps = []
        for left_formula, right_formula in combinations(component_formulas, 2):
            aligned = pd.concat(
                [
                    component_values[left_formula].rename("left"),
                    component_values[right_formula].rename("right"),
                ],
                axis=1,
            ).dropna()
            corr = float(aligned["left"].corr(aligned["right"])) if len(aligned) >= 3 else 0.0
            if pd.isna(corr):
                corr = 0.0
            correlations.append(abs(corr))
            overlaps.append(len(aligned))
        max_abs_corr = max(correlations) if correlations else 0.0
        rows.append(
            {
                "bundle_candidate_id": str(candidate.get("candidate_id", "")),
                "formula": str(candidate.get("formula", "")),
                "component_candidate_ids": "|".join(str(value) for value in candidate.get("component_candidate_ids") or []),
                "component_count": len(component_formulas),
                "max_abs_component_corr": round(float(max_abs_corr), 8),
                "mean_abs_component_corr": round(float(sum(correlations) / len(correlations)), 8)
                if correlations
                else 0.0,
                "min_component_overlap": int(min(overlaps)) if overlaps else 0,
                "high_corr_pair_count": int(sum(value >= float(high_corr_threshold) for value in correlations)),
                "redundancy_verdict": "redundant_research_memory_only"
                if max_abs_corr >= float(high_corr_threshold)
                else "diversified_enough_for_research",
            }
        )
    return pd.DataFrame(rows)


def _resolve_path(base_dir: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _display_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
