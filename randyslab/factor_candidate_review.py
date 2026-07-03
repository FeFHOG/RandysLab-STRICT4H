from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ReviewRules:
    min_completed_rows: int = 15
    min_mean_sharpe: float = 0.25
    min_median_sharpe: float = 0.25
    min_positive_row_share: float = 0.55
    min_positive_assets: int = 3
    scope_mode: str = "multi_asset"
    min_validation_mean_sharpe: float = 0.0
    min_blind_mean_sharpe: float = 0.0
    max_positive_asset_share: float = 0.55
    max_mean_max_dd: float = 0.45
    max_worst_max_dd: float = 0.80


def review_factor_candidate_sensitivity(
    *,
    detail_csv: str | Path,
    out_dir: str | Path,
    rules: ReviewRules | None = None,
) -> dict[str, Any]:
    rules = rules or ReviewRules()
    detail_csv = Path(detail_csv)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(detail_csv).fillna("")
    review = build_factor_candidate_review(detail, rules)

    review_csv = out / "factor_candidate_review.csv"
    review.to_csv(review_csv, index=False)
    payload = {
        "artifact_type": "randyslab_factor_candidate_review",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "role": "conservative_research_review",
        "safety": {
            "research_only": True,
            "paper_only": True,
            "no_live_orders": True,
            "no_exchange_trading_keys": True,
            "not_runtime_publish_payload": True,
            "does_not_update_quantumrandy_runtime": True,
            "does_not_admit_factors": True,
        },
        "source": {
            "detail_csv": detail_csv.as_posix(),
        },
        "rules": asdict(rules),
        "candidate_count": int(len(review)),
        "verdict_counts": review["review_verdict"].value_counts().to_dict() if not review.empty else {},
        "artifacts": {
            "review_csv": review_csv.name,
            "report_md": "factor_candidate_review_report.md",
        },
    }
    (out / "factor_candidate_review_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (out / "factor_candidate_review_report.md").write_text(
        render_factor_candidate_review_report(payload, review),
        encoding="utf-8",
    )
    return payload


def build_factor_candidate_review(detail: pd.DataFrame, rules: ReviewRules | None = None) -> pd.DataFrame:
    rules = rules or ReviewRules()
    if detail.empty:
        return pd.DataFrame()
    required = {"candidate_id", "formula", "asset", "window", "status", "sharpe", "max_dd"}
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"sensitivity detail missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    variant_columns = [
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
            "intended_scope",
            "applicability_hypothesis",
            "out_of_scope_policy",
        ]
        if column in detail.columns
    ]
    group_columns = ["candidate_id", "formula"] + variant_columns
    for group_key, group in detail.groupby(group_columns, dropna=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        key_map = dict(zip(group_columns, key_values, strict=False))
        completed = group[group["status"] == "completed"].copy()
        failures = _failure_reasons(completed, group, rules)
        positive = completed[completed["sharpe"] > 0].copy()
        positive_assets = sorted(str(value) for value in positive["asset"].unique())
        window_means = _mean_by(completed, "window", "sharpe")
        threshold_means = _mean_by(completed, "threshold", "sharpe")
        asset_positive_counts = positive["asset"].value_counts().to_dict()
        scope_assets = sorted(str(value) for value in completed["asset"].unique())
        best = (
            completed.sort_values(["sharpe", "cagr"], ascending=[False, False]).iloc[0]
            if not completed.empty
            else {}
        )
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
                "intended_scope": key_map.get("intended_scope", ""),
                "applicability_hypothesis": key_map.get("applicability_hypothesis", ""),
                "out_of_scope_policy": key_map.get("out_of_scope_policy", ""),
                "scope_mode": rules.scope_mode,
                "scope_asset_count": int(len(scope_assets)),
                "review_verdict": "research_watchlist" if not failures else "blocked_by_conservative_rules",
                "failure_reasons": "|".join(failures),
                "completed_rows": int(len(completed)),
                "total_rows": int(len(group)),
                "positive_rows": int(len(positive)),
                "positive_row_share": _ratio(len(positive), len(completed)),
                "positive_assets": "|".join(positive_assets),
                "positive_asset_count": int(len(positive_assets)),
                "max_positive_asset_share": _max_positive_asset_share(asset_positive_counts),
                "mean_sharpe": _mean(completed, "sharpe"),
                "median_sharpe": _median(completed, "sharpe"),
                "worst_sharpe": _min(completed, "sharpe"),
                "best_sharpe": _max(completed, "sharpe"),
                "mean_max_dd": _mean(completed, "max_dd"),
                "worst_max_dd": _max(completed, "max_dd"),
                "training_mean_sharpe": window_means.get("training", 0.0),
                "validation_mean_sharpe": window_means.get("validation", 0.0),
                "blind_mean_sharpe": window_means.get("blind", 0.0),
                "threshold_mean_sharpe": _format_float_map(threshold_means),
                "best_asset": str(best.get("asset", "")) if isinstance(best, pd.Series) else "",
                "best_window": str(best.get("window", "")) if isinstance(best, pd.Series) else "",
                "best_threshold": best.get("threshold", "") if isinstance(best, pd.Series) else "",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["review_verdict", "mean_sharpe", "positive_row_share", "worst_sharpe"],
        ascending=[False, False, False, False],
    )


def render_factor_candidate_review_report(payload: dict[str, Any], review: pd.DataFrame) -> str:
    lines = [
        "# RandysLab Factor Candidate Review",
        "",
        "This is a conservative research review of strict sensitivity evidence.",
        "It is not a runtime publish payload, factor admission decision, portfolio construction step, or live "
        "execution plan.",
        "",
        "## Run",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Source detail CSV: `{payload['source']['detail_csv']}`",
        f"- Candidate count: `{payload['candidate_count']}`",
        f"- Verdict counts: `{payload['verdict_counts']}`",
        "",
        "## Candidate Review",
        "",
        "| Candidate | Verdict | Mean Sharpe | Median Sharpe | Worst Sharpe | Positive Rows | Mean Max DD | "
        "Worst Max DD | Failures | Variant |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    if review.empty:
        lines.append("| none | none | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0000 | 0.0000 | none | none |")
    else:
        for row in review.to_dict(orient="records"):
            failures = row.get("failure_reasons", "") or "none"
            lines.append(
                "| {candidate_id} | `{review_verdict}` | {mean_sharpe:.4f} | {median_sharpe:.4f} | "
                "{worst_sharpe:.4f} | {positive_rows:.0f}/{completed_rows:.0f} | {mean_max_dd:.4f} | "
                "{worst_max_dd:.4f} | `{failures}` | `{variant_id}` |".format(
                    failures=failures,
                    candidate_id=row["candidate_id"],
                    variant_id=row.get("variant_id", "default"),
                    review_verdict=row["review_verdict"],
                    mean_sharpe=float(row["mean_sharpe"]),
                    median_sharpe=float(row["median_sharpe"]),
                    worst_sharpe=float(row["worst_sharpe"]),
                    positive_rows=float(row["positive_rows"]),
                    completed_rows=float(row["completed_rows"]),
                    mean_max_dd=float(row["mean_max_dd"]),
                    worst_max_dd=float(row["worst_max_dd"]),
                )
            )
    if not review.empty and "intended_scope" in review.columns:
        lines.extend(
            [
                "",
                "## Scope Contract",
                "",
                "| Candidate | Variant | Scope Mode | Intended Scope | Applicability Hypothesis | Out-of-Scope Policy |",
                "|---|---|---|---|---|---|",
            ]
        )
        for row in review.to_dict(orient="records"):
            lines.append(
                "| {candidate_id} | `{variant_id}` | `{scope_mode}` | `{intended_scope}` | {applicability_hypothesis} | "
                "`{out_of_scope_policy}` |".format(
                    candidate_id=_table_text(row.get("candidate_id", "")),
                    variant_id=_table_text(row.get("variant_id", "default")),
                    scope_mode=_table_text(row.get("scope_mode", "")),
                    intended_scope=_table_text(row.get("intended_scope", "")),
                    applicability_hypothesis=_table_text(row.get("applicability_hypothesis", "")),
                    out_of_scope_policy=_table_text(row.get("out_of_scope_policy", "")),
                )
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Conservative review verdicts are diagnostic research labels only.",
            "- A `research_watchlist` label is not factor admission.",
            "- A blocked label should preserve the failure reason for future factor-factory memory.",
            "- No exchange trading credentials, no live orders, and no QuantumRandy runtime update.",
            "",
        ]
    )
    return "\n".join(lines)


def _failure_reasons(completed: pd.DataFrame, group: pd.DataFrame, rules: ReviewRules) -> list[str]:
    if completed.empty:
        return ["no_completed_rows"]
    if rules.scope_mode not in {"multi_asset", "declared"}:
        raise ValueError("scope_mode must be one of multi_asset or declared")
    failures: list[str] = []
    positive = completed[completed["sharpe"] > 0]
    window_means = _mean_by(completed, "window", "sharpe")
    asset_positive_counts = positive["asset"].value_counts().to_dict()
    scope_asset_count = int(completed["asset"].nunique())
    if len(completed) < rules.min_completed_rows:
        failures.append("too_few_completed_rows")
    if len(completed) < len(group):
        failures.append("has_failed_rows")
    if _mean(completed, "sharpe") < rules.min_mean_sharpe:
        failures.append("low_mean_sharpe")
    if _median(completed, "sharpe") < rules.min_median_sharpe:
        failures.append("low_median_sharpe")
    if _ratio(len(positive), len(completed)) < rules.min_positive_row_share:
        failures.append("low_positive_row_share")
    min_positive_assets = rules.min_positive_assets
    if rules.scope_mode == "declared":
        min_positive_assets = max(1, min(rules.min_positive_assets, scope_asset_count))
    if len(positive["asset"].unique()) < min_positive_assets:
        failures.append("too_few_positive_assets")
    if window_means.get("validation", 0.0) < rules.min_validation_mean_sharpe:
        failures.append("weak_validation_window")
    if window_means.get("blind", 0.0) < rules.min_blind_mean_sharpe:
        failures.append("weak_blind_window")
    if (
        (rules.scope_mode != "declared" or scope_asset_count > 1)
        and _max_positive_asset_share(asset_positive_counts) > rules.max_positive_asset_share
    ):
        failures.append("positive_evidence_asset_concentrated")
    if _mean(completed, "max_dd") > rules.max_mean_max_dd:
        failures.append("high_mean_drawdown")
    if _max(completed, "max_dd") > rules.max_worst_max_dd:
        failures.append("extreme_row_drawdown")
    return failures


def _mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return round(float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).mean()), 8)


def _median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return round(float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).median()), 8)


def _min(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return round(float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).min()), 8)


def _max(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return round(float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).max()), 8)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 8)


def _mean_by(frame: pd.DataFrame, group_column: str, value_column: str) -> dict[Any, float]:
    if frame.empty:
        return {}
    grouped = frame.groupby(group_column, dropna=False)[value_column].mean()
    return {key: round(float(value), 8) for key, value in grouped.items()}


def _format_float_map(values: dict[Any, float]) -> str:
    return "|".join(f"{key}:{value:.8f}" for key, value in sorted(values.items(), key=lambda item: str(item[0])))


def _max_positive_asset_share(counts: dict[Any, int]) -> float:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return 0.0
    return round(max(int(value) for value in counts.values()) / total, 8)


def _table_text(value: Any) -> str:
    return str(value).replace("|", "/").replace("\n", " ").strip()
