from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CostProfile, load_settings
from .data import align_funding_to_ohlcv, read_funding, read_ohlcv, slice_window
from .formula_candidates import judge_factor_candidate, load_factor_candidate_jsonl

DEFAULT_WINDOWS = ["training", "validation", "blind"]


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    threshold: float
    signal_mode: str
    exposure_cap: float
    volatility_cap_label: str = "none"
    volatility_cap_formula: str = ""
    volatility_cap_max: float | None = None
    drawdown_stop_label: str = "none"
    drawdown_stop_max: float | None = None
    drawdown_stop_cooldown_bars: int = 0
    mitigation_family: str = ""


@dataclass(frozen=True)
class StressScenario:
    scenario_id: str
    stress_family: str
    label: str
    windows: list[str] = field(default_factory=lambda: list(DEFAULT_WINDOWS))
    taker_bps: float | None = None
    slippage_bps: float | None = None
    fee_multiplier: float | None = None
    slippage_multiplier: float | None = None
    funding_multiplier: float | None = None
    custom_start: str | None = None
    custom_end: str | None = None
    custom_window_label: str | None = None
    asset_exclusions: list[str] = field(default_factory=list)


def parse_variant_specs(review: pd.DataFrame, variant_ids: list[str]) -> list[VariantSpec]:
    if review.empty:
        raise ValueError("review data is empty")
    if not variant_ids:
        raise ValueError("at least one variant id is required")

    rows = review.drop_duplicates("variant_id").set_index("variant_id", drop=False)
    missing = [variant_id for variant_id in variant_ids if variant_id not in rows.index]
    if missing:
        raise ValueError(f"variant ids not found: {missing}")

    variants: list[VariantSpec] = []
    for variant_id in variant_ids:
        row = rows.loc[variant_id].to_dict()
        parsed = _parse_variant_id(variant_id)
        volatility_label = str(_clean(row.get("volatility_cap_label")) or parsed["volatility_cap_label"])
        volatility_formula = str(_clean(row.get("volatility_cap_formula")) or _volatility_formula(volatility_label))
        volatility_max = _optional_float(row.get("volatility_cap_max"))
        if volatility_max is None:
            volatility_max = _volatility_max(volatility_label)
        drawdown_label = str(_clean(row.get("drawdown_stop_label")) or parsed["drawdown_stop_label"])
        drawdown_max = _optional_float(row.get("drawdown_stop_max"))
        if drawdown_max is None:
            drawdown_max = _drawdown_stop_max(drawdown_label)
        cooldown = _optional_int(row.get("drawdown_stop_cooldown_bars"))
        if cooldown is None:
            cooldown = _drawdown_stop_cooldown(drawdown_label)
        variants.append(
            VariantSpec(
                variant_id=variant_id,
                threshold=_optional_float(row.get("threshold")) or parsed["threshold"],
                signal_mode=str(_clean(row.get("signal_mode")) or parsed["signal_mode"]),
                exposure_cap=_optional_float(row.get("exposure_cap")) or parsed["exposure_cap"],
                volatility_cap_label=volatility_label,
                volatility_cap_formula=volatility_formula,
                volatility_cap_max=volatility_max,
                drawdown_stop_label=drawdown_label,
                drawdown_stop_max=drawdown_max,
                drawdown_stop_cooldown_bars=int(cooldown or 0),
                mitigation_family=str(_clean(row.get("mitigation_family")) or ""),
            )
        )
    return variants


def build_default_watchlist_stress_scenarios(assets: list[dict[str, str]]) -> list[StressScenario]:
    asset_labels = [_asset_label(asset) for asset in assets]
    scenarios = [
        StressScenario("base_all_windows", "baseline", "Base all windows"),
        StressScenario(
            "fee_slip_1p5x",
            "higher_costs",
            "Higher taker fee and slippage, 1.5x",
            taker_bps=6.0,
            slippage_bps=3.0,
        ),
        StressScenario(
            "fee_slip_2x",
            "higher_costs",
            "Higher taker fee and slippage, 2x",
            taker_bps=8.0,
            slippage_bps=4.0,
        ),
        StressScenario("funding_1p5x", "funding", "Funding multiplier 1.5x", funding_multiplier=1.5),
        StressScenario("funding_2x", "funding", "Funding multiplier 2x", funding_multiplier=2.0),
        StressScenario(
            "harsh_fee_slip_funding",
            "combined_harsh_costs",
            "Combined 2x taker fee, 2x slippage, 2x funding",
            taker_bps=8.0,
            slippage_bps=4.0,
            funding_multiplier=2.0,
        ),
        StressScenario(
            "very_harsh_fee_slip_funding",
            "combined_harsh_costs",
            "Combined 2.5x taker fee, 2.5x slippage, 2.5x funding",
            taker_bps=10.0,
            slippage_bps=5.0,
            funding_multiplier=2.5,
        ),
        StressScenario(
            "crash_2020_covid",
            "crash_period",
            "Crash-focused 2020 COVID window",
            windows=[],
            custom_start="2020-02-01",
            custom_end="2020-05-01",
            custom_window_label="crash_2020_covid",
        ),
        StressScenario(
            "crash_2021_2022",
            "crash_period",
            "Crash-focused 2021-2022 window",
            windows=[],
            custom_start="2021-01-01",
            custom_end="2023-01-01",
            custom_window_label="crash_2021_2022",
        ),
        StressScenario("validation_only", "validation_blind", "Validation-only stress", windows=["validation"]),
        StressScenario("blind_only", "validation_blind", "Blind-only stress", windows=["blind"]),
    ]
    scenarios.extend(
        StressScenario(
            scenario_id=f"exclude_{_safe_id(asset_label)}",
            stress_family="asset_exclusion",
            label=f"Leave out {asset_label}",
            asset_exclusions=[asset_label],
        )
        for asset_label in asset_labels
    )
    return scenarios


def build_selector_v082_crash_hypothesis_scenarios(assets: list[dict[str, str]]) -> list[StressScenario]:
    scenarios = build_default_watchlist_stress_scenarios(assets)
    asset_labels = [_asset_label(asset) for asset in assets]
    labels_upper = {label.upper() for label in asset_labels}

    sol_avax = [label for label in asset_labels if label.upper().startswith(("SOL", "AVAX"))]
    if sol_avax:
        scenarios.extend(
            [
                StressScenario(
                    "exclude_sol_avax",
                    "asset_exclusion",
                    "Leave out SOL and AVAX together",
                    asset_exclusions=sol_avax,
                ),
                StressScenario(
                    "validation_exclude_sol_avax",
                    "asset_exclusion",
                    "Validation-only leave out SOL and AVAX together",
                    windows=["validation"],
                    asset_exclusions=sol_avax,
                ),
            ]
        )

    if {"BTCUSDT", "ETHUSDT"}.issubset(labels_upper):
        btc_eth_only_exclusions = [
            label for label in asset_labels if not label.upper().startswith(("BTC", "ETH"))
        ]
        scenarios.extend(
            [
                StressScenario(
                    "btc_eth_only_all_windows",
                    "asset_exclusion",
                    "BTC and ETH only, all windows",
                    asset_exclusions=btc_eth_only_exclusions,
                ),
                StressScenario(
                    "btc_eth_only_validation",
                    "asset_exclusion",
                    "BTC and ETH only, validation window",
                    windows=["validation"],
                    asset_exclusions=btc_eth_only_exclusions,
                ),
                StressScenario(
                    "crash_2020_btc_eth_only",
                    "crash_period",
                    "Crash-focused 2020 COVID window, BTC and ETH only",
                    windows=[],
                    custom_start="2020-02-01",
                    custom_end="2020-05-01",
                    custom_window_label="crash_2020_btc_eth_only",
                    asset_exclusions=btc_eth_only_exclusions,
                ),
            ]
        )

    return scenarios


def run_watchlist_robustness_gauntlet(
    *,
    config_path: str | Path,
    candidate_jsonl: str | Path,
    assets: list[dict[str, str]],
    variants: list[VariantSpec],
    scenarios: list[StressScenario],
    out_dir: str | Path,
) -> dict[str, Any]:
    if not assets:
        raise ValueError("at least one asset is required")
    if not variants:
        raise ValueError("at least one variant is required")
    if not scenarios:
        raise ValueError("at least one stress scenario is required")

    config_path = Path(config_path).resolve()
    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    candidate_jsonl = _resolve_path(base_dir, candidate_jsonl)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    settings = load_settings(config_path)
    candidates = load_factor_candidate_jsonl(candidate_jsonl)
    market_cache = _load_market_cache(base_dir, settings, assets)
    rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        scenario_costs = _scenario_costs(settings.costs, scenario)
        included_assets = _included_assets(assets, scenario)
        for asset in included_assets:
            asset_label = _asset_label(asset)
            full_ohlcv, full_funding = market_cache[asset_label]
            window_frames = _scenario_windows(settings, scenario, full_ohlcv, full_funding)
            for window_label, ohlcv, funding in window_frames:
                for variant in variants:
                    for candidate in candidates:
                        result = judge_factor_candidate(
                            candidate,
                            ohlcv=ohlcv,
                            funding=funding,
                            costs=scenario_costs,
                            risk_free_rate=settings.risk_free_rate,
                            signal_threshold=variant.threshold,
                            signal_mode=variant.signal_mode,
                            max_exposure_abs=variant.exposure_cap,
                            volatility_cap_formula=variant.volatility_cap_formula or None,
                            volatility_cap_max=variant.volatility_cap_max,
                            drawdown_stop_label=variant.drawdown_stop_label,
                            drawdown_stop_max=variant.drawdown_stop_max,
                            drawdown_stop_cooldown_bars=variant.drawdown_stop_cooldown_bars,
                        )
                        rows.append(
                            _detail_row(
                                result,
                                scenario=scenario,
                                costs=scenario_costs,
                                asset=asset_label,
                                window=window_label,
                                variant=variant,
                            )
                        )

    detail = pd.DataFrame(rows)
    scenario_summary, variant_ranking = build_robustness_summaries(detail)
    detail_csv = out / "watchlist_robustness_detail.csv"
    scenario_csv = out / "watchlist_robustness_scenario_summary.csv"
    ranking_csv = out / "watchlist_robustness_variant_ranking.csv"
    detail.to_csv(detail_csv, index=False)
    scenario_summary.to_csv(scenario_csv, index=False)
    variant_ranking.to_csv(ranking_csv, index=False)

    payload = {
        "artifact_type": "randyslab_watchlist_robustness_gauntlet",
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "role": "strict_watchlist_robustness_stress",
        "safety": {
            "research_only": True,
            "paper_only": True,
            "no_live_orders": True,
            "no_exchange_trading_keys": True,
            "not_runtime_publish_payload": True,
            "does_not_update_quantumrandy_runtime": True,
            "does_not_admit_factors": True,
        },
        "config_path": str(config_path),
        "candidate_jsonl": str(candidate_jsonl),
        "assets": assets,
        "variants": [asdict(variant) for variant in variants],
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "detail_row_count": int(len(detail)),
        "scenario_summary_count": int(len(scenario_summary)),
        "variant_count": int(len(variant_ranking)),
        "artifacts": {
            "detail_csv": detail_csv.name,
            "scenario_summary_csv": scenario_csv.name,
            "variant_ranking_csv": ranking_csv.name,
            "report_md": "watchlist_robustness_report.md",
        },
    }
    (out / "watchlist_robustness_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (out / "watchlist_robustness_report.md").write_text(
        render_watchlist_robustness_report(payload, scenario_summary, variant_ranking),
        encoding="utf-8",
    )
    return payload


def build_robustness_summaries(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = detail.copy()
    for column, default in {
        "candidate_id": "unknown",
        "formula": "",
        "mitigation_family": "",
        "asset_exclusions": "",
        "stress_family": "",
        "scenario_id": "",
        "window": "",
        "asset": "",
        "intended_scope": "",
        "status": "failed",
        "failure_reasons": "",
    }.items():
        if column not in frame.columns:
            frame[column] = default
    numeric_columns = ["sharpe", "max_dd", "cagr", "trades"]
    for column in numeric_columns:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    scenario_rows: list[dict[str, Any]] = []
    group_columns = ["candidate_id", "formula", "variant_id", "scenario_id", "stress_family"]
    for group_key, group in frame.groupby(group_columns, dropna=False):
        key_map = dict(zip(group_columns, group_key if isinstance(group_key, tuple) else (group_key,), strict=False))
        completed = group[group["status"] == "completed"].copy()
        diagnostic_failures = _conservative_failures(completed, group)
        labels = _robustness_labels(completed, str(key_map["stress_family"]), diagnostic_failures)
        intended_scope = _first(group, "intended_scope")
        scope_asset = _declared_scope_asset(intended_scope)
        scope_group, scope_completed, scope_hard_gate = _scope_hard_gate(group, scope_asset)
        scope_failures = _conservative_failures(scope_completed, scope_group) if scope_hard_gate else []
        if scope_hard_gate:
            conservative_verdict = "research_watchlist" if not scope_failures else "blocked_by_robustness_stress"
        else:
            conservative_verdict = "diagnostic_only_out_of_scope"
        scenario_rows.append(
            {
                **key_map,
                "mitigation_family": _first(group, "mitigation_family"),
                "asset_exclusions": _first(group, "asset_exclusions"),
                "intended_scope": intended_scope,
                "scope_asset": scope_asset,
                "scope_hard_gate": bool(scope_hard_gate),
                "scope_completed_rows": int(len(scope_completed)),
                "scope_total_rows": int(len(scope_group)),
                "conservative_verdict": conservative_verdict,
                "failure_reasons": "|".join(scope_failures),
                "diagnostic_verdict": "research_watchlist"
                if not diagnostic_failures
                else "blocked_by_robustness_stress",
                "diagnostic_failure_reasons": "|".join(diagnostic_failures),
                "robustness_labels": "|".join(labels),
                "completed_rows": int(len(completed)),
                "total_rows": int(len(group)),
                "positive_rows": int((completed["sharpe"] > 0).sum()) if not completed.empty else 0,
                "positive_row_share": _ratio(int((completed["sharpe"] > 0).sum()), len(completed)),
                "mean_sharpe": _mean(completed, "sharpe"),
                "median_sharpe": _median(completed, "sharpe"),
                "worst_sharpe": _min(completed, "sharpe"),
                "mean_max_dd": _mean(completed, "max_dd"),
                "worst_max_dd": _max(completed, "max_dd"),
                "validation_mean_sharpe": _window_mean(completed, "validation"),
                "blind_mean_sharpe": _window_mean(completed, "blind"),
                "scope_mean_sharpe": _mean(scope_completed, "sharpe"),
                "scope_median_sharpe": _median(scope_completed, "sharpe"),
                "scope_worst_sharpe": _min(scope_completed, "sharpe"),
                "scope_mean_max_dd": _mean(scope_completed, "max_dd"),
                "scope_worst_max_dd": _max(scope_completed, "max_dd"),
                "scope_validation_mean_sharpe": _window_mean(scope_completed, "validation"),
                "scope_blind_mean_sharpe": _window_mean(scope_completed, "blind"),
                "worst_asset": _worst_asset(completed),
            }
        )
    scenario_summary = pd.DataFrame(scenario_rows).sort_values(
        ["conservative_verdict", "mean_sharpe", "worst_max_dd"],
        ascending=[False, False, True],
    )
    variant_ranking = _build_variant_ranking(scenario_summary)
    return scenario_summary, variant_ranking


def render_watchlist_robustness_report(
    payload: dict[str, Any],
    scenario_summary: pd.DataFrame,
    variant_ranking: pd.DataFrame,
) -> str:
    lines = [
        "# Watchlist Robustness Gauntlet",
        "",
        "This is a research-only robustness stress report for watchlist candidate variants.",
        "It is not factor admission, portfolio construction, runtime publishing, or live execution approval.",
        "",
        "## Run",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Detail rows: `{payload['detail_row_count']}`",
        f"- Scenarios: `{len(payload['scenarios'])}`",
        f"- Variants: `{payload['variant_count']}`",
        "",
        "## Variant Ranking",
        "",
        "| Candidate | Variant | Verdict | Survival | Mean Sharpe | Median Sharpe | Worst Sharpe | Mean Max DD | Worst Max DD | "
        "Validation Mean Sharpe | Blind Mean Sharpe | Labels |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    if variant_ranking.empty:
        lines.append(
            "| none | none | none | 0/0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | none |"
        )
    else:
        for row in variant_ranking.to_dict(orient="records"):
            survival = f"{int(row['stress_survival_count'])}/{int(row['stress_count'])}"
            labels = row.get("robustness_labels", "") or "none"
            lines.append(
                "| `{candidate_id}` | `{variant_id}` | `{conservative_verdict}` | {survival} | {mean_sharpe:.4f} | "
                "{median_sharpe:.4f} | {worst_sharpe:.4f} | {mean_max_dd:.4f} | {worst_max_dd:.4f} | "
                "{validation_mean_sharpe:.4f} | {blind_mean_sharpe:.4f} | `{labels}` |".format(
                    survival=survival,
                    labels=labels,
                    **row,
                )
            )
    lines.extend(
        [
            "",
            "## Scenario Stress Summary",
            "",
            "| Scenario | Family | Candidate | Variant | Verdict | Mean Sharpe | Worst Sharpe | Mean Max DD | "
            "Worst Max DD | Labels |",
            "|---|---|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    if scenario_summary.empty:
        lines.append("| none | none | none | none | none | 0.0000 | 0.0000 | 0.0000 | 0.0000 | none |")
    else:
        for row in scenario_summary.to_dict(orient="records"):
            labels = row.get("robustness_labels", "") or "none"
            lines.append(
                "| `{scenario_id}` | `{stress_family}` | `{candidate_id}` | `{variant_id}` | `{conservative_verdict}` | "
                "{mean_sharpe:.4f} | {worst_sharpe:.4f} | {mean_max_dd:.4f} | {worst_max_dd:.4f} | "
                "`{labels}` |".format(labels=labels, **row)
            )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Research/backtest diagnostics only.",
            "- No exchange trading credentials.",
            "- No live order placement.",
            "- No selector evidence61 run.",
            "- No automatic QuantumRandy runtime publishing.",
            "- Stress verdicts are conservative research labels, not production approvals.",
            "- Declared-scope assets are hard-gated; out-of-scope rows remain diagnostic labels.",
            "",
        ]
    )
    return "\n".join(lines)


def _detail_row(
    result: dict[str, Any],
    *,
    scenario: StressScenario,
    costs: CostProfile,
    asset: str,
    window: str,
    variant: VariantSpec,
) -> dict[str, Any]:
    metrics = result.get("metrics", {}) or {}
    return {
        "scenario_id": scenario.scenario_id,
        "stress_family": scenario.stress_family,
        "stress_label": scenario.label,
        "asset_exclusions": "|".join(scenario.asset_exclusions),
        "asset": asset,
        "window": window,
        "threshold": variant.threshold,
        "signal_mode": variant.signal_mode,
        "candidate_id": result.get("candidate_id", ""),
        "formula": result.get("formula", ""),
        "intended_scope": result.get("intended_scope", ""),
        "applicability_hypothesis": result.get("applicability_hypothesis", ""),
        "out_of_scope_policy": result.get("out_of_scope_policy", ""),
        "variant_id": variant.variant_id,
        "mitigation_family": variant.mitigation_family,
        "exposure_cap": variant.exposure_cap,
        "volatility_cap_label": variant.volatility_cap_label,
        "volatility_cap_formula": variant.volatility_cap_formula,
        "volatility_cap_max": variant.volatility_cap_max if variant.volatility_cap_max is not None else "",
        "drawdown_stop_label": variant.drawdown_stop_label,
        "drawdown_stop_max": variant.drawdown_stop_max if variant.drawdown_stop_max is not None else "",
        "drawdown_stop_cooldown_bars": variant.drawdown_stop_cooldown_bars,
        "taker_bps": costs.taker_bps,
        "slippage_bps": costs.slippage_bps,
        "fee_multiplier": costs.fee_multiplier,
        "slippage_multiplier": costs.slippage_multiplier,
        "funding_multiplier": costs.funding_multiplier,
        "status": result.get("status", ""),
        "failure_reasons": "|".join(result.get("failure_reasons", [])),
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


def _build_variant_ranking(scenario_summary: pd.DataFrame) -> pd.DataFrame:
    if scenario_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (candidate_id, formula, variant_id), group in scenario_summary.groupby(
        ["candidate_id", "formula", "variant_id"], dropna=False
    ):
        labels = _join_unique(group["robustness_labels"])
        hard_group = _hard_gate_scenarios(group)
        metric_group = hard_group if not hard_group.empty else group
        failures = _join_unique(hard_group["failure_reasons"]) if not hard_group.empty else ""
        diagnostic_failures = (
            _join_unique(group["diagnostic_failure_reasons"]) if "diagnostic_failure_reasons" in group else ""
        )
        survival_count = int((hard_group["conservative_verdict"] == "research_watchlist").sum())
        stress_count = int(len(hard_group))
        rows.append(
            {
                "candidate_id": candidate_id,
                "formula": formula,
                "variant_id": variant_id,
                "mitigation_family": _first(group, "mitigation_family"),
                "intended_scope": _first(group, "intended_scope"),
                "scope_asset": _first(group, "scope_asset"),
                "conservative_verdict": "research_watchlist"
                if stress_count > 0 and survival_count == stress_count
                else "blocked_pending_new_hypotheses",
                "failure_reasons": failures,
                "diagnostic_failure_reasons": diagnostic_failures,
                "robustness_labels": labels,
                "stress_survival_count": survival_count,
                "stress_count": stress_count,
                "diagnostic_scenario_count": int(len(group) - len(hard_group)),
                "stress_survival_score": _ratio(survival_count, stress_count),
                "mean_sharpe": _mean(metric_group, "scope_mean_sharpe"),
                "median_sharpe": _median(metric_group, "scope_median_sharpe"),
                "worst_sharpe": _min(metric_group, "scope_worst_sharpe"),
                "mean_max_dd": _mean(metric_group, "scope_mean_max_dd"),
                "worst_max_dd": _max(metric_group, "scope_worst_max_dd"),
                "validation_mean_sharpe": _mean_nonzero(metric_group, "scope_validation_mean_sharpe"),
                "blind_mean_sharpe": _mean_nonzero(metric_group, "scope_blind_mean_sharpe"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["stress_survival_score", "mean_sharpe", "worst_max_dd"],
        ascending=[False, False, True],
    )


def _hard_gate_scenarios(group: pd.DataFrame) -> pd.DataFrame:
    if "scope_hard_gate" not in group:
        return group
    return group[group["scope_hard_gate"].map(_truthy)].copy()


def _scope_hard_gate(group: pd.DataFrame, scope_asset: str) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    if not scope_asset:
        completed = group[group["status"] == "completed"].copy()
        return group, completed, True

    mask = _scope_asset_mask(group, scope_asset)
    if not bool(mask.any()):
        return group.iloc[0:0].copy(), group.iloc[0:0].copy(), False

    scope_group = group[mask].copy()
    scope_completed = scope_group[scope_group["status"] == "completed"].copy()
    return scope_group, scope_completed, True


def _scope_asset_mask(frame: pd.DataFrame, scope_asset: str) -> pd.Series:
    if not scope_asset or "asset" not in frame:
        return pd.Series([True] * len(frame), index=frame.index)
    assets = frame["asset"].map(_normalise_asset_value)
    return assets.eq(scope_asset) | assets.str.startswith(scope_asset)


def _declared_scope_asset(raw: Any) -> str:
    cleaned = _clean(raw)
    if cleaned is None:
        return ""
    text = str(cleaned).strip().upper()
    for separator in ["/", ":", "-", "."]:
        text = text.replace(separator, "_")
    parts = [part for part in text.split("_") if part]
    for part in parts:
        token = _normalise_asset_value(part)
        if _looks_like_asset_label(token):
            return token
    if len(parts) >= 2:
        token = _normalise_asset_value(parts[0] + parts[1])
        if _looks_like_asset_label(token):
            return token
    return ""


def _normalise_asset_value(raw: Any) -> str:
    return "".join(char for char in str(raw).strip().upper() if char.isalnum())


def _looks_like_asset_label(token: str) -> bool:
    quote_assets = ("USDT", "USDC", "BUSD", "USD", "BTC", "ETH")
    return any(token.endswith(quote) and len(token) > len(quote) for quote in quote_assets)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _conservative_failures(completed: pd.DataFrame, group: pd.DataFrame) -> list[str]:
    if completed.empty:
        return ["no_completed_rows"]
    failures: list[str] = []
    positive_share = _ratio(int((completed["sharpe"] > 0).sum()), len(completed))
    if len(completed) < len(group):
        failures.append("has_failed_rows")
    if _mean(completed, "sharpe") < 0.25:
        failures.append("low_mean_sharpe")
    if _median(completed, "sharpe") < 0.25:
        failures.append("low_median_sharpe")
    if positive_share < 0.55:
        failures.append("low_positive_row_share")
    if "validation" in set(completed["window"]) and _window_mean(completed, "validation") < 0.0:
        failures.append("weak_validation_window")
    if "blind" in set(completed["window"]) and _window_mean(completed, "blind") < 0.0:
        failures.append("weak_blind_window")
    if _mean(completed, "max_dd") > 0.45:
        failures.append("high_mean_drawdown")
    if _max(completed, "max_dd") > 0.80:
        failures.append("extreme_row_drawdown")
    return failures


def _robustness_labels(completed: pd.DataFrame, stress_family: str, failures: list[str]) -> list[str]:
    labels: list[str] = []
    if completed.empty:
        return ["no_completed_rows"]
    by_asset = completed.groupby("asset", dropna=False)["sharpe"].mean()
    if any(str(asset).upper().startswith("BTC") and value < 0.25 for asset, value in by_asset.items()):
        labels.append("btc_weakness")

    positive = completed[completed["sharpe"] > 0]
    if not positive.empty:
        sol_avax_positive = positive["asset"].astype(str).str.upper().str.startswith(("SOL", "AVAX")).sum()
        if _ratio(int(sol_avax_positive), len(positive)) > 0.55:
            labels.append("sol_avax_concentration")
    worst_asset = _worst_asset(completed)
    if worst_asset.upper().startswith(("SOL", "AVAX")):
        labels.append("sol_avax_concentration")

    if "validation" in set(completed["window"]) and (
        _window_mean(completed, "validation") < 0.25 or _window_min(completed, "validation") < -0.5
    ):
        labels.append("validation_weakness")
    if "blind" in set(completed["window"]) and (
        _window_mean(completed, "blind") < 0.25 or _window_min(completed, "blind") < -0.5
    ):
        labels.append("blind_weakness")
    if stress_family == "crash_period" and _max(completed, "max_dd") >= 0.60:
        labels.append("crash_period_drawdown")
    if stress_family in {"higher_costs", "combined_harsh_costs"} and failures:
        labels.append("fee_fragility")
    if stress_family in {"funding", "combined_harsh_costs"} and failures:
        labels.append("funding_fragility")
    if stress_family == "asset_exclusion" and failures:
        labels.append("asset_exclusion_fragility")
    return sorted(set(labels))


def _scenario_costs(base: CostProfile, scenario: StressScenario) -> CostProfile:
    return replace(
        base,
        taker_bps=base.taker_bps if scenario.taker_bps is None else float(scenario.taker_bps),
        slippage_bps=base.slippage_bps if scenario.slippage_bps is None else float(scenario.slippage_bps),
        fee_multiplier=base.fee_multiplier
        if scenario.fee_multiplier is None
        else float(scenario.fee_multiplier),
        slippage_multiplier=base.slippage_multiplier
        if scenario.slippage_multiplier is None
        else float(scenario.slippage_multiplier),
        funding_multiplier=base.funding_multiplier
        if scenario.funding_multiplier is None
        else float(scenario.funding_multiplier),
    )


def _load_market_cache(
    base_dir: Path,
    settings: Any,
    assets: list[dict[str, str]],
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    cache: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for asset in assets:
        label = _asset_label(asset)
        ohlcv_path = _resolve_path(base_dir, asset.get("ohlcv_csv") or settings.paths.ohlcv_csv)
        funding_path = _resolve_path(base_dir, asset.get("funding_csv") or settings.paths.funding_csv)
        ohlcv = read_ohlcv(ohlcv_path)
        funding = align_funding_to_ohlcv(ohlcv, read_funding(funding_path))
        cache[label] = (ohlcv, funding)
    return cache


def _scenario_windows(
    settings: Any,
    scenario: StressScenario,
    full_ohlcv: pd.DataFrame,
    full_funding: pd.Series,
) -> list[tuple[str, pd.DataFrame, pd.Series]]:
    if scenario.custom_start or scenario.custom_end:
        label = scenario.custom_window_label or scenario.scenario_id
        ohlcv = slice_window(full_ohlcv, scenario.custom_start, scenario.custom_end)
        if ohlcv.empty:
            return []
        return [(label, ohlcv, full_funding.reindex(ohlcv.index))]
    frames: list[tuple[str, pd.DataFrame, pd.Series]] = []
    for window in scenario.windows or DEFAULT_WINDOWS:
        if window == "all":
            ohlcv = full_ohlcv
        else:
            ohlcv = slice_window(
                full_ohlcv,
                getattr(settings.windows, f"{window}_start"),
                getattr(settings.windows, f"{window}_end"),
            )
        if ohlcv.empty:
            continue
        frames.append((window, ohlcv, full_funding.reindex(ohlcv.index)))
    return frames


def _included_assets(assets: list[dict[str, str]], scenario: StressScenario) -> list[dict[str, str]]:
    exclusions = {label.upper() for label in scenario.asset_exclusions}
    return [asset for asset in assets if _asset_label(asset).upper() not in exclusions]


def _parse_variant_id(variant_id: str) -> dict[str, Any]:
    parts = variant_id.split("_")
    if len(parts) < 6 or parts[0] != "thr" or "cap" not in parts:
        raise ValueError(f"cannot parse variant id: {variant_id}")
    cap_idx = parts.index("cap")
    threshold = _id_float(parts[1])
    signal_mode = "_".join(parts[2:cap_idx])
    exposure_cap = _id_float(parts[cap_idx + 1])
    tail = parts[cap_idx + 2 :]
    if "dd" in tail:
        dd_idx = tail.index("dd")
        volatility_cap_label = "_".join(tail[:dd_idx]) or "none"
        drawdown_stop_label = "_".join(tail[dd_idx:])
    else:
        volatility_cap_label = "_".join(tail) or "none"
        drawdown_stop_label = "none"
    return {
        "threshold": threshold,
        "signal_mode": signal_mode,
        "exposure_cap": exposure_cap,
        "volatility_cap_label": volatility_cap_label,
        "drawdown_stop_label": drawdown_stop_label,
    }


def _volatility_formula(label: str) -> str:
    return "" if label == "none" else "zscore(std(close,48),144)"


def _volatility_max(label: str) -> float | None:
    if label.startswith("rvz_lte_"):
        return _id_float(label.replace("rvz_lte_", ""))
    return None


def _drawdown_stop_max(label: str) -> float | None:
    parts = label.split("_")
    if len(parts) >= 3 and parts[0] == "dd" and parts[1] == "stop":
        value = _id_float(parts[2])
        return value / 100.0 if value > 1.0 else value
    return None


def _drawdown_stop_cooldown(label: str) -> int:
    parts = label.split("_")
    if "cd" not in parts:
        return 0
    idx = parts.index("cd")
    return int(_id_float(parts[idx + 1])) if idx + 1 < len(parts) else 0


def _id_float(value: str) -> float:
    return float(str(value).replace("m", "-").replace("p", "."))


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if str(value).strip().lower() in {"", "nan", "none", "null"}:
        return None
    return value


def _optional_float(value: Any) -> float | None:
    cleaned = _clean(value)
    if cleaned is None:
        return None
    return float(cleaned)


def _optional_int(value: Any) -> int | None:
    cleaned = _clean(value)
    if cleaned is None:
        return None
    return int(float(cleaned))


def _asset_label(asset: dict[str, str]) -> str:
    label = str(asset.get("symbol") or asset.get("label") or "").strip()
    if label:
        return label
    return Path(asset.get("ohlcv_csv", "asset")).stem.replace("_4h", "")


def _resolve_path(base_dir: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _safe_id(raw: str) -> str:
    text = str(raw).strip().lower()
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in text) or "asset"


def _first(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame:
        return ""
    values = [str(value) for value in frame[column].dropna().unique() if str(value).strip()]
    return values[0] if values else ""


def _mean(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return round(float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).mean()), 8)


def _mean_nonzero(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    values = values[values != 0.0]
    return round(float(values.mean()), 8) if not values.empty else 0.0


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


def _window_mean(frame: pd.DataFrame, window: str) -> float:
    if frame.empty or "window" not in frame:
        return 0.0
    subset = frame[frame["window"] == window]
    return _mean(subset, "sharpe")


def _window_min(frame: pd.DataFrame, window: str) -> float:
    if frame.empty or "window" not in frame:
        return 0.0
    subset = frame[frame["window"] == window]
    return _min(subset, "sharpe")


def _worst_asset(frame: pd.DataFrame) -> str:
    if frame.empty or "max_dd" not in frame:
        return ""
    row = frame.sort_values("max_dd", ascending=False).iloc[0]
    return str(row.get("asset", ""))


def _join_unique(series: pd.Series) -> str:
    values: list[str] = []
    for raw in series.fillna(""):
        for item in str(raw).split("|"):
            item = item.strip()
            if item and item not in values:
                values.append(item)
    return "|".join(values)
