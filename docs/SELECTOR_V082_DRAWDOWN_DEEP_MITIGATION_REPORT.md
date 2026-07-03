# Selector v0.8.2 Drawdown Deep-Mitigation Report

Date: 2026-07-03

Scope: research-only RandysLab strict judging for the selector v0.8.2 participation-plus-realized-volatility combo:

```text
mean(zscore(ema(volume,48),120), zscore(std(close,48),144))
```

This campaign did not run selector evidence61, did not touch live execution, did not use exchange private keys, did
not publish runtime factors, did not modify active runtime strategies, did not create RandyPortfolio, and did not
migrate QuantumRandy portfolio modules.

## Verdict

The best current variant reaches RandysLab `research_watchlist`, but not factor admission:

```text
thr_0p5_long_flat_cap_0p5_none_dd_stop_35_cd_42
```

It pairs threshold `0.5`, `long_flat`, half exposure, no realized-volatility filter, and a research-only drawdown
cooldown rule at `35%` drawdown with a `42` bar cooldown.

Metrics:

- Mean Sharpe: `0.6419`.
- Median Sharpe: `0.6415`.
- Worst Sharpe: `-1.0450`.
- Positive rows: `14/15`.
- Mean max drawdown: `0.2683`.
- Worst max drawdown: `0.5888`.

Follow-up: a stricter watchlist robustness gauntlet later downgraded this family to `blocked pending new hypotheses`.
See `docs/SELECTOR_V082_WATCHLIST_ROBUSTNESS_GAUNTLET_REPORT.md` for the current robustness verdict. The variant above
remains the strongest member of this family, but it no longer retains `research_watchlist` status after stricter
crash-window validation.

The full-exposure long-flat control remains blocked:

```text
thr_0p5_long_flat_cap_1p0_none_none
```

It has higher mean Sharpe (`0.6909`) but fails conservative review for
`high_mean_drawdown|extreme_row_drawdown`, with mean max drawdown `0.4690` and worst max drawdown `0.9239`.

## Artifacts

Deep sensitivity sweep:

```text
reports/factor_candidate_sensitivity/selector_v082_combo_drawdown_deep_mitigation/
```

Conservative review:

```text
reports/factor_candidate_review/selector_v082_combo_drawdown_deep_mitigation/
reports/factor_candidate_review/selector_v082_combo_drawdown_deep_mitigation/deep_mitigation_variant_ranking.csv
reports/factor_candidate_review/selector_v082_combo_drawdown_deep_mitigation/deep_mitigation_family_summary.csv
```

Episode diagnostics:

```text
reports/factor_candidate_drawdown_diagnostics/selector_v082_combo_drawdown_deep_mitigation_core/
reports/factor_candidate_drawdown_diagnostics/selector_v082_combo_drawdown_deep_mitigation_core/drawdown_episode_summary.csv
reports/factor_candidate_drawdown_diagnostics/selector_v082_combo_drawdown_deep_mitigation_core/drawdown_failure_label_summary.csv
```

Research-only diagnostics code:

```text
randyslab/drawdown_diagnostics.py
scripts/summarize_factor_drawdowns.py
```

## Deep Sweep Design

The expanded strict sweep covered:

- Assets: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, AVAXUSDT.
- Windows: training, validation, blind.
- Thresholds: `0.5`, `1.0`, `1.5`, `2.0`.
- Signal modes: `long_short`, `long_flat`.
- Exposure caps: `1.0`, `0.75`, `0.5`, `0.25`.
- Realized-volatility filters: none, `zscore(std(close,48),144) <= 1.5`, and
  `zscore(std(close,48),144) <= 1.0`.
- Research drawdown cooldowns: none, `35%` drawdown with `42` bar cooldown, and `25%` drawdown with `42` bar
  cooldown.

The deep sweep produced `4320` strict rows and `288` variant-level conservative reviews.

| Verdict | Count |
|---|---:|
| `research_watchlist` | 44 |
| `blocked_by_conservative_rules` | 244 |

## Variant Ranking

| Variant | Verdict | Mean Sharpe | Median Sharpe | Worst Sharpe | Mean Max DD | Worst Max DD | Positive Rows |
|---|---|---:|---:|---:|---:|---:|---:|
| `thr_0p5_long_flat_cap_0p5_none_dd_stop_35_cd_42` | `research_watchlist` | 0.6419 | 0.6415 | -1.0450 | 0.2683 | 0.5888 | 14/15 |
| `thr_0p5_long_flat_cap_0p5_none_none` | `research_watchlist` | 0.6378 | 0.5469 | -0.8595 | 0.2785 | 0.6975 | 14/15 |
| `thr_0p5_long_short_cap_0p75_none_none` | `research_watchlist` | 0.5775 | 0.7131 | -0.8563 | 0.4493 | 0.7883 | 11/15 |
| `thr_0p5_long_flat_cap_1p0_rvz_lte_1p5_none` | `research_watchlist` | 0.5645 | 0.3077 | -0.5388 | 0.3641 | 0.6181 | 10/15 |
| `thr_0p5_long_flat_cap_1p0_none_none` | `blocked_by_conservative_rules` | 0.6909 | 0.6115 | -0.7943 | 0.4690 | 0.9239 | 14/15 |
| `thr_0p5_long_flat_cap_1p0_none_dd_stop_35_cd_42` | `blocked_by_conservative_rules` | 0.6285 | 0.7759 | -1.1620 | 0.4609 | 0.8238 | 12/15 |
| `thr_0p5_long_short_cap_1p0_none_none` | `blocked_by_conservative_rules` | 0.5918 | 0.7308 | -0.8394 | 0.5516 | 0.8919 | 11/15 |

## Mitigation Family Summary

| Family | Variants | Watchlist | Best Variant | Best Mean Sharpe | Best Mean Max DD | Best Worst Max DD |
|---|---:|---:|---|---:|---:|---:|
| Drawdown stop/cooldown | 192 | 29 | `thr_0p5_long_flat_cap_0p5_none_dd_stop_35_cd_42` | 0.6419 | 0.2683 | 0.5888 |
| Capped long-flat | 12 | 5 | `thr_0p5_long_flat_cap_0p5_none_none` | 0.6378 | 0.2785 | 0.6975 |
| Long-short family | 48 | 4 | `thr_0p5_long_short_cap_0p75_none_none` | 0.5775 | 0.4493 | 0.7883 |
| Vol-filtered long-flat | 8 | 3 | `thr_0p5_long_flat_cap_1p0_rvz_lte_1p5_none` | 0.5645 | 0.3641 | 0.6181 |
| Capped vol-filtered long-flat | 24 | 3 | `thr_0p5_long_flat_cap_0p75_rvz_lte_1p5_none` | 0.5371 | 0.2868 | 0.5024 |
| Full long-flat control | 4 | 0 | `thr_0p5_long_flat_cap_1p0_none_none` | 0.6909 | 0.4690 | 0.9239 |

## Worst Drawdown Episodes

The episode audit covered a focused core grid of `1440` rows and captured `3329` episodes.

Worst episode:

| Asset | Window | Variant | Peak | Trough | Max DD | Labels |
|---|---|---|---|---|---:|---|
| SOLUSDT | validation | `thr_0p5_long_flat_cap_1p0_none_none` | 2021-09-09 00:00 UTC | 2022-11-09 16:00 UTC | 0.9239 | `sol_avax_concentration|validation_weakness|crash_period_drawdown|extreme_row_drawdown` |

Frequent episode labels:

| Label | Episode Count |
|---|---:|
| `sol_avax_concentration` | 1394 |
| `validation_weakness` | 1241 |
| `crash_period_drawdown` | 950 |
| `btc_weakness` | 534 |
| `extreme_row_drawdown` | 16 |

Interpretation: the drawdown blocker is not only BTC weakness. The deepest row comes from SOLUSDT validation during a
long crash-period episode. SOL/AVAX positive evidence still carries concentration risk because the same assets can also
carry the worst drawdowns.

## What Reduced Drawdown

Exposure caps remained the cleanest drawdown reducer. Half exposure with threshold `0.5` moved the full long-flat
control from blocked to watchlist:

| Variant | Verdict | Mean Sharpe | Mean Max DD | Worst Max DD | Failures |
|---|---|---:|---:|---:|---|
| `thr_0p5_long_flat_cap_1p0_none_none` | `blocked_by_conservative_rules` | 0.6909 | 0.4690 | 0.9239 | `high_mean_drawdown|extreme_row_drawdown` |
| `thr_0p5_long_flat_cap_0p5_none_none` | `research_watchlist` | 0.6378 | 0.2785 | 0.6975 | `none` |

The best drawdown-stop result was not a standalone stop. It worked best when paired with half exposure:

| Variant | Verdict | Mean Sharpe | Mean Max DD | Worst Max DD | Failures |
|---|---|---:|---:|---:|---|
| `thr_0p5_long_flat_cap_0p5_none_dd_stop_35_cd_42` | `research_watchlist` | 0.6419 | 0.2683 | 0.5888 | `none` |

The `rvz_lte_1p5` realized-volatility filter also helped full exposure:

| Variant | Verdict | Mean Sharpe | Mean Max DD | Worst Max DD | Failures |
|---|---|---:|---:|---:|---|
| `thr_0p5_long_flat_cap_1p0_rvz_lte_1p5_none` | `research_watchlist` | 0.5645 | 0.3641 | 0.6181 | `none` |

## What Failed

Full exposure remained the wrong default for this combo. The full long-flat control still failed despite good mean
Sharpe.

Drawdown stops alone did not reliably fix the blocker. Full-exposure long-flat with the `35%` drawdown cooldown still
failed for `high_mean_drawdown|extreme_row_drawdown`; worst max drawdown remained `0.8238`.

The strict `rvz_lte_1p0` filter often reduced drawdown but frequently failed on weak median Sharpe, low positive-row
share, weak validation, or weak blind-window performance.

Long-short can be made watchlist with a cap, but the full long-short control remains blocked. Long-short also keeps
larger crash-period episode exposure and should not replace long-flat as the default interpretation.

## Remaining Blockers

- BTC weakness remains material, especially in validation episodes and weak Sharpe rows.
- SOL/AVAX concentration is double-edged: those assets supply much of the positive evidence but also dominate worst
  crash-period drawdown episodes.
- Validation remains the weakest window and is the main stress point for watchlist variants.
- Crash-period drawdowns around 2021-2022 are the primary root cause of `extreme_row_drawdown`.
- A `research_watchlist` verdict is still not factor admission, not runtime publishing, not portfolio construction, and
  not live execution approval.

## QuantumRandy Factor-Factory Memory

Feed back positive memory:

- Keep the participation-plus-realized-volatility blend as a useful research theme.
- Prefer long-flat interpretations over full long-short when using this exact combo as a diagnostic.
- Treat half exposure and moderate realized-volatility filtering as first-class strict-judge mitigation hints.

Feed back negative memory:

- Do not promote full-exposure variants of this combo.
- Do not treat drawdown stops as a cure when exposure remains too high.
- Preserve explicit failure labels for `btc_weakness`, `sol_avax_concentration`, `validation_weakness`, and
  `crash_period_drawdown`.
- Do not let SOL/AVAX positive rows hide crash-period drawdown concentration.

## Boundary

- Research-only.
- Paper/backtest artifacts only.
- No live exchange orders.
- No exchange trading credentials.
- No runtime strategy updates.
- No automatic factor admission.
- No RandyPortfolio creation.
- No QuantumRandy portfolio-module migration.
- RandysLab remains the strict judge; QuantumRandy remains the factor factory.
