# Selector v0.8.2 Drawdown Reduction Pass

Date: 2026-07-03

Scope: research-only RandysLab strict judging for the selector v0.8.2 participation-plus-realized-volatility combo:

```text
mean(zscore(ema(volume,48),120), zscore(std(close,48),144))
```

This pass did not run selector evidence61, did not touch live execution, did not use exchange private keys, did not
publish runtime factors, did not modify active runtime strategies, did not create RandyPortfolio, and did not migrate
QuantumRandy portfolio modules.

## Verdict

Some conservative long-flat variants now qualify as `research_watchlist` under the current RandysLab conservative
review rules. This is a diagnostic research label only. It is not factor admission, runtime publishing, portfolio
construction, or live execution approval.

The best drawdown-balanced watchlist variant was:

```text
threshold=0.5, signal_mode=long_flat, exposure_cap=0.5, volatility_cap=none
```

It reduced mean max drawdown from the full-exposure no-filter control's `0.4690` to `0.2785`, and reduced worst-row
drawdown from `0.9239` to `0.6975`, while keeping mean Sharpe `0.6378`, median Sharpe `0.5469`, and positive rows
`14/15`.

## Artifacts

Sensitivity artifacts:

```text
reports/factor_candidate_sensitivity/selector_v082_combo_drawdown_reduction_pass/
```

Conservative review artifacts:

```text
reports/factor_candidate_review/selector_v082_combo_drawdown_reduction_pass/
```

Scoped RandysLab-only combo candidate:

```text
reports/factor_candidate_combination_exports/selector_v082_participation_realized_vol_drawdown_reduction/
```

Additional diagnostics:

```text
reports/factor_candidate_review/selector_v082_combo_drawdown_reduction_pass/drawdown_variant_asset_diagnostics.csv
reports/factor_candidate_review/selector_v082_combo_drawdown_reduction_pass/drawdown_variant_window_diagnostics.csv
reports/factor_candidate_review/selector_v082_combo_drawdown_reduction_pass/drawdown_variant_worst_rows.csv
```

## Pass Design

All variants were long-flat only. The pass swept:

- Assets: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, AVAXUSDT.
- Windows: training, validation, blind.
- Thresholds: `0.5`, `1.0`, `1.5`, `2.0`.
- Exposure caps: `1.0`, `0.75`, `0.5`, `0.25`.
- Volatility caps: no filter, `zscore(std(close,48),144) <= 1.5`,
  `zscore(std(close,48),144) <= 1.0`.

The sensitivity pass produced `720` completed rows and the review produced `48` variant-level verdicts:

| Verdict | Count |
|---|---:|
| `research_watchlist` | 11 |
| `blocked_by_conservative_rules` | 37 |

## Watchlist Variants

| Variant | Mean Sharpe | Median Sharpe | Worst Sharpe | Positive Rows | Mean Max DD | Worst Max DD |
|---|---:|---:|---:|---:|---:|---:|
| `thr_0p5_long_flat_cap_0p5_none` | 0.6378 | 0.5469 | -0.8595 | 14/15 | 0.2785 | 0.6975 |
| `thr_0p5_long_flat_cap_1p0_rvz_lte_1p5` | 0.5645 | 0.3077 | -0.5388 | 10/15 | 0.3641 | 0.6181 |
| `thr_0p5_long_flat_cap_0p75_rvz_lte_1p5` | 0.5371 | 0.2905 | -0.5694 | 10/15 | 0.2868 | 0.5024 |
| `thr_0p5_long_flat_cap_0p25_none` | 0.5315 | 0.4442 | -0.9898 | 12/15 | 0.1529 | 0.4375 |
| `thr_1p0_long_flat_cap_0p75_none` | 0.4936 | 0.6994 | -0.8265 | 9/15 | 0.3435 | 0.7932 |
| `thr_0p5_long_flat_cap_0p5_rvz_lte_1p5` | 0.4824 | 0.2561 | -0.6306 | 10/15 | 0.2010 | 0.3623 |
| `thr_1p0_long_flat_cap_0p5_none` | 0.4534 | 0.6434 | -0.8770 | 9/15 | 0.2467 | 0.6369 |
| `thr_1p0_long_flat_cap_0p25_none` | 0.3326 | 0.4752 | -1.0288 | 9/15 | 0.1334 | 0.3862 |
| `thr_1p0_long_flat_cap_1p0_rvz_lte_1p5` | 0.3120 | 0.3067 | -0.5799 | 9/15 | 0.2692 | 0.5767 |
| `thr_1p0_long_flat_cap_1p0_rvz_lte_1p0` | 0.2871 | 0.4387 | -0.8715 | 9/15 | 0.1992 | 0.3880 |
| `thr_1p0_long_flat_cap_0p75_rvz_lte_1p5` | 0.2744 | 0.2785 | -0.6056 | 9/15 | 0.2089 | 0.4640 |

## What Reduced Drawdown

Lower exposure caps gave the cleanest improvement. At threshold `0.5` with no volatility filter, reducing exposure from
`1.0` to `0.5` changed the review result from blocked to watchlist:

| Variant | Verdict | Mean Sharpe | Mean Max DD | Worst Max DD | Failures |
|---|---|---:|---:|---:|---|
| `thr_0p5_long_flat_cap_1p0_none` | `blocked_by_conservative_rules` | 0.6909 | 0.4690 | 0.9239 | `high_mean_drawdown|extreme_row_drawdown` |
| `thr_0p5_long_flat_cap_0p5_none` | `research_watchlist` | 0.6378 | 0.2785 | 0.6975 | `none` |

The `rvz_lte_1p5` volatility cap also helped the full-exposure threshold `0.5` variant by reducing worst-row drawdown
below the conservative cap:

| Variant | Verdict | Mean Sharpe | Mean Max DD | Worst Max DD | Failures |
|---|---|---:|---:|---:|---|
| `thr_0p5_long_flat_cap_1p0_rvz_lte_1p5` | `research_watchlist` | 0.5645 | 0.3641 | 0.6181 | `none` |

The stricter `rvz_lte_1p0` filter reduced drawdown but often removed too much useful exposure. It mostly failed on
low median Sharpe, low positive-row share, and weak validation or blind windows.

## What Still Failed

The full-exposure no-filter control at threshold `0.5` still failed for:

```text
high_mean_drawdown|extreme_row_drawdown
```

Other blocked variants preserved their exact failure reasons in:

```text
reports/factor_candidate_review/selector_v082_combo_drawdown_reduction_pass/factor_candidate_review.csv
```

The most common remaining blockers were:

- `extreme_row_drawdown` for high-exposure no-filter variants.
- `weak_validation_window` for stricter threshold or filtered variants.
- `low_median_sharpe`, `low_positive_row_share`, and `weak_blind_window` when volatility filters became too strict.
- `too_few_positive_assets` and `positive_evidence_asset_concentrated` in the sparsest high-threshold filtered cases.

## Asset And Window Diagnostics

Watchlist variants still show uneven evidence:

- SOLUSDT remained the strongest contributor, with watchlist mean Sharpe `0.8292`, but its worst watchlist drawdown
  reached `0.7932`.
- BNBUSDT was also strong in watchlist variants, with mean Sharpe `0.6730`.
- BTCUSDT improved enough to avoid the earlier all-blocking drawdown profile, but remained weak: watchlist median
  Sharpe was `-0.0596` and worst Sharpe was `-1.0288`.
- Validation remained the weakest window for watchlist variants, with mean Sharpe `0.1310` and worst drawdown `0.7932`.
- Training and blind windows were materially stronger for watchlist variants, with mean Sharpe `0.6144` and `0.5927`.

Interpretation: the pass reduced drawdown enough to create research-watchlist variants, but BTC weakness and validation
fragility remain first-class caveats.

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
