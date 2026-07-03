# Selector v0.8.2 Watchlist Robustness Gauntlet Report

Date: 2026-07-03

Scope: research-only robustness stress for selector v0.8.2 participation-plus-realized-volatility watchlist variants:

```text
mean(zscore(ema(volume,48),120), zscore(std(close,48),144))
```

This campaign did not run selector evidence61, did not use live trading, did not use exchange private keys, did not
publish runtime factors, did not modify active runtime strategies, did not create RandyPortfolio, and did not migrate
QuantumRandy portfolio modules.

## Verdict

No tested variant remains `research_watchlist` after the stricter robustness gauntlet. The whole tested family is
downgraded to:

```text
blocked pending new hypotheses
```

The strongest prior watchlist variant remains the least fragile member:

```text
thr_0p5_long_flat_cap_0p5_none_dd_stop_35_cd_42
```

It survived `15/16` stress scenarios, including fee/slippage, funding, combined harsh-cost, leave-one-asset-out,
blind-only, validation-only, and the 2021-2022 crash-focused window. It failed the 2020 COVID crash-focused window on
low mean/median Sharpe and low positive-row share, led by BTC/ETH weakness. Under the conservative all-scenario gate,
that is not robust enough to preserve a watchlist label.

## Artifacts

Robustness code:

```text
randyslab/factor_candidate_robustness.py
scripts/run_watchlist_robustness_gauntlet.py
tests/test_factor_candidate_robustness.py
```

Robustness output:

```text
reports/factor_candidate_robustness/selector_v082_combo_watchlist_robustness_gauntlet/
reports/factor_candidate_robustness/selector_v082_combo_watchlist_robustness_gauntlet/watchlist_robustness_detail.csv
reports/factor_candidate_robustness/selector_v082_combo_watchlist_robustness_gauntlet/watchlist_robustness_scenario_summary.csv
reports/factor_candidate_robustness/selector_v082_combo_watchlist_robustness_gauntlet/watchlist_robustness_variant_ranking.csv
reports/factor_candidate_robustness/selector_v082_combo_watchlist_robustness_gauntlet/watchlist_robustness_report.md
reports/factor_candidate_robustness/selector_v082_combo_watchlist_robustness_gauntlet/watchlist_robustness_summary.json
```

Command:

```bash
python3 scripts/run_watchlist_robustness_gauntlet.py
```

Run size:

- Detail rows: `1281`.
- Scenario summaries: `112`.
- Variants tested: `7`.
- Stress scenarios: `16`.

Empty custom crash windows are skipped instead of counted as failures. This matters for SOL/AVAX in the 2020 crash
window because the local datasets begin after that episode.

## Stress Design

The gauntlet compared:

- best drawdown-stop long-flat variant;
- best capped long-flat variant without drawdown stop;
- best volatility-filtered long-flat variant;
- full-exposure long-flat control;
- best capped long-short comparator;
- full-exposure drawdown-stop near miss;
- full-exposure long-short near miss.

Stress families:

- baseline all windows;
- `1.5x` and `2x` taker-fee/slippage scenarios;
- `1.5x` and `2x` funding multiplier scenarios;
- `2x` and `2.5x` combined harsh-cost scenarios;
- 2020 COVID and 2021-2022 crash-focused windows;
- leave-one-asset-out tests for BTC, ETH, SOL, BNB, and AVAX;
- validation-only and blind-only stress rankings.

Scenario-level conservative gates blocked low mean Sharpe, low median Sharpe, low positive-row share, weak validation or
blind rows, high mean max drawdown, extreme row drawdown, and failed rows. Variant-level final review required survival
across all stress scenarios.

## Variant Ranking

| Variant | Final Verdict | Survival | Mean Sharpe | Median Sharpe | Worst Sharpe | Mean Max DD | Worst Max DD | Validation Mean Sharpe | Blind Mean Sharpe | Labels |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `thr_0p5_long_flat_cap_0p5_none_dd_stop_35_cd_42` | `blocked_pending_new_hypotheses` | 15/16 | 0.5255 | 0.6406 | -2.1651 | 0.2824 | 0.6005 | 0.3126 | 0.8714 | `btc_weakness|sol_avax_concentration|validation_weakness` |
| `thr_0p5_long_flat_cap_0p5_none_none` | `blocked_pending_new_hypotheses` | 13/16 | 0.5193 | 0.5090 | -2.1651 | 0.2946 | 0.7014 | 0.2943 | 0.8714 | `btc_weakness|sol_avax_concentration|validation_weakness|crash_period_drawdown` |
| `thr_0p5_long_flat_cap_1p0_rvz_lte_1p5_none` | `blocked_pending_new_hypotheses` | 9/16 | 0.4511 | 0.2939 | -1.9383 | 0.3755 | 0.6564 | 0.1055 | 0.2409 | `btc_weakness|sol_avax_concentration|validation_weakness|blind_weakness|fee_fragility|funding_fragility|asset_exclusion_fragility|crash_period_drawdown` |
| `thr_0p5_long_short_cap_0p75_none_none` | `blocked_pending_new_hypotheses` | 7/16 | 0.4086 | 0.6541 | -2.9200 | 0.4644 | 0.7978 | 0.2678 | 0.9488 | `btc_weakness|sol_avax_concentration|validation_weakness|fee_fragility|funding_fragility|asset_exclusion_fragility|crash_period_drawdown` |
| `thr_0p5_long_flat_cap_1p0_none_none` | `blocked_pending_new_hypotheses` | 1/16 | 0.5710 | 0.5592 | -2.1322 | 0.4906 | 0.9259 | 0.3367 | 0.9382 | `btc_weakness|sol_avax_concentration|validation_weakness|fee_fragility|funding_fragility|asset_exclusion_fragility|crash_period_drawdown` |
| `thr_0p5_long_flat_cap_1p0_none_dd_stop_35_cd_42` | `blocked_pending_new_hypotheses` | 1/16 | 0.4408 | 0.7535 | -3.3489 | 0.4833 | 0.8337 | 0.2962 | 0.8892 | `btc_weakness|sol_avax_concentration|validation_weakness|fee_fragility|funding_fragility|asset_exclusion_fragility|crash_period_drawdown` |
| `thr_0p5_long_short_cap_1p0_none_none` | `blocked_pending_new_hypotheses` | 1/16 | 0.4225 | 0.6735 | -2.9099 | 0.5679 | 0.8983 | 0.2792 | 0.9665 | `btc_weakness|sol_avax_concentration|validation_weakness|fee_fragility|funding_fragility|asset_exclusion_fragility|crash_period_drawdown` |

## Compact Metric Leaders

| Ranking Metric | Best Variant | Value | Verdict |
|---|---|---:|---|
| Mean Sharpe | `thr_0p5_long_flat_cap_1p0_none_none` | 0.5710 | `blocked_pending_new_hypotheses` |
| Median Sharpe | `thr_0p5_long_flat_cap_1p0_none_dd_stop_35_cd_42` | 0.7535 | `blocked_pending_new_hypotheses` |
| Worst Sharpe | `thr_0p5_long_flat_cap_1p0_rvz_lte_1p5_none` | -1.9383 | `blocked_pending_new_hypotheses` |
| Mean Max DD | `thr_0p5_long_flat_cap_0p5_none_dd_stop_35_cd_42` | 0.2824 | `blocked_pending_new_hypotheses` |
| Worst Max DD | `thr_0p5_long_flat_cap_0p5_none_dd_stop_35_cd_42` | 0.6005 | `blocked_pending_new_hypotheses` |
| Validation Mean Sharpe | `thr_0p5_long_flat_cap_1p0_none_none` | 0.3367 | `blocked_pending_new_hypotheses` |
| Blind Mean Sharpe | `thr_0p5_long_short_cap_1p0_none_none` | 0.9665 | `blocked_pending_new_hypotheses` |
| Stress Survival Count | `thr_0p5_long_flat_cap_0p5_none_dd_stop_35_cd_42` | 15/16 | `blocked_pending_new_hypotheses` |

## What Survived

The half-cap drawdown-stop variant survived all direct cost/funding stresses:

- `fee_slip_1p5x`: mean Sharpe `0.6184`, worst max DD `0.5917`.
- `fee_slip_2x`: mean Sharpe `0.5998`, worst max DD `0.5947`.
- `funding_1p5x`: mean Sharpe `0.6172`, worst max DD `0.5898`.
- `funding_2x`: mean Sharpe `0.5971`, worst max DD `0.5908`.
- `harsh_fee_slip_funding`: mean Sharpe `0.5486`, worst max DD `0.5966`.
- `very_harsh_fee_slip_funding`: mean Sharpe `0.5082`, worst max DD `0.6005`.

It also survived all leave-one-asset-out tests. Removing BTC improved its mean Sharpe to `0.7764`, while removing BNB
was the weakest leave-one-out case at `0.4859`. This says the top variant is not solely dependent on one asset, but it
still carries `btc_weakness`, `sol_avax_concentration`, and `validation_weakness` labels.

It survived the 2021-2022 crash-focused window with mean Sharpe `0.3319`, mean max drawdown `0.4490`, and worst max
drawdown `0.5888`.

## What Failed

The 2020 COVID crash window was the decisive blocker. Six of seven variants failed it. The top half-cap drawdown-stop
variant had mean Sharpe `-0.5020`, median weakness, low positive-row share, and worst Sharpe `-2.1651`. The only variant
that passed the 2020 crash slice was `thr_0p5_long_flat_cap_1p0_rvz_lte_1p5_none`, but that variant failed broader
validation, blind, harsh-cost, and asset-exclusion stresses.

The 2021-2022 crash window and validation-only review also blocked six of seven variants. The half-cap drawdown-stop
variant was the only one to survive both. The half-cap no-stop variant failed 2021-2022 and validation-only on high mean
drawdown (`0.4794`), even though its Sharpe remained positive.

Full-exposure controls remain blocked. The full long-flat control has the best overall mean Sharpe (`0.5710`) but only
survived `1/16` scenarios because drawdown stayed too high. Full-exposure long-short and full-exposure drawdown-stop
near misses also survived only `1/16`.

## Stressors That Mattered Most

Crash-period and validation-only stresses mattered most. They changed the conclusion from "top variant is watchlist" to
"family is blocked pending new hypotheses."

Fee, slippage, and funding stresses did not break the two half-cap long-flat variants, but they did expose fragility in
full-exposure, volatility-filtered, and long-short variants. The stress labels `fee_fragility` and `funding_fragility`
therefore apply to the family, but not to the top half-cap drawdown-stop variant.

Asset exclusion mattered mainly as a filter against full-exposure and long-short variants. The two half-cap long-flat
variants survived all leave-one-asset-out tests, while full-exposure variants survived none.

## Drawdown-Stop Read

The drawdown-stop cooldown is doing real drawdown mitigation in the 2021-2022/validation stress rather than merely
changing labels:

| Variant | 2021-2022 Verdict | Mean Sharpe | Mean Max DD | Worst Max DD |
|---|---|---:|---:|---:|
| `thr_0p5_long_flat_cap_0p5_none_dd_stop_35_cd_42` | `research_watchlist` | 0.3319 | 0.4490 | 0.5888 |
| `thr_0p5_long_flat_cap_0p5_none_none` | `blocked_by_robustness_stress` | 0.3196 | 0.4794 | 0.6975 |

However, the stop does not solve the 2020 BTC/ETH crash weakness. In that slice, the half-cap stop and half-cap no-stop
variants both failed with mean Sharpe `-0.5020` and worst Sharpe `-2.1651`. The stop helps, but it is not a complete
robustness thesis.

## QuantumRandy Factor-Factory Memory

Feed back positive memory:

- Participation plus realized volatility remains a useful research theme.
- Long-flat half-cap variants are clearly more robust than full-exposure variants.
- Drawdown-stop cooldown can reduce 2021-2022/validation drawdown when paired with exposure control.

Feed back negative memory:

- Do not keep this family as `research_watchlist` after the stricter robustness gauntlet.
- Do not promote full-exposure variants, even when mean Sharpe ranks first.
- Do not treat realized-volatility filtering alone as robust; it passed 2020 but failed too many other stresses.
- Preserve `btc_weakness`, `sol_avax_concentration`, `validation_weakness`, `blind_weakness`, `crash_period_drawdown`,
  `fee_fragility`, `funding_fragility`, and `asset_exclusion_fragility` as explicit memory labels.
- New hypotheses should target crash-specific BTC/ETH behavior, not just lower exposure or drawdown stops.

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
