# Selector v0.8.2 Crash-Remediation Hypothesis Report

Date: 2026-07-03

Scope: research-only RandysLab strict robustness follow-up for selector v0.8.2 participation-plus-realized-volatility
memory. This pass targets the blockers from the prior gauntlet: BTC/ETH COVID crash weakness, validation-window
drawdown, and SOL/AVAX concentration risk.

This campaign did not run selector evidence61, did not use live trading, did not use exchange private keys, did not
publish runtime factors, did not modify active runtime strategies, did not create RandyPortfolio, and did not migrate
QuantumRandy portfolio modules.

## Verdict

No tested crash-remediation hypothesis earns `research_watchlist` after the expanded strict gate. All `49`
candidate-variant rows are:

```text
blocked_pending_new_hypotheses
```

The best new diagnostic row was:

```text
combo_volume48_ret24_calmvol_funding_calm_mean
thr_0p5_long_flat_cap_1p0_none_dd_stop_35_cd_42
```

It survived `16/21` scenarios, with mean Sharpe `0.5466`, median Sharpe `0.3854`, worst Sharpe `-0.7764`, mean max
drawdown `0.2815`, and worst max drawdown `0.7142`. It fixed the broad 2020 COVID crash slice, but failed the stricter
BTC/ETH-only COVID slice on low positive-row share and failed validation-focused slices on low mean/median Sharpe.

Conservative label:

```text
diagnostic_memory_only
```

## What Remains Useful

The prior robustness artifacts remain useful as factor-factory memory, not as watchlist evidence:

- Participation plus realized volatility is still a productive research theme.
- Half exposure and drawdown cooldowns remain useful mitigators, but they are insufficient by themselves.
- The decisive weakness is not generic fees or funding. It is crash-window and validation-window regime behavior.
- SOL/AVAX must be treated as both positive-evidence contributors and drawdown/concentration risks.

The new pass adds one useful memory update: adding a 24-bar trend guard, inverted realized volatility, and inverted
funding pressure can improve the COVID crash symptom. It is not robust enough for watchlist because validation and
paired-exclusion stresses remain weak.

## Hypotheses Tested

The export kept the original combo as a benchmark and added six research-only combo candidates:

- `combo_volume48_volatility48_ret24_mean`: original blend plus 24-bar return trend.
- `combo_volume48_volatility48_ret48_mean`: original blend plus 48-bar return trend.
- `combo_volume48_ret24_calmvol_mean`: participation plus 24-bar trend plus inverted realized volatility.
- `combo_volume48_ret48_calmvol_mean`: participation plus 48-bar trend plus inverted realized volatility.
- `combo_volume48_volatility48_funding_calm_mean`: original blend plus inverted funding-rate z-score.
- `combo_volume48_ret24_calmvol_funding_calm_mean`: participation plus 24-bar trend, inverted realized volatility, and
  inverted funding-rate z-score.

The expanded scenario set reused the prior 16 stresses and added:

- paired `exclude_sol_avax`;
- `validation_exclude_sol_avax`;
- `btc_eth_only_all_windows`;
- `btc_eth_only_validation`;
- `crash_2020_btc_eth_only`.

## Artifacts

Research-only hypothesis export:

```text
reports/factor_candidate_combination_exports/selector_v082_crash_remediation_hypotheses/
reports/factor_candidate_combination_exports/selector_v082_crash_remediation_hypotheses/factor_candidate_combos.jsonl
```

Strict robustness output:

```text
reports/factor_candidate_robustness/selector_v082_crash_remediation_hypothesis_gauntlet/
reports/factor_candidate_robustness/selector_v082_crash_remediation_hypothesis_gauntlet/watchlist_robustness_detail.csv
reports/factor_candidate_robustness/selector_v082_crash_remediation_hypothesis_gauntlet/watchlist_robustness_scenario_summary.csv
reports/factor_candidate_robustness/selector_v082_crash_remediation_hypothesis_gauntlet/watchlist_robustness_variant_ranking.csv
reports/factor_candidate_robustness/selector_v082_crash_remediation_hypothesis_gauntlet/watchlist_robustness_report.md
reports/factor_candidate_robustness/selector_v082_crash_remediation_hypothesis_gauntlet/watchlist_robustness_summary.json
```

Focused diagnostic support:

```text
randyslab/factor_candidate_robustness.py
tests/test_factor_candidate_robustness.py
```

Run size:

- Candidates: `7`.
- Mitigation variants per candidate: `7`.
- Stress scenarios: `21`.
- Detail rows: `10045`.
- Scenario summaries: `1029`.
- Final candidate-variant rankings: `49`.

## Compact Ranking

| Candidate | Variant | Verdict | Survival | Mean Sharpe | Worst Sharpe | Mean Max DD | Worst Max DD | Key Blockers |
|---|---|---|---:|---:|---:|---:|---:|---|
| `combo_volume48_ret24_calmvol_funding_calm_mean` | `thr_0p5_long_flat_cap_1p0_none_dd_stop_35_cd_42` | `blocked_pending_new_hypotheses` | 16/21 | 0.5466 | -0.7764 | 0.2815 | 0.7142 | low positive share, weak validation |
| `combo_volume48_volatility48_mean` | `thr_0p5_long_flat_cap_0p5_none_dd_stop_35_cd_42` | `blocked_pending_new_hypotheses` | 16/21 | 0.3424 | -2.1651 | 0.2879 | 0.6005 | BTC/ETH crash, weak validation |
| `combo_volume48_ret24_calmvol_funding_calm_mean` | `thr_0p5_long_flat_cap_1p0_none_none` | `blocked_pending_new_hypotheses` | 15/21 | 0.5593 | -0.7764 | 0.2833 | 0.7159 | validation drawdown, low median |
| `combo_volume48_volatility48_ret24_mean` | `thr_0p5_long_flat_cap_0p5_none_dd_stop_35_cd_42` | `blocked_pending_new_hypotheses` | 15/21 | 0.2183 | -2.5363 | 0.2616 | 0.6266 | BTC/ETH crash, weak validation |

## Stress Read

The strongest new hypothesis improved the original COVID-crash failure:

| Scenario | Prior Top Variant | New Best Defensive Variant |
|---|---:|---:|
| `crash_2020_covid` mean Sharpe | -0.5020 | 0.7769 |
| `crash_2020_covid` worst max DD | 0.2724 | 0.1667 |
| `crash_2020_btc_eth_only` mean Sharpe | -1.5744 | 0.5393 |
| `btc_eth_only_all_windows` mean Sharpe | 0.1952 | 0.7041 |
| `btc_eth_only_validation` mean Sharpe | -0.4825 | 0.2829 |

That improvement does not clear the strict gate. The top new row failed:

- `crash_2020_btc_eth_only`: low positive-row share, because ETH remained negative.
- `very_harsh_fee_slip_funding`: weak validation under combined harsh costs.
- `crash_2021_2022`: low mean and median Sharpe.
- `validation_only`: low mean and median Sharpe.
- `validation_exclude_sol_avax`: low mean and median Sharpe.

## Conservative Memory

Feed back positive memory:

- A trend-plus-calm-volatility guard can reduce the 2020 BTC crash damage that broke the prior watchlist candidate.
- Inverted realized volatility is more useful for crash remediation than simply adding return trend to the original
  realized-volatility-positive blend.
- Funding calmness may help when combined with trend and calm volatility, but it is not strong enough alone.

Feed back negative memory:

- Do not restore a `research_watchlist` label for selector v0.8.2 crash-remediation candidates.
- Do not treat the COVID all-asset pass as sufficient. BTC/ETH-only and ETH-specific weakness still matter.
- Do not treat SOL/AVAX paired exclusion as solved, especially in validation.
- Do not publish these formulas to runtime or use them for portfolio construction.

Next strict research should move beyond this formula family toward new public crypto-native components or explicit
regime diagnostics, with ETH crash behavior and validation robustness as first-class gates.

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
