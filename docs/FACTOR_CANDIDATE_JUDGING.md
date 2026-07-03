# Factor Candidate Judging

RandysLab can strictly judge QuantumRandy research-only factor-candidate exports. This path is for research evidence
only. It is not a runtime publish payload, portfolio construction step, or live execution plan.

## Command

```powershell
python scripts\judge_factor_candidates.py `
  --config configs\strict4h.yaml `
  --candidates ..\QuantumRandy\reports\factor_candidate_exports\selector_v082_milestone_4_60\factor_candidates.jsonl `
  --window all `
  --out reports\factor_candidate_judging\selector_v082_milestone_4_60_btc
```

The judge reads each JSONL candidate, evaluates its formula on the configured OHLCV/funding data, converts the factor
to a simple directional signal, and runs the existing STRICT4H accounting:

- bar `t` signal is exposed from bar `t+1`;
- taker fees, slippage, and funding costs are applied;
- bar-level ledgers are written for each completed candidate;
- metrics and failure reasons are preserved in summary artifacts.

The default signal rule is intentionally simple:

- long when `factor > 0`;
- short when `factor < 0`;
- flat when `factor == 0` or unavailable.

This is a first strict judge, not a final portfolio policy.

## Outputs

- `factor_candidate_judging_summary.json`: machine-readable research artifact with safety flags.
- `factor_candidate_judging_summary.csv`: compact candidate comparison table.
- `factor_candidate_judging_report.md`: human-readable audit report.
- `*_ledger.csv`: per-candidate strict4h ledgers when ledgers are enabled.

## Selector v0.8.2 First Pass

The first all-window pass completed all `7` selector v0.8.2 milestone candidates across BTC, ETH, SOL, BNB, and AVAX
with no formula failures. Under the blunt zero-threshold directional rule, BTC was negative across all seven
candidates, while SOL and AVAX carried the strongest positive evidence.

Best individual asset/candidate rows by Sharpe:

| Candidate | Formula | Sharpe | CAGR | Max DD | Trades |
|---|---|---:|---:|---:|---:|
| `qr_c3ccb8e228` on SOL | `zscore(std(close,48),120)` | 0.9834 | 0.6352 | 0.9556 | 245 |
| `qr_aded180101` on AVAX | `zscore(ema(volume,24),96)` | 0.9243 | 0.5282 | 0.8782 | 481 |
| `qr_295d2e9ee2` on AVAX | `zscore(ema(volume,24),120)` | 0.9110 | 0.5022 | 0.9431 | 429 |
| `qr_3cdea28d1b` on SOL | `zscore(ema(volume,36),144)` | 0.8751 | 0.4363 | 0.8021 | 339 |
| `qr_e23cfc8ae6` on SOL | `zscore(std(close,48),144)` | 0.8339 | 0.3677 | 0.9630 | 251 |

Mean Sharpe by formula across the five local assets:

| Candidate | Formula | Mean Sharpe | Best Sharpe | Worst Sharpe | Positive Assets |
|---|---|---:|---:|---:|---:|
| `qr_e23cfc8ae6` | `zscore(std(close,48),144)` | 0.4826 | 0.8339 | -0.0173 | 4 |
| `qr_c3ccb8e228` | `zscore(std(close,48),120)` | 0.4381 | 0.9834 | -0.0389 | 4 |
| `qr_a2cd9fd69f` | `zscore(ema(volume,48),120)` | 0.3230 | 0.7644 | -0.1328 | 3 |
| `qr_3cdea28d1b` | `zscore(ema(volume,36),144)` | 0.3089 | 0.8751 | -0.2967 | 4 |
| `qr_41f6f43247` | `zscore(std(close,36),144)` | 0.2702 | 0.7080 | -0.0612 | 3 |
| `qr_aded180101` | `zscore(ema(volume,24),96)` | 0.2635 | 0.9243 | -0.3188 | 4 |
| `qr_295d2e9ee2` | `zscore(ema(volume,24),120)` | 0.2307 | 0.9110 | -0.5640 | 4 |

Interpretation: the first strict pass is mixed, not admission evidence. Positive realized-volatility state looks more
stable than smoothed participation on this direct-sign rule, but drawdowns remain large and BTC is weak. These formulas
need per-window review, threshold sensitivity, and factor-combination research before any downstream use.

## Selector v0.8.2 Window/Threshold Sensitivity

The first sensitivity pass swept the `7` selector v0.8.2 standalone candidates across:

- Assets: BTC, ETH, SOL, BNB, AVAX.
- Windows: training, validation, blind.
- Thresholds: `0.0`, `0.5`, `1.0`.

All `315` standalone rows completed with no formula failures.

Standalone stability summary:

| Candidate | Formula | Mean Sharpe | Median Sharpe | Worst Sharpe | Positive Rows | Completed Rows | Mean Max DD |
|---|---|---:|---:|---:|---:|---:|---:|
| `qr_a2cd9fd69f` | `zscore(ema(volume,48),120)` | 0.4083 | 0.4156 | -1.5848 | 30 | 45 | 0.5814 |
| `qr_e23cfc8ae6` | `zscore(std(close,48),144)` | 0.3558 | 0.4822 | -1.2775 | 29 | 45 | 0.5848 |
| `qr_295d2e9ee2` | `zscore(ema(volume,24),120)` | 0.3510 | 0.3317 | -1.7473 | 29 | 45 | 0.5817 |
| `qr_3cdea28d1b` | `zscore(ema(volume,36),144)` | 0.3148 | 0.4549 | -1.4396 | 28 | 45 | 0.5833 |
| `qr_aded180101` | `zscore(ema(volume,24),96)` | 0.3048 | 0.2575 | -1.3770 | 28 | 45 | 0.5941 |
| `qr_c3ccb8e228` | `zscore(std(close,48),120)` | 0.1597 | 0.3832 | -2.1883 | 29 | 45 | 0.6039 |
| `qr_41f6f43247` | `zscore(std(close,36),144)` | 0.1411 | 0.3685 | -1.7005 | 28 | 45 | 0.6173 |

Window/threshold read:

- Blind rows were strongest on average: threshold `0.5` had mean Sharpe `0.7981`.
- Validation improved as threshold rose: threshold `1.0` had mean Sharpe `0.1927`.
- Training was dominated by SOL/AVAX high-Sharpe rows and should not be mistaken for robust admission evidence.

## Simple Combination Research

The judge also supports research-only `component_formulas` with an equal-weight mean. This is a factor-combination
diagnostic only, not RandyPortfolio and not runtime publishing.

The first simple-combo sweep tested four equal-weight blends across the same assets, windows, and thresholds. All
`180` combo rows completed with no formula failures.

Combo stability summary:

| Candidate | Formula | Mean Sharpe | Median Sharpe | Worst Sharpe | Positive Rows | Completed Rows | Mean Max DD |
|---|---|---:|---:|---:|---:|---:|---:|
| `combo_volume48_volatility48_mean` | `mean(ema(volume,48) zscore, std(close,48) zscore)` | 0.5412 | 0.7308 | -1.0577 | 35 | 45 | 0.5336 |
| `combo_volume24_volatility48_mean` | `mean(ema(volume,24) zscore, std(close,48) zscore)` | 0.3339 | 0.4208 | -1.8016 | 32 | 45 | 0.5614 |
| `combo_volume_smooth_mean` | `mean(ema(volume,48) zscore, ema(volume,36) zscore)` | 0.3327 | 0.4821 | -1.4633 | 29 | 45 | 0.5862 |
| `combo_volatility_48_mean` | `mean(std(close,48) zscore 120/144)` | 0.3266 | 0.3351 | -1.3635 | 30 | 45 | 0.5840 |

Interpretation: the first combo read favors a participation-plus-realized-volatility blend. It improved mean Sharpe,
median Sharpe, positive-row count, worst Sharpe, and mean max drawdown versus the standalone formulas. This is still
research evidence only; large drawdowns and window dependence remain.

## Conservative Review

The conservative review layer applies explicit blockers to sensitivity evidence:

- low mean or median Sharpe;
- weak validation or blind-window mean Sharpe;
- too few positive assets or concentrated positive evidence;
- high mean drawdown;
- extreme per-row drawdown.

Review command:

```powershell
python scripts\review_factor_candidate_sensitivity.py `
  --detail-csv reports\factor_candidate_sensitivity\selector_v082_milestone_4_60_multi_asset_windows_thresholds\factor_candidate_sensitivity_detail.csv `
  --out reports\factor_candidate_review\selector_v082_milestone_4_60_multi_asset_windows_thresholds
```

Standalone review result: all `7` standalone candidates were `blocked_by_conservative_rules`.

Primary blockers:

- high mean drawdown;
- extreme row drawdown;
- weak validation for several participation candidates;
- low mean Sharpe for the weaker realized-volatility variants.

Simple-combo review result: all `4` combos were also `blocked_by_conservative_rules`.

The strongest combo, `combo_volume48_volatility48_mean`, remained the best diagnostic candidate but was blocked by:

- `high_mean_drawdown`;
- `extreme_row_drawdown`.

Interpretation: the strict review says the current formulas and simple combos are useful research evidence, especially
for participation-plus-realized-volatility memory, but none should be admitted or promoted. The next research step is
to reduce drawdown and window instability before adding stronger factor-factory memory.

## Long/Flat Drawdown Probe

The judge supports research-only signal modes:

- `long_short`: long positive factor values and short negative factor values.
- `long_flat`: long positive factor values and flat otherwise.
- `short_flat`: short negative factor values and flat otherwise.

The first drawdown-reduction probe reran the simple combos across `long_short` and `long_flat`.

Aggregate mode comparison:

| Signal Mode | Mean Sharpe | Median Sharpe | Worst Sharpe | Mean Max DD | Worst Max DD | Positive Rows |
|---|---:|---:|---:|---:|---:|---:|
| `long_flat` | 0.6101 | 0.7385 | -1.2124 | 0.4808 | 0.9249 | 151/180 |
| `long_short` | 0.3836 | 0.5220 | -1.8016 | 0.5663 | 0.9572 | 126/180 |

Best long/flat combo:

| Candidate | Mean Sharpe | Median Sharpe | Worst Sharpe | Mean Max DD | Worst Max DD | Positive Rows |
|---|---:|---:|---:|---:|---:|---:|
| `combo_volume48_volatility48_mean` | 0.6969 | 0.7924 | -0.8382 | 0.4626 | 0.9239 | 37/45 |

The combined-mode conservative review still blocked all `4` combos. The strongest combo improved materially under
`long_flat`, but remained blocked by `high_mean_drawdown` and `extreme_row_drawdown`.

## Drawdown-Reduction Variant Pass

The scoped drawdown-reduction pass tested only the strongest participation-plus-realized-volatility combo:

```text
mean(zscore(ema(volume,48),120), zscore(std(close,48),144))
```

The pass was long-flat only and swept BTC, ETH, SOL, BNB, and AVAX across training, validation, and blind windows;
thresholds `0.5`, `1.0`, `1.5`, and `2.0`; exposure caps `1.0`, `0.75`, `0.5`, and `0.25`; and simple realized-vol
caps using `zscore(std(close,48),144) <= 1.5` or `<= 1.0`.

Artifacts:

```text
reports/factor_candidate_sensitivity/selector_v082_combo_drawdown_reduction_pass/
reports/factor_candidate_review/selector_v082_combo_drawdown_reduction_pass/
docs/SELECTOR_V082_DRAWDOWN_REDUCTION_PASS.md
```

Result: `11/48` variants became `research_watchlist`; `37/48` remained `blocked_by_conservative_rules`. The best
drawdown-balanced variant was `thr_0p5_long_flat_cap_0p5_none`, with mean Sharpe `0.6378`, median Sharpe `0.5469`,
mean max drawdown `0.2785`, worst max drawdown `0.6975`, and positive rows `14/15`.

The full-exposure no-filter control at threshold `0.5` still failed for
`high_mean_drawdown|extreme_row_drawdown`. Lower exposure caps and the `rvz_lte_1p5` volatility cap were the clearest
drawdown reducers. The stricter `rvz_lte_1p0` cap often reduced drawdown further but usually failed on weak median
Sharpe, low positive-row share, or weak validation/blind windows.

Interpretation: some variants now merit research-watchlist tracking under conservative rules, but this is not factor
admission, runtime publishing, portfolio construction, or live execution approval. BTC weakness and validation-window
fragility remain material caveats.

## Selector v0.8.2 Deep Drawdown Mitigation

A deeper research-only mitigation campaign extended the scoped pass for the same participation-plus-realized-volatility
combo. It added:

- research drawdown-stop/cooldown variants;
- a strict `4320` row multi-asset, multi-window, multi-threshold mitigation sweep;
- a `288` variant conservative review;
- reproducible drawdown episode diagnostics with explicit failure labels.

Artifacts:

```text
reports/factor_candidate_sensitivity/selector_v082_combo_drawdown_deep_mitigation/
reports/factor_candidate_review/selector_v082_combo_drawdown_deep_mitigation/
reports/factor_candidate_drawdown_diagnostics/selector_v082_combo_drawdown_deep_mitigation_core/
docs/SELECTOR_V082_DRAWDOWN_DEEP_MITIGATION_REPORT.md
```

Result: `44/288` variants reached `research_watchlist`; `244/288` remained
`blocked_by_conservative_rules`. The best current variant was
`thr_0p5_long_flat_cap_0p5_none_dd_stop_35_cd_42`, with mean Sharpe `0.6419`, median Sharpe `0.6415`, mean max
drawdown `0.2683`, and worst max drawdown `0.5888`.

The episode audit found the worst drawdown at SOLUSDT validation from `2021-09-09 00:00 UTC` to
`2022-11-09 16:00 UTC` for the full-exposure long-flat control. The preserved labels were
`sol_avax_concentration|validation_weakness|crash_period_drawdown|extreme_row_drawdown`.

Interpretation: exposure caps remain the cleanest mitigation, moderate realized-volatility filters help, and drawdown
cooldowns are useful only when paired with exposure control. Full-exposure variants remain blocked, and BTC weakness,
SOL/AVAX concentration, validation weakness, and crash-period drawdowns should feed back into QuantumRandy
factor-factory memory.

## Selector v0.8.2 Crash-Remediation Hypotheses

A follow-up strict robustness gauntlet tested new research-only crash-remediation hypotheses after the prior watchlist
family was downgraded to `blocked pending new hypotheses`.

Artifacts:

```text
reports/factor_candidate_combination_exports/selector_v082_crash_remediation_hypotheses/
reports/factor_candidate_robustness/selector_v082_crash_remediation_hypothesis_gauntlet/
docs/SELECTOR_V082_CRASH_REMEDIATION_HYPOTHESIS_REPORT.md
```

The pass tested `7` candidate combos, `7` mitigation variants per candidate, and `21` stress scenarios. It produced
`10045` strict detail rows, `1029` scenario summaries, and `49` final candidate-variant rankings.

Result: all `49` final rows are `blocked_pending_new_hypotheses`. No crash-remediation hypothesis earns
`research_watchlist`.

The best new diagnostic row was
`combo_volume48_ret24_calmvol_funding_calm_mean` with
`thr_0p5_long_flat_cap_1p0_none_dd_stop_35_cd_42`. It survived `16/21` stresses and improved the prior 2020 COVID
weakness, but it still failed the stricter BTC/ETH-only COVID slice on low positive-row share and failed validation
stress slices on low mean/median Sharpe.

Interpretation: trend plus calm-volatility plus funding-calm components are useful factor-factory memory, but only as
diagnostic memory. Selector v0.8.2 remains blocked pending genuinely new hypotheses, especially for ETH crash behavior
and validation robustness.

## Boundary

- No live exchange orders.
- No exchange trading credentials.
- No QuantumRandy runtime updates.
- No automatic factor admission.
- Failure reasons and weak metrics should be treated as first-class evidence.
