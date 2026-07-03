# RandysLab Role In The Randy Quant Stack

RandysLab is the baseline and data-utility companion to QuantumRandy.

## Responsibilities

- Maintain historical OHLCV and funding-rate datasets.
- Run strict 4h baseline backtests with explicit execution delay.
- Judge QuantumRandy research-only factor-candidate exports with strict 4h execution, costs, and failure reasons.
- Consume declared scope metadata from candidate exports and apply scope-aware conservative reviews.
- Provide traditional strategy baselines for comparison.
- Run parameter search and cost-scenario robustness checks.
- Produce audit-friendly dashboards and reports.

## Non-Responsibilities

- No live exchange order placement.
- No LLM/MCTS factor mining.
- No automatic strategy publishing into QuantumRandy runtime.
- No private trading API credentials.
- No automatic factor admission from candidate-judging results.
- No factor scope invention; scope metadata must come from the producing research artifact.
- No portfolio scheduling or regime-allocation ownership.

## Integration With QuantumRandy

QuantumRandy should treat RandysLab outputs as baseline/control artifacts:

- compare QuantumRandy factors against RandysLab baseline strategies;
- reuse downloaded OHLCV/funding files where appropriate;
- use RandysLab strict execution assumptions as a sanity reference;
- eventually export baseline paper strategies into the QuantumRandy runtime observation layer.

Baseline exports are produced with:

```powershell
python scripts\export_baselines.py --window all --out reports\quantumrandy_baselines
```

The export writes `baseline_summary.json`, `baseline_summary.csv`, `baseline_report.md`, and optional per-strategy
ledgers. These artifacts are a control group for runtime/paper reports. They are not QuantumRandy runtime publish
payloads and must not bypass the manual factor publishing or review flow.

Research-only QuantumRandy factor candidates are judged with:

```powershell
python scripts\judge_factor_candidates.py --config configs\strict4h.yaml --candidates path\to\factor_candidates.jsonl --window all
```

The judge writes summary JSON/CSV, a Markdown report, and optional per-candidate ledgers. These are strict research
artifacts with failure reasons, not factor-admission decisions or runtime payloads.

Sensitivity sweeps are produced with:

```powershell
python scripts\sweep_factor_candidates.py --config configs\strict4h.yaml --candidates path\to\factor_candidates.jsonl
```

The sweep aggregates strict judge runs across assets, windows, and thresholds. It may also judge research-only
component-factor combinations. These outputs are diagnostic evidence only and must not become automatic runtime or
portfolio actions.

Conservative reviews are produced with:

```powershell
python scripts\review_factor_candidate_sensitivity.py --detail-csv path\to\factor_candidate_sensitivity_detail.csv
```

The review applies explicit blockers for weak windows, asset concentration, high drawdown, and extreme row drawdown.
Blocked candidates may still be useful factor-factory research evidence, but they are not admissible outputs.

Declared-scope reviews may use `scope_mode="declared"` in code paths that intentionally evaluate a scoped candidate
such as `BTCUSDT_4h`. In that mode, RandysLab still enforces window, Sharpe, positive-row, and drawdown gates, but does
not require a single-asset factor to satisfy universal multi-asset positive-asset counts. Review CSV and Markdown
artifacts preserve the producing artifact's `intended_scope`, `applicability_hypothesis`, and `out_of_scope_policy`.
Out-of-scope asset rows should remain diagnostic labels and memory, not silent deployment requirements.

CLI example:

```powershell
python scripts\review_factor_candidate_sensitivity.py `
  --detail-csv path\to\factor_candidate_sensitivity_detail.csv `
  --scope-mode declared
```

RandysLab supports the strict formula profile needed for current QuantumRandy scoped-regime inputs: `zscore`, `ema`,
`std`, `sma`, `delta`, `ret`, `neg`, `add`, `sub`, `mul`, `div`, `abs`, `log`, `sqrt`, `sign`, `corr`, `min`, `max`,
and `rsi`. New base fields such as open interest, basis, liquidation imbalance, taker imbalance, or depth proxies must
first pass a data-readiness audit before being accepted as judge inputs.

## Package Name

The canonical package name is `randyslab`.
