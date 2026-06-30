# RandysLab Role In The Randy Quant Stack

RandysLab is the baseline and data-utility companion to QuantumRandy.

## Responsibilities

- Maintain historical OHLCV and funding-rate datasets.
- Run strict 4h baseline backtests with explicit execution delay.
- Provide traditional strategy baselines for comparison.
- Run parameter search and cost-scenario robustness checks.
- Produce audit-friendly dashboards and reports.

## Non-Responsibilities

- No live exchange order placement.
- No LLM/MCTS factor mining.
- No automatic strategy publishing into QuantumRandy runtime.
- No private trading API credentials.

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

## Package Name

The canonical package name is `randyslab`.
