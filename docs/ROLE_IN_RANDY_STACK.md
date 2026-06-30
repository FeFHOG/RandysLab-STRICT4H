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

## Naming Migration

The project was formerly named AutoQuant STRICT4H. The canonical package name is now `randyslab`.

Temporary compatibility imports remain under `autoquant` so older scripts can migrate gradually:

```python
from autoquant.engine import run_strict4h  # still works
from randyslab.engine import run_strict4h  # preferred
```

New code should use `randyslab`.
