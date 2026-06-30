# RandysLab STRICT4H

RandysLab is the baseline research and data utility layer for the Randy quant stack. It focuses on strict 4-hour
cryptocurrency perpetual-futures backtesting, traditional strategy baselines, parameter search, robustness screening,
and audit-friendly dashboards.

This project was formerly named **AutoQuant STRICT4H**. The Python package is now `randyslab`; a thin `autoquant`
compatibility layer remains for older imports.

RandysLab is not a live trading system. It does not place orders and does not require exchange trading credentials for
historical research.

## Role In The Stack

RandysLab is the control group for QuantumRandy:

- fetch and maintain Binance OHLCV/funding datasets;
- run strict 4h baseline strategy backtests;
- provide traditional strategy comparisons for QuantumRandy factors;
- keep execution-cost assumptions auditable;
- produce dashboards and summary reports for baseline experiments.

QuantumRandy remains responsible for LLM/MCTS factor mining, formula validation, paper runtime, and factor publishing.

## Implemented Features

- 4h OHLCV plus 8h funding-rate inputs.
- Strict `t+1` execution: signals from bar `t` are executed from bar `t+1`.
- Funding-rate alignment with forward-fill only; no future funding observations are used.
- Cost decomposition: taker fee, slippage, and funding cost.
- Stage I parameter search with Optuna TPE when available, random fallback otherwise.
- Stage II cost-scenario robustness screening.
- Local HTML dashboards with equity, drawdown, K-line, and trade markers.
- Beginner-friendly MA crossover and Bollinger breakout baseline strategies.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Format

Default config reads:

- `data/BTCUSDT_4h.csv`: `timestamp,open,high,low,close,volume`
- `data/BTCUSDT_funding.csv`: `timestamp,funding_rate`

Timestamps should be UTC.

## Fetch Binance USD-M Perpetual Data

```powershell
python scripts\fetch_binance.py --symbol BTC/USDT:USDT --file-prefix BTCUSDT --start 2019-09-08 --end 2025-11-24
```

With a local proxy:

```powershell
python scripts\fetch_binance.py --symbol BTC/USDT:USDT --file-prefix BTCUSDT --start 2019-09-08 --end 2025-11-24 --proxy http://127.0.0.1:7890
```

## Run The Baseline Workflow

```powershell
python scripts\run_backtest.py --window training --out reports\btc_real_smoke
python scripts\run_stage1.py
python scripts\run_stage2.py
python scripts\make_dashboard.py
```

Open:

```text
reports/dashboard.html
```

## Outputs

- `ledger.csv`: bar-level signal, exposure, returns, and cost decomposition.
- `summary.json`: CAGR, monthly geometric return, Sharpe, max drawdown, and trade count.
- `trials.csv`: Stage I parameter-search results.
- `robust_summary.csv`: Stage II scenario aggregation.
- `dashboard.html`: local visual audit dashboard.

## Built-In Baseline Strategies

Two simple baseline strategies are included:

- `configs/strategies/ma_cross.json`: moving-average crossover.
- `configs/strategies/bb_breakout.json`: Bollinger breakout.

Run both and generate dashboards:

```powershell
python scripts\run_basic_strategies.py
```

Outputs:

- `reports/dashboard_ma_cross.html`
- `reports/dashboard_bb_breakout.html`

## Next Improvements

- Add a baseline export format that QuantumRandy can ingest as a paper-runtime comparison.
- Add cross-asset baseline configs for ETH, SOL, BNB, and AVAX.
- Add portfolio-level baseline reports for fixed baseline blends.
- Add stronger dashboard summaries for server-side paper observation.
- Keep old `autoquant` imports working until downstream scripts have migrated.
