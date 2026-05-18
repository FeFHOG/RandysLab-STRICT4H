# AutoQuant STRICT4H

一个面向加密货币永续合约的 STRICT4H 回测与策略筛选实验室。

这个项目参考论文 `AutoQuant: An Auditable Expert-System Framework for Execution-Constrained Auto-Tuning in Cryptocurrency Perpetual Futures` 的公开描述实现。它不是论文作者的官方代码，也不是实盘交易系统。它的用途是帮助你学习并复现实验流程：严格执行语义、资金费率对齐、交易成本、参数搜索、双阶段筛选和可视化审计。

## 已实现

- 4小时 OHLCV 与 8小时 funding 输入
- STRICT `t+1` 执行：bar `t` 收盘生成信号，从 `t+1` 开始持仓
- funding no-look-ahead：资金费率只向前填充，不使用未来值
- 成本拆分：手续费、滑点、资金费率
- Stage I 参数搜索：默认 Optuna TPE，未安装时退回随机搜索
- Stage II 成本情景筛选：默认 `3 x 3` fee/funding grid
- 本地 HTML 仪表盘：Binance 深色风格，K线、交易点、权益曲线、回撤曲线

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 数据格式

默认配置读取：

- `data/BTCUSDT_4h.csv`: `timestamp,open,high,low,close,volume`
- `data/BTCUSDT_funding.csv`: `timestamp,funding_rate`

时间戳使用 UTC。历史回测通常不需要交易所 API key。

## 拉取 Binance U 本位永续数据

```powershell
python scripts\fetch_binance.py --symbol BTC/USDT:USDT --file-prefix BTCUSDT --start 2019-09-08 --end 2025-11-24
```

如果需要本地代理：

```powershell
python scripts\fetch_binance.py --symbol BTC/USDT:USDT --file-prefix BTCUSDT --start 2019-09-08 --end 2025-11-24 --proxy http://127.0.0.1:7890
```

## 运行回测流程

```powershell
python scripts\run_backtest.py --window training --out reports\btc_real_smoke
python scripts\run_stage1.py
python scripts\run_stage2.py
python scripts\make_dashboard.py
```

然后打开：

```text
reports/dashboard.html
```

## 输出说明

- `ledger.csv`: 每根 bar 的 signal、exposure、收益和成本拆分
- `summary.json`: CAGR、月几何收益、Sharpe、最大回撤、交易次数等
- `trials.csv`: Stage I 参数搜索结果
- `robust_summary.csv`: Stage II 成本情景聚合结果
- `dashboard.html`: 本地可视化面板

## 两个基础策略

项目内置了两个新手友好的基础策略：

- `configs/strategies/ma_cross.json`: 双均线交叉
- `configs/strategies/bb_breakout.json`: 布林带突破

一次性运行并生成两个仪表盘：

```powershell
python scripts\run_basic_strategies.py
```

输出：

- `reports/dashboard_ma_cross.html`
- `reports/dashboard_bb_breakout.html`

## 进一步调教方向

- 把 Stage I 目标从“只追收益”改成“收益 - 回撤惩罚 - 换手惩罚”
- 加 ATR 止损止盈、最大回撤降仓、连续亏损冷却
- 补全论文提到的 anomaly / breakout / risk knobs
- 用 ETH、SOL、AVAX 做跨品种验证
- 加纸交易日志和回测账本一致性检查
