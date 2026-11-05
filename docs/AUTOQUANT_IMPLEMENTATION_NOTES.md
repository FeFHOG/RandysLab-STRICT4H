# AutoQuant Implementation Notes

Where RandysLab deliberately differs from the AutoQuant paper (arXiv 2512.22476, source in `arXiv-2512.22476v1/`), and
what the paper describes that is not implemented. Changes dated 2026-11-05 come from the Phase 1 evaluation-spine work
(QuantumRandy `docs/PHASE1_EVALUATION_SPINE.md`).

## Deliberate Differences

| Area | Paper | RandysLab | Why |
|---|---|---|---|
| Funding cost | 8h rate carried forward to the 4h grid | Each settlement is charged to the bar holding the position at that instant (`randyslab/accounting.py`) | Binance moves symbols to 4h/2h/1h settlement. On SOL (98 settlements at 2h in Nov 2022) the carried-forward method was off by about 420 bps/yr for a constant long. |
| Funding feature | Carried-forward 8h rate | Last rate settled at or before the bar open, rescaled to an 8h equivalent; stamps floored to the minute | Keeps the feature comparable across interval changes. 43% of Binance stamps carry millisecond jitter that pushed them one bar late. |
| Funding bias gate | Eq. `funding-threshold`: κ per 1.0 decimal of funding, κ ∈ [0, 0.005], symmetric `abs(fr)` | κ per basis point (as the parameter name `funding_bias_k_thr_per_bps` says) and directional: long entries are gated by positive funding, short entries by negative | With the paper's units the largest possible threshold change is 1.5e-5 against thresholds of at least 5e-4, so the gate never changes a signal. The paper's text describes the short side with "the sign of fr reversed in the adverse-direction case". |
| Stage II window | Training + Validation | Training + Validation (`--window train_val`, default) | Restored on 2026-11-05. Commit `6533fe1` (2026-06-30, "Rename AutoQuant to RandysLab") had switched it to the training window only; `--window training` reproduces that. |
| Position size | `max_exposure_abs` as an exposure cap (0–5) | Position = signal × `max_exposure_abs` | Signals are -1/0/1, so clipping made every value above 1 behave like 1. |
| Bollinger breakout baseline | Not in the paper | Holds the last breakout direction until the opposite band breaks | The previous `ffill` ran on zeros and never held a position. |

Stage I/II results produced before 2026-11-05 used an inert funding gate, the training-only Stage II window, and the
carried-forward funding cost. Treat them as superseded.

## Not Implemented

- Rolling held-out windows (2000 bars, step 500) and block bootstrap.
- Stress ladders: 2× and 3× costs, funding off, zero-cost and fee-only runs; t+0 ablation; TPE versus random search.
- ATR exits (parameters exist but are unused), exposure smoothing, the tail-quantile anomaly, the 50k notional cap.
- Data validation rules: treating any gap as fatal, OHLC consistency checks, the 1 bp funding fallback.
- Multi-asset configs, windows and cost profiles for ETH, SOL and AVAX.

Selection statistics the paper reports (Deflated Sharpe, PBO via CSCV) are added as reusable functions in Phase 1 rather
than as paper-specific experiment scripts.
