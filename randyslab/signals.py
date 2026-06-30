from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_THETA = {
    "strategy": "randyslab",
    "ema_fast": 12,
    "ema_slow": 26,
    "ema_threshold": 0.003,
    "ma_fast": 20,
    "ma_slow": 60,
    "bb_period": 20,
    "bb_dev": 2.0,
    "min_hold_bars": 1,
    "cooldown_hours": 0,
    "atr_period": 0,
    "atr_k_sl": 0.0,
    "atr_k_tp": 0.0,
    "max_exposure_abs": 1.0,
    "funding_bias_thr_bps": 0.0,
    "funding_bias_k_thr_per_bps": 0.0,
    "w_mom": 0.75,
}


def make_signal(ohlcv: pd.DataFrame, funding_rate: pd.Series, theta: dict) -> pd.Series:
    p = DEFAULT_THETA | theta
    strategy = str(p.get("strategy", "randyslab")).lower()
    if strategy in {"ma_cross", "dual_ma", "double_ma"}:
        raw = ma_cross_signal(ohlcv, p)
    elif strategy in {"bb_breakout", "bollinger_breakout"}:
        raw = bollinger_breakout_signal(ohlcv, p)
    else:
        raw = randyslab_signal(ohlcv, funding_rate, p)
    return apply_hold_and_cooldown(raw, int(p["min_hold_bars"]), int(p["cooldown_hours"] // 4))


def randyslab_signal(ohlcv: pd.DataFrame, funding_rate: pd.Series, p: dict) -> pd.Series:
    fast = int(p["ema_fast"])
    slow = int(p["ema_slow"])
    if fast >= slow:
        raise ValueError("ema_fast must be shorter than ema_slow")

    close = ohlcv["close"].astype(float)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    spread = (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)
    scale = max(float(p["ema_threshold"]), 1e-9)
    momentum = 0.5 * (1.0 + np.tanh(spread / scale))

    bb_period = int(p["bb_period"])
    bb_dev = float(p["bb_dev"])
    mean = close.rolling(bb_period, min_periods=bb_period).mean()
    std = close.rolling(bb_period, min_periods=bb_period).std(ddof=0)
    z = (close - mean) / std.replace(0, np.nan)
    anomaly = (np.abs(z) / max(bb_dev, 1e-9)).clip(0, 1).fillna(0)

    w_mom = float(p["w_mom"])
    composite = (w_mom * momentum + (1.0 - w_mom) * anomaly).fillna(0.5)

    base_thr = float(p["ema_threshold"])
    bias_thr = float(p["funding_bias_thr_bps"]) / 10_000.0
    k = float(p["funding_bias_k_thr_per_bps"])
    adverse = (funding_rate.abs() - bias_thr).clip(lower=0.0)
    dynamic_thr = base_thr + k * adverse

    long_cut = 0.5 + dynamic_thr
    short_cut = 0.5 - dynamic_thr
    raw = pd.Series(0, index=ohlcv.index, dtype="int8")
    raw[composite >= long_cut] = 1
    raw[composite <= short_cut] = -1
    return raw


def ma_cross_signal(ohlcv: pd.DataFrame, p: dict) -> pd.Series:
    fast = int(p.get("ma_fast", p.get("ema_fast", 20)))
    slow = int(p.get("ma_slow", p.get("ema_slow", 60)))
    if fast >= slow:
        raise ValueError("ma_fast must be shorter than ma_slow")
    close = ohlcv["close"].astype(float)
    ma_fast = close.rolling(fast, min_periods=fast).mean()
    ma_slow = close.rolling(slow, min_periods=slow).mean()
    raw = pd.Series(0, index=ohlcv.index, dtype="int8")
    raw[ma_fast > ma_slow] = 1
    raw[ma_fast < ma_slow] = -1
    return raw


def bollinger_breakout_signal(ohlcv: pd.DataFrame, p: dict) -> pd.Series:
    period = int(p["bb_period"])
    dev = float(p["bb_dev"])
    close = ohlcv["close"].astype(float)
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + dev * std
    lower = mid - dev * std
    raw = pd.Series(0, index=ohlcv.index, dtype="int8")
    raw[close > upper] = 1
    raw[close < lower] = -1
    return raw.ffill().fillna(0).astype("int8")


def apply_hold_and_cooldown(raw: pd.Series, min_hold_bars: int, cooldown_bars: int) -> pd.Series:
    signal = []
    current = 0
    held = 0
    cooldown = 0
    for wanted in raw.astype(int).to_numpy():
        if cooldown > 0:
            wanted = 0
            cooldown -= 1
        if wanted != current and held < max(min_hold_bars, 1):
            wanted = current
        if wanted != current:
            if wanted == 0 and cooldown_bars > 0:
                cooldown = cooldown_bars
            current = wanted
            held = 0
        signal.append(current)
        held += 1
    return pd.Series(signal, index=raw.index, dtype="int8")
