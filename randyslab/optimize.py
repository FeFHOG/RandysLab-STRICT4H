from __future__ import annotations

import random
import warnings

import pandas as pd

from .config import CostProfile
from .engine import run_strict4h
from .metrics import cagr, summarize


def sample_theta(rng: random.Random) -> dict:
    fast = rng.randint(6, 32)
    slow = rng.randint(max(20, fast + 1), 96)
    return {
        "ema_fast": fast,
        "ema_slow": slow,
        "ema_threshold": rng.uniform(0.0005, 0.0060),
        "bb_period": rng.randint(10, 30),
        "bb_dev": rng.uniform(1.0, 2.5),
        "min_hold_bars": rng.randint(1, 6),
        "cooldown_hours": rng.choice([0, 4, 8]),
        "atr_period": 0,
        "atr_k_sl": 0.0,
        "atr_k_tp": 0.0,
        "max_exposure_abs": rng.uniform(0.25, 2.0),
        "funding_bias_thr_bps": rng.uniform(0.0, 10.0),
        "funding_bias_k_thr_per_bps": rng.uniform(0.0, 0.005),
    }


def random_search(ohlcv, funding, costs: CostProfile, n_trials: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for trial in range(n_trials):
        theta = sample_theta(rng)
        ledger = run_strict4h(ohlcv, funding, theta, costs)
        rows.append({"trial": trial, "score": cagr(ledger["r_net"]), "theta": theta, **summarize(ledger)})
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def _theta_from_trial(trial) -> dict:
    fast = trial.suggest_int("ema_fast", 6, 32)
    slow = trial.suggest_int("ema_slow", max(20, fast + 1), 96)
    return {
        "ema_fast": fast,
        "ema_slow": slow,
        "ema_threshold": trial.suggest_float("ema_threshold", 0.0005, 0.0060),
        "bb_period": trial.suggest_int("bb_period", 10, 30),
        "bb_dev": trial.suggest_float("bb_dev", 1.0, 2.5),
        "min_hold_bars": trial.suggest_int("min_hold_bars", 1, 6),
        "cooldown_hours": trial.suggest_categorical("cooldown_hours", [0, 4, 8]),
        "atr_period": 0,
        "atr_k_sl": 0.0,
        "atr_k_tp": 0.0,
        "max_exposure_abs": trial.suggest_float("max_exposure_abs", 0.25, 2.0),
        "funding_bias_thr_bps": trial.suggest_float("funding_bias_thr_bps", 0.0, 10.0),
        "funding_bias_k_thr_per_bps": trial.suggest_float("funding_bias_k_thr_per_bps", 0.0, 0.005),
    }


def tpe_search(ohlcv, funding, costs: CostProfile, n_trials: int, seed: int) -> pd.DataFrame:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows = []

    def objective(trial):
        theta = _theta_from_trial(trial)
        ledger = run_strict4h(ohlcv, funding, theta, costs)
        score = cagr(ledger["r_net"])
        rows.append({"trial": trial.number, "score": score, "theta": theta, **summarize(ledger)})
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def search(ohlcv, funding, costs: CostProfile, n_trials: int, seed: int, method: str = "tpe") -> pd.DataFrame:
    """Run Stage I search; the ``optimizer`` column records which method actually produced the trials."""
    if method == "random":
        return random_search(ohlcv, funding, costs, n_trials, seed).assign(optimizer="random")
    try:
        return tpe_search(ohlcv, funding, costs, n_trials, seed).assign(optimizer="tpe")
    except ModuleNotFoundError as exc:
        if exc.name != "optuna":
            raise
        warnings.warn(
            "optuna is not installed; Stage I fell back to random search (install the 'tuning' extra)",
            RuntimeWarning,
            stacklevel=2,
        )
        return random_search(ohlcv, funding, costs, n_trials, seed).assign(optimizer="random_fallback_optuna_missing")
