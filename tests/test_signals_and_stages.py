from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import randyslab.optimize as optimize
from randyslab.config import CostProfile, Windows
from randyslab.screening import stage2_window
from randyslab.signals import DEFAULT_THETA, bollinger_breakout_signal, randyslab_signal


def _ohlcv(close: np.ndarray) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(close), freq="4h", tz="UTC")
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=index)


def _random_walk(n: int = 2000, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return _ohlcv(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))))


def test_stage2_screens_training_plus_validation_by_default():
    windows = Windows(training_start="2019-09-08", training_end="2021-01-01", validation_end="2023-01-01")
    assert stage2_window(windows) == ("2019-09-08", "2023-01-01")
    assert stage2_window(windows, "training") == ("2019-09-08", "2021-01-01")
    with pytest.raises(ValueError):
        stage2_window(windows, "blind")


def test_funding_gate_is_measured_in_bps_and_only_blocks_the_paying_side():
    ohlcv = _random_walk()
    funding = pd.Series(0.0001, index=ohlcv.index)  # +1 bp per 8h: longs pay
    base = DEFAULT_THETA | {"ema_threshold": 0.002, "w_mom": 1.0}
    off = randyslab_signal(ohlcv, funding, base | {"funding_bias_k_thr_per_bps": 0.0})
    on = randyslab_signal(ohlcv, funding, base | {"funding_bias_k_thr_per_bps": 0.005})

    assert int((on == 1).sum()) < int((off == 1).sum())
    assert int((on == -1).sum()) == int((off == -1).sum())

    receiving = randyslab_signal(ohlcv, -funding, base | {"funding_bias_k_thr_per_bps": 0.005})
    assert int((receiving == 1).sum()) == int((off == 1).sum())


def test_funding_gate_respects_activation_threshold():
    ohlcv = _random_walk()
    funding = pd.Series(0.0001, index=ohlcv.index)
    base = DEFAULT_THETA | {"ema_threshold": 0.002, "w_mom": 1.0, "funding_bias_k_thr_per_bps": 0.005}
    below = randyslab_signal(ohlcv, funding, base | {"funding_bias_thr_bps": 2.0})
    off = randyslab_signal(ohlcv, funding, base | {"funding_bias_k_thr_per_bps": 0.0})
    assert below.equals(off)


def test_bollinger_breakout_holds_until_the_opposite_band_breaks():
    flat = np.full(30, 100.0)
    path = np.concatenate([flat, [110.0], np.linspace(110.0, 112.0, 15), [80.0], np.full(5, 80.0)])
    signal = bollinger_breakout_signal(_ohlcv(path), DEFAULT_THETA | {"bb_period": 20, "bb_dev": 2.0})
    breakout_up = 30
    breakout_down = 30 + 1 + 15
    assert signal.iloc[:breakout_up].eq(0).all()
    assert signal.iloc[breakout_up:breakout_down].eq(1).all()
    assert signal.iloc[breakout_down:].eq(-1).all()


def test_search_records_optimizer_and_warns_on_missing_optuna(monkeypatch):
    ohlcv = _random_walk(400)
    funding = pd.Series(0.0, index=ohlcv.index)

    def missing_optuna(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'optuna'", name="optuna")

    monkeypatch.setattr(optimize, "tpe_search", missing_optuna)
    with pytest.warns(RuntimeWarning, match="optuna is not installed"):
        results = optimize.search(ohlcv, funding, CostProfile(), n_trials=2, seed=1)
    assert set(results["optimizer"]) == {"random_fallback_optuna_missing"}
    assert set(optimize.search(ohlcv, funding, CostProfile(), 2, 1, method="random")["optimizer"]) == {"random"}
