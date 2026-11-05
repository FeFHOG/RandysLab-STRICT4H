from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from randyslab.stats import (
    annualize_sharpe,
    benjamini_hochberg,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    newey_west_se,
    newey_west_tstat,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    psr_from_returns,
    rank_ic_with_hac,
    sharpe_ratio,
)


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=_index(len(values)))


def test_sharpe_ratio_drops_non_finite_values_and_guards_degenerate_input() -> None:
    assert sharpe_ratio([0.01, np.nan, 0.02, 0.03, np.inf]) == pytest.approx(2.0)
    assert sharpe_ratio(pd.Series([0.01, 0.01, 0.01])) == 0.0
    assert sharpe_ratio([0.05]) == 0.0
    assert annualize_sharpe(0.1, 2190.0) == pytest.approx(0.1 * math.sqrt(2190.0))


def test_psr_is_one_half_at_the_benchmark_and_grows_with_track_record() -> None:
    assert probabilistic_sharpe_ratio(0.1, 0.1, 500, -0.5, 6.0) == 0.5
    psr = [probabilistic_sharpe_ratio(0.05, 0.0, n, -0.5, 6.0) for n in (50, 200, 1000, 5000)]
    assert np.all(np.diff(psr) > 0)
    assert math.isnan(probabilistic_sharpe_ratio(1.0, 0.0, 100, 5.0, 3.0))  # variance term 1 - 5 + 0.5 < 0


def test_psr_from_returns_detects_a_positive_mean() -> None:
    # Pin the sample moments of the N(0.05, 1) draw so the check does not hinge on the seed (raw draws miss ~12%).
    z = np.random.default_rng(7).standard_normal(5000)
    returns = 0.05 + (z - z.mean()) / z.std(ddof=1)
    assert sharpe_ratio(returns) == pytest.approx(0.05)
    assert psr_from_returns(returns) > 0.99


def test_expected_max_sharpe_follows_the_false_strategy_theorem() -> None:
    assert expected_max_sharpe(1, 0.01, sr_mean=0.02) == 0.02
    values = [expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000)]
    assert np.all(np.diff(values) > 0)
    assert expected_max_sharpe(50, 0.04) == pytest.approx(2.0 * expected_max_sharpe(50, 0.01))


def test_dsr_reproduces_the_bailey_lopez_de_prado_2014_example() -> None:
    # 100 trials with annualized V[SR] = 0.5; best SR 2.5 annualized over 1250 daily bars, skew -3, kurtosis 10.
    sr0 = expected_max_sharpe(100, 0.5 / 250)
    assert sr0 == pytest.approx(0.1132, abs=1e-4)
    assert probabilistic_sharpe_ratio(2.5 / math.sqrt(250), sr0, 1250, -3.0, 10.0) == pytest.approx(0.9004, abs=1e-4)


def test_dsr_deflates_psr_and_falls_as_trials_grow() -> None:
    returns = np.random.default_rng(11).normal(0.04, 1.0, 3000)
    psr = psr_from_returns(returns)
    assert deflated_sharpe_ratio(returns, 1, 4e-4)["dsr"] == pytest.approx(psr)
    dsr = [deflated_sharpe_ratio(returns, n, 4e-4)["dsr"] for n in (2, 10, 100, 1000)]
    assert dsr[0] < psr
    assert np.all(np.diff(dsr) < 0)
    out = deflated_sharpe_ratio(returns, 100, 4e-4)
    assert set(out) == {"dsr", "sr", "sr0", "n_obs", "skew", "kurtosis", "n_trials"}
    assert out["sr0"] == pytest.approx(expected_max_sharpe(100, 4e-4))
    assert (out["n_obs"], out["n_trials"]) == (3000, 100)


def test_min_track_record_length_shrinks_with_sharpe_and_is_infinite_at_or_below_benchmark() -> None:
    lengths = [min_track_record_length(sr, 0.0, -0.5, 6.0) for sr in (0.02, 0.05, 0.1, 0.2)]
    assert np.all(np.diff(lengths) < 0)
    assert min_track_record_length(0.05, 0.05, 0.0, 3.0) == math.inf
    assert min_track_record_length(0.01, 0.05, 0.0, 3.0) == math.inf
    n_star = min_track_record_length(0.08, 0.02, -0.5, 6.0, prob=0.95)
    assert probabilistic_sharpe_ratio(0.08, 0.02, n_star, -0.5, 6.0) == pytest.approx(0.95)


def test_pbo_of_pure_noise_is_near_one_half() -> None:
    # One panel's PBO is itself noisy (sd ~0.19 across seeds), so average a few independent panels.
    results = [pbo_cscv(_frame(np.random.default_rng(seed).normal(0.0, 1.0, (2000, 30)))) for seed in range(8)]
    assert 0.3 < np.mean([out["pbo"] for out in results]) < 0.7
    first = results[0]
    assert first["n_combinations"] == 12870 and first["logits"].shape == (12870,)
    assert (first["n_strategies"], first["n_partitions"]) == (30, 16)


@pytest.mark.parametrize("metric", ["sharpe", "mean"])
def test_pbo_is_low_with_a_planted_edge(metric: str) -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(0.0, 1.0, (4000, 20))
    values[:, 7] = rng.normal(0.1, 1.0, 4000)
    out = pbo_cscv(_frame(values), metric=metric)
    assert out["pbo"] < 0.1
    assert out["prob_oos_loss"] < 0.1


def test_pbo_matches_brute_force_cscv_with_eight_partitions() -> None:
    rng = np.random.default_rng(9)
    perf = pd.DataFrame(rng.normal(0.01, 1.0, (83, 6)))
    perf[5] = perf[2]  # duplicate strategy -> tied OOS ranks
    out = pbo_cscv(perf, n_partitions=8)

    expected = []
    for in_blocks in combinations(range(8), 4):  # 83 rows -> 8 blocks of 10, last 3 rows dropped
        is_rows = np.concatenate([np.arange(10 * b, 10 * b + 10) for b in in_blocks])
        oos_rows = np.setdiff1d(np.arange(80), is_rows)
        is_sharpe = perf.iloc[is_rows].apply(sharpe_ratio)
        oos_sharpe = perf.iloc[oos_rows].apply(sharpe_ratio)
        omega = oos_sharpe.rank()[is_sharpe.idxmax()] / 7.0
        expected.append(math.log(omega / (1.0 - omega)))
    assert out["n_combinations"] == 70
    np.testing.assert_allclose(out["logits"], expected)
    assert out["pbo"] == pytest.approx(np.mean(np.array(expected) <= 0.0))


def test_pbo_rejects_invalid_inputs() -> None:
    perf = pd.DataFrame(np.random.default_rng(0).normal(size=(40, 3)))
    for kwargs in ({"n_partitions": 5}, {"n_partitions": 32}, {"metric": "sortino"}):
        with pytest.raises(ValueError):
            pbo_cscv(perf, **kwargs)
    missing = perf.copy()
    missing.iloc[3, 1] = np.nan
    with pytest.raises(ValueError):
        pbo_cscv(missing, n_partitions=4)


def test_newey_west_without_lags_is_the_ddof0_variance_of_the_mean() -> None:
    x = np.random.default_rng(1).normal(0.1, 2.0, 500)
    iid_se = math.sqrt(np.var(x, ddof=0) / x.size)
    assert newey_west_se(x, lags=0) == pytest.approx(iid_se)
    assert newey_west_tstat(x, lags=0) == pytest.approx(x.mean() / iid_se)
    # u = (-1.5, -0.5, 0.5, 1.5): g0 = 1.25, g1 = 0.3125, S = g0 + 2 * (1/2) * g1 = 1.5625, se = sqrt(S / 4)
    assert newey_west_se([1.0, 2.0, 3.0, 4.0], lags=1) == pytest.approx(0.625)


def test_newey_west_widens_the_standard_error_under_autocorrelation() -> None:
    eps = np.random.default_rng(2).normal(size=5000)
    x = np.empty_like(eps)
    x[0] = eps[0]
    for t in range(1, x.size):
        x[t] = 0.7 * x[t - 1] + eps[t]
    assert newey_west_se(x) > 1.5 * newey_west_se(x, lags=0)


def test_rank_ic_on_independent_noise_is_insignificant() -> None:
    rng = np.random.default_rng(4)
    idx = _index(3000)
    out = rank_ic_with_hac(pd.Series(rng.normal(size=3000), index=idx), pd.Series(rng.normal(size=3000), index=idx))
    assert (out["n"], out["lags"]) == (3000, 8)  # floor(4 * (3000 / 100) ** (2 / 9))
    assert abs(out["tstat"]) < 3.5


def test_rank_ic_detects_a_planted_monotone_relationship() -> None:
    rng = np.random.default_rng(6)
    idx = _index(3000)
    factor = pd.Series(rng.normal(size=3000), index=idx)
    forward = pd.Series(0.5 * np.tanh(factor.to_numpy()) + rng.normal(size=3000), index=idx)
    forward.iloc[:5] = np.nan
    forward = forward.iloc[:-10]  # alignment drops the unmatched factor tail
    out = rank_ic_with_hac(factor, forward)
    spearman = pd.DataFrame({"f": factor, "r": forward}).corr(method="spearman").loc["f", "r"]
    assert out["n"] == 2985
    assert out["tstat"] > 5
    assert out["ic"] == pytest.approx(spearman, abs=1e-9)
    short = rank_ic_with_hac(factor.iloc[:9], factor.iloc[:9])
    assert short["n"] == 9 and math.isnan(short["ic"]) and math.isnan(short["tstat"])


def test_benjamini_hochberg_matches_a_hand_computed_step_up() -> None:
    # Sorted p 0.001, 0.045, 0.05, 0.3, 0.9 vs thresholds q*i/m = 0.02, 0.04, 0.06, 0.08, 0.10: the largest pass is
    # i = 3, so 0.045 is rejected even though it misses its own threshold. Raw m*p/i = 0.005, 0.1125, 0.0833, 0.375, 0.9
    # and the running minimum from the top pulls 0.1125 down to 0.0833.
    discoveries, adjusted = benjamini_hochberg([0.3, 0.05, 0.001, 0.9, 0.045], q=0.10)
    assert discoveries.tolist() == [False, True, True, False, True]
    np.testing.assert_allclose(adjusted, [0.375, 0.25 / 3, 0.005, 0.9, 0.25 / 3])
    # NaN counts toward m = 2 (adjusted 0.01 * 2 / 1) but is never a discovery.
    discoveries, adjusted = benjamini_hochberg([0.01, np.nan], q=0.10)
    assert discoveries.tolist() == [True, False]
    assert adjusted[0] == pytest.approx(0.02) and math.isnan(adjusted[1])
