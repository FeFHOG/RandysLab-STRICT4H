"""Selection statistics for telling real edges from multiple-testing noise.

Sharpe ratios are per-period (not annualized) and kurtosis is non-excess (3 for normal returns) throughout.
"""

from __future__ import annotations

import math
from itertools import combinations
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

EULER_GAMMA = 0.5772156649015329
STD_EPSILON = 1e-12
MIN_IC_OBS = 10
PBO_METRICS = ("sharpe", "mean")
_PBO_CHUNK_CELLS = 1 << 21
_NORMAL = NormalDist()


def sharpe_ratio(returns: ArrayLike) -> float:
    """Per-period Sharpe ratio mean / std (ddof=1) of the finite values; 0.0 if n < 2 or the std is 0."""
    x = _finite(returns)
    if x.size < 2:
        return 0.0
    std = float(x.std(ddof=1))
    return float(x.mean()) / std if std > STD_EPSILON else 0.0


def annualize_sharpe(sr: float, periods_per_year: float) -> float:
    """sr * sqrt(periods_per_year) (iid scaling), e.g. periods_per_year=metrics.BARS_PER_YEAR_4H for 4h bars."""
    return float(sr) * math.sqrt(periods_per_year)


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, n_obs: float, skew: float, kurtosis: float) -> float:
    """Probabilistic Sharpe ratio, Bailey & Lopez de Prado (2012) "The Sharpe Ratio Efficient Frontier".

    PSR(SR*) = Phi((SR - SR*) sqrt(n - 1) / sqrt(1 - g3 SR + (g4 - 1) / 4 SR^2)) with per-period SR and SR*, skewness g3
    and non-excess kurtosis g4. NaN if n_obs < 2 or the variance term under the root is <= 0.
    """
    var = _sharpe_variance_term(sr, skew, kurtosis)
    if n_obs < 2 or not np.isfinite(var) or var <= 0:
        return math.nan
    return _NORMAL.cdf((sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(var))


def psr_from_returns(returns: ArrayLike, sr_benchmark: float = 0.0) -> float:
    """PSR of the sample Sharpe ratio of the finite returns against sr_benchmark.

    Skewness and kurtosis are the plain (ddof=0) moment estimators m3 / m2^1.5 and m4 / m2^2. NaN if fewer than 2
    observations or zero variance.
    """
    x = _finite(returns)
    skew, kurtosis = _skew_kurtosis(x)
    return probabilistic_sharpe_ratio(sharpe_ratio(x), sr_benchmark, x.size, skew, kurtosis)


def expected_max_sharpe(n_trials: float, sr_variance: float, sr_mean: float = 0.0) -> float:
    """Expected maximum Sharpe ratio of n_trials unskilled trials (False Strategy Theorem).

    E[max SR] ~ E[SR] + sqrt(V[SR]) ((1 - g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e))), g = Euler-Mascheroni constant;
    Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio". Returns sr_mean for N <= 1.
    """
    if sr_variance < 0:
        raise ValueError("sr_variance must be non-negative")
    if n_trials <= 1:
        return float(sr_mean)
    z_first = _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    z_second = _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return float(sr_mean) + math.sqrt(sr_variance) * ((1.0 - EULER_GAMMA) * z_first + EULER_GAMMA * z_second)


def deflated_sharpe_ratio(returns: ArrayLike, n_trials: int, sr_variance: float) -> dict[str, Any]:
    """Deflated Sharpe ratio, Bailey & Lopez de Prado (2014): PSR with SR* = expected_max_sharpe(n_trials, sr_variance).

    `returns` are the per-period returns of the selected strategy. `sr_variance` is the cross-trial variance of the
    per-period (non-annualized) Sharpe estimates of all trials, and `n_trials` must count EVERY candidate evaluated,
    including rejects; understating either one inflates the DSR.
    """
    x = _finite(returns)
    sr = sharpe_ratio(x)
    skew, kurtosis = _skew_kurtosis(x)
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return {
        "dsr": probabilistic_sharpe_ratio(sr, sr0, x.size, skew, kurtosis),
        "sr": sr,
        "sr0": sr0,
        "n_obs": int(x.size),
        "skew": skew,
        "kurtosis": kurtosis,
        "n_trials": int(n_trials),
    }


def min_track_record_length(
    sr: float,
    sr_benchmark: float,
    skew: float,
    kurtosis: float,
    prob: float = 0.95,
) -> float:
    """Minimum track record length, Bailey & Lopez de Prado (2012), in observations at the frequency of sr.

    MinTRL = 1 + (1 - g3 SR + (g4 - 1) / 4 SR^2) (Phi^-1(prob) / (SR - SR*))^2, the n at which PSR(SR*) reaches prob.
    inf if SR <= SR*, NaN if the variance term is <= 0.
    """
    if sr <= sr_benchmark:
        return math.inf
    var = _sharpe_variance_term(sr, skew, kurtosis)
    if not np.isfinite(var) or var <= 0:
        return math.nan
    return 1.0 + var * (_NORMAL.inv_cdf(prob) / (sr - sr_benchmark)) ** 2


def pbo_cscv(perf: pd.DataFrame, n_partitions: int = 16, metric: str = "sharpe") -> dict[str, Any]:
    """Probability of backtest overfitting by CSCV, Bailey, Borwein, Lopez de Prado & Zhu (2017).

    `perf` holds T time-ordered rows x N strategy columns of per-period returns. The rows are cut into n_partitions
    contiguous equal blocks (remainder rows at the end are dropped). For every choice of half the blocks as in-sample
    (IS), the best IS strategy by `metric` ("sharpe": mean / std ddof=1, or "mean") gets its out-of-sample (OOS)
    relative rank w = rank / (N + 1) (rank 1 = worst, ties averaged) and logit ln(w / (1 - w)); PBO is the share of
    combinations with logit <= 0. prob_oos_loss is the share where the selected strategy's OOS metric is < 0 and
    degradation_slope the OLS slope of its OOS metric on its IS metric.
    """
    values = np.asarray(perf, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("perf must be a T x N frame with at least 2 strategy columns")
    if n_partitions < 2 or n_partitions % 2:
        raise ValueError("n_partitions must be an even integer >= 2")
    if values.shape[0] < 2 * n_partitions:
        raise ValueError("perf needs at least 2 * n_partitions rows")
    if metric not in PBO_METRICS:
        raise ValueError(f"metric must be one of {PBO_METRICS}")
    if not np.isfinite(values).all():
        raise ValueError("perf must be finite; align or fill missing returns explicitly")

    n_strategies = values.shape[1]
    block_len = values.shape[0] // n_partitions
    values = values[: block_len * n_partitions]
    col_mean = values.mean(axis=0)
    # Centered columns keep the sum-of-squares variance numerically stable.
    blocks = (values - col_mean).reshape(n_partitions, block_len, n_strategies)
    block_sum = blocks.sum(axis=1)
    block_sumsq = np.square(blocks).sum(axis=1)
    total_sum = block_sum.sum(axis=0)
    total_sumsq = block_sumsq.sum(axis=0)
    n_half = block_len * n_partitions // 2

    combos = np.array(list(combinations(range(n_partitions), n_partitions // 2)))
    n_combinations = len(combos)
    in_sample = np.zeros((n_combinations, n_partitions))
    in_sample[np.arange(n_combinations)[:, None], combos] = 1.0

    logits = np.empty(n_combinations)
    is_selected = np.empty(n_combinations)
    oos_selected = np.empty(n_combinations)
    chunk = max(1, _PBO_CHUNK_CELLS // n_strategies)
    for start in range(0, n_combinations, chunk):
        part = slice(start, start + chunk)
        is_sum = in_sample[part] @ block_sum
        is_sumsq = in_sample[part] @ block_sumsq
        is_metric = _metric_from_sums(is_sum, is_sumsq, n_half, col_mean, metric)
        oos_metric = _metric_from_sums(total_sum - is_sum, total_sumsq - is_sumsq, n_half, col_mean, metric)
        rows = np.arange(is_metric.shape[0])
        best = is_metric.argmax(axis=1)
        selected = oos_metric[rows, best][:, None]
        rank = (oos_metric < selected).sum(axis=1) + ((oos_metric == selected).sum(axis=1) + 1) / 2.0
        omega = rank / (n_strategies + 1.0)
        logits[part] = np.log(omega / (1.0 - omega))
        is_selected[part] = is_metric[rows, best]
        oos_selected[part] = selected[:, 0]

    return {
        "pbo": float(np.mean(logits <= 0.0)),
        "logits": logits,
        "n_combinations": int(n_combinations),
        "n_strategies": int(n_strategies),
        "n_partitions": int(n_partitions),
        "prob_oos_loss": float(np.mean(oos_selected < 0.0)),
        "degradation_slope": _ols_slope(is_selected, oos_selected),
    }


def newey_west_se(x: ArrayLike, lags: int | None = None) -> float:
    """Newey-West (1987) HAC standard error of the sample mean of the finite values, Bartlett kernel.

    Long-run variance S = g_0 + 2 sum_{j=1..L} (1 - j / (L + 1)) g_j with g_j = (1/T) sum_t u_t u_{t-j}, u = x - mean;
    Var(mean) = S / T. The autocovariances use the 1/T convention (no degrees-of-freedom correction), so lags=0 gives
    var(x, ddof=0) / T. Default L = floor(4 (T / 100)^(2/9)) (Newey & West 1994). NaN if T < 2.
    """
    v = _finite(x)
    n = v.size
    if n < 2:
        return math.nan
    n_lags = _resolve_lags(n, lags)
    u = v - v.mean()
    long_run_var = float(u @ u) / n
    for j in range(1, min(n_lags, n - 1) + 1):
        long_run_var += 2.0 * (1.0 - j / (n_lags + 1.0)) * float(u[j:] @ u[:-j]) / n
    return math.sqrt(max(long_run_var, 0.0) / n)


def newey_west_tstat(x: ArrayLike, lags: int | None = None) -> float:
    """t-stat of mean = 0 with the Newey-West standard error; NaN if the standard error is 0 or undefined."""
    v = _finite(x)
    se = newey_west_se(v, lags)
    if not np.isfinite(se) or se <= STD_EPSILON:
        return math.nan
    return float(v.mean()) / se


def rank_ic_with_hac(factor: pd.Series, forward_return: pd.Series, lags: int | None = None) -> dict[str, Any]:
    """Spearman rank IC of factor vs forward_return with a Newey-West t-stat.

    The series are inner-joined on the index (in the factor's order, which should be time order) and non-finite pairs
    dropped. With z-scored (ddof=0) average ranks, mean(z_f * z_r) is the Pearson correlation of the ranks, i.e. the
    Spearman IC, and the t-stat is newey_west_tstat of those products, robust to the serial correlation of overlapping
    forward returns. ic and tstat are NaN if fewer than MIN_IC_OBS pairs remain or either side is constant.
    """
    left, right = factor.align(forward_return, join="inner")
    f = left.to_numpy(dtype=float)
    r = right.to_numpy(dtype=float)
    keep = np.isfinite(f) & np.isfinite(r)
    n = int(keep.sum())
    out: dict[str, Any] = {"ic": math.nan, "tstat": math.nan, "n": n, "lags": _resolve_lags(n, lags)}
    if n < MIN_IC_OBS:
        return out
    z_factor = _zscore(pd.Series(f[keep]).rank(method="average").to_numpy())
    z_return = _zscore(pd.Series(r[keep]).rank(method="average").to_numpy())
    if z_factor is None or z_return is None:
        return out
    products = z_factor * z_return
    out["ic"] = float(products.mean())
    out["tstat"] = newey_west_tstat(products, out["lags"])
    return out


def benjamini_hochberg(pvalues: ArrayLike, q: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini & Hochberg (1995) step-up FDR control at level q; returns (discovery mask, adjusted p) in input order.

    Rejects the k smallest p-values, k = max{i : p_(i) <= q i / m}; adjusted p_(i) = min_{j >= i} min(1, m p_(j) / j).
    NaN p-values count toward m (conservative) but are never discoveries and keep a NaN adjusted p-value.
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1)")
    p = np.asarray(pvalues, dtype=float).ravel()
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    missing = np.isnan(p)
    m = p.size
    order = np.argsort(np.where(missing, 1.0, p), kind="mergesort")
    ranked = np.where(missing, 1.0, p)[order]
    rank = np.arange(1, m + 1)
    passed = np.flatnonzero(ranked <= q * rank / m)
    discoveries = np.zeros(m, dtype=bool)
    if passed.size:
        discoveries[order[: passed[-1] + 1]] = True
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(np.minimum.accumulate((ranked * m / rank)[::-1])[::-1], 1.0)
    adjusted[missing] = np.nan
    return discoveries, adjusted


def _finite(values: ArrayLike) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    return x[np.isfinite(x)]


def _skew_kurtosis(x: np.ndarray) -> tuple[float, float]:
    if x.size < 2:
        return math.nan, math.nan
    d = x - x.mean()
    m2 = float(np.mean(d**2))
    if math.sqrt(m2) <= STD_EPSILON:
        return math.nan, math.nan
    return float(np.mean(d**3)) / m2**1.5, float(np.mean(d**4)) / m2**2


def _sharpe_variance_term(sr: float, skew: float, kurtosis: float) -> float:
    return 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2


def _metric_from_sums(
    centered_sum: np.ndarray,
    centered_sumsq: np.ndarray,
    n: int,
    col_mean: np.ndarray,
    metric: str,
) -> np.ndarray:
    mean = col_mean + centered_sum / n
    if metric == "mean":
        return mean
    std = np.sqrt(np.maximum(centered_sumsq - centered_sum**2 / n, 0.0) / (n - 1))
    return np.divide(mean, std, out=np.zeros_like(mean), where=std > STD_EPSILON)


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    dx = x - x.mean()
    sxx = float(dx @ dx)
    return float(dx @ (y - y.mean())) / sxx if sxx > 0 else math.nan


def _resolve_lags(n_obs: int, lags: int | None) -> int:
    if lags is None:
        return math.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0))
    if lags < 0:
        raise ValueError("lags must be non-negative")
    return int(lags)


def _zscore(x: np.ndarray) -> np.ndarray | None:
    std = float(x.std())
    return (x - x.mean()) / std if std > STD_EPSILON else None
