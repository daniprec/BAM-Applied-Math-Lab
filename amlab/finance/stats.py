"""Statistics for financial returns: aggregation, autocorrelation,
normality testing and tail estimation.

References
----------
Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov test for normality with
mean and variance unknown. JASA 62, 399-402.
Hill, B. M. (1975). A simple general approach to inference about the tail of
a distribution. Ann. Statist. 3, 1163-1174.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def log_returns(prices: pd.Series | np.ndarray) -> np.ndarray:
    """Log returns r_t = ln(P_t / P_{t-1}).

    Parameters
    ----------
    prices : array-like
        Positive prices in time order.

    Returns
    -------
    numpy.ndarray
        Array of length len(prices) - 1.
    """
    p = np.asarray(prices, dtype=float)
    return np.diff(np.log(p))


def aggregate_returns(r: np.ndarray, horizon: int) -> np.ndarray:
    """Sum log returns over non-overlapping blocks of ``horizon`` steps.

    Log returns add up, so the block sum is the log return over the horizon.
    Incomplete blocks at the end are dropped.
    """
    r = np.asarray(r, dtype=float)
    m = r.size // horizon
    return r[: m * horizon].reshape(m, horizon).sum(axis=1)


def autocorrelation(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Sample autocorrelation at lags 0, 1, ..., max_lag."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    var = np.dot(x, x)
    return np.array([np.dot(x[: x.size - k], x[k:]) / var for k in range(max_lag + 1)])


def _ks_normal_statistic(x: np.ndarray) -> float:
    """KS distance between the sample and N(mean(x), std(x)^2)."""
    mu, sd = np.mean(x), np.std(x, ddof=1)
    return stats.kstest(x, "norm", args=(mu, sd)).statistic


def ks_normal_bootstrap(
    x: np.ndarray,
    n_boot: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Kolmogorov-Smirnov normality test with estimated mean and variance.

    The statistic D compares the empirical CDF with a normal CDF whose mean
    and standard deviation are estimated from the same data. The standard KS
    p-value is not valid in this case (Lilliefors 1967). Here the null
    distribution of D is obtained by parametric bootstrap: simulate
    ``n_boot`` normal samples of the same size, re-estimate the parameters
    on each one, and recompute D.

    Parameters
    ----------
    x : numpy.ndarray
        Data sample.
    n_boot : int
        Number of bootstrap samples.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    D : float
        Observed KS statistic.
    p_value : float
        Bootstrap p-value (1 + #{D_b >= D}) / (n_boot + 1).
    """
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x, dtype=float)
    d_obs = _ks_normal_statistic(x)
    # The statistic does not depend on the true mean and variance, so we
    # can simulate from N(0, 1).
    d_boot = np.array(
        [_ks_normal_statistic(rng.standard_normal(x.size)) for _ in range(n_boot)]
    )
    p_value = (1 + np.sum(d_boot >= d_obs)) / (n_boot + 1)
    return d_obs, p_value


def hill_estimator(x: np.ndarray, k: int | np.ndarray) -> float | np.ndarray:
    """Hill estimator of the tail index from the k largest values.

    With X_(1) >= X_(2) >= ... the order statistics of the positive data,

        xi_hat = (1/k) * sum_{i=1}^{k} ln(X_(i) / X_(k+1)),
        tail index alpha_hat = 1 / xi_hat,

    so that P(X > x) ~ x^(-alpha) (Hill 1975). Pass absolute returns to
    estimate the tail of |r|.

    Parameters
    ----------
    x : numpy.ndarray
        Data. Only positive values are used.
    k : int or array of int
        Number of upper order statistics (1 <= k < number of positive values).

    Returns
    -------
    float or numpy.ndarray
        Tail index estimate for each k.
    """
    xs = np.sort(np.asarray(x, dtype=float)[np.asarray(x) > 0])[::-1]
    logs = np.log(xs)
    cums = np.cumsum(logs)
    k_arr = np.atleast_1d(k).astype(int)
    if np.any(k_arr < 1) or np.any(k_arr >= xs.size):
        raise ValueError("k must satisfy 1 <= k < number of positive values")
    xi = cums[k_arr - 1] / k_arr - logs[k_arr]
    alpha = 1.0 / xi
    return alpha[0] if np.ndim(k) == 0 else alpha
