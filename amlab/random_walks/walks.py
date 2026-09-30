"""Random walk simulations and power-law tools.

All functions take an optional ``rng`` argument (a ``numpy.random.Generator``)
so that results are reproducible. If ``rng`` is None a new generator is
created with ``np.random.default_rng()``.

References
----------
Berg, H. C. (1993). Random Walks in Biology, Chapter 1.
Newman, M. E. J. (2005). Power laws, Pareto distributions and Zipf's law.
Clauset, A., Shalizi, C. R. and Newman, M. E. J. (2009). Power-law
distributions in empirical data.
"""

from __future__ import annotations

import numpy as np


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def simulate_lattice_walk(
    n_walkers: int,
    n_steps: int,
    p: float = 0.5,
    dx: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate independent 1D lattice random walks that start at x = 0.

    At every step each walker moves +dx with probability p and -dx with
    probability 1 - p.

    Parameters
    ----------
    n_walkers : int
        Number of independent walkers.
    n_steps : int
        Number of steps per walker.
    p : float
        Probability of a step to the right.
    dx : float
        Step length.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    numpy.ndarray
        Positions with shape (n_walkers, n_steps + 1). Column 0 is the start.
    """
    rng = _rng(rng)
    steps = np.where(rng.random((n_walkers, n_steps)) < p, dx, -dx)
    x = np.zeros((n_walkers, n_steps + 1))
    x[:, 1:] = np.cumsum(steps, axis=1)
    return x


def simulate_gaussian_walk(
    n_walkers: int,
    n_steps: int,
    dim: int = 2,
    sigma: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate walks in ``dim`` dimensions with independent Gaussian steps.

    Each coordinate of each step is drawn from N(0, sigma^2).

    Returns
    -------
    numpy.ndarray
        Positions with shape (n_walkers, n_steps + 1, dim).
    """
    rng = _rng(rng)
    steps = rng.normal(0.0, sigma, size=(n_walkers, n_steps, dim))
    x = np.zeros((n_walkers, n_steps + 1, dim))
    x[:, 1:, :] = np.cumsum(steps, axis=1)
    return x


def mean_squared_displacement(positions: np.ndarray) -> np.ndarray:
    """Ensemble mean squared displacement from the starting point.

    Parameters
    ----------
    positions : numpy.ndarray
        Shape (n_walkers, n_times) for 1D walks or
        (n_walkers, n_times, dim) for walks in ``dim`` dimensions.

    Returns
    -------
    numpy.ndarray
        MSD at every time index, shape (n_times,).
    """
    disp = positions - positions[:, :1]
    sq = disp**2
    if sq.ndim == 3:
        sq = sq.sum(axis=2)
    return sq.mean(axis=0)


def heat_kernel(x: np.ndarray, t: float, D: float) -> np.ndarray:
    """Point-source solution of the 1D diffusion equation.

    p(x, t) = exp(-x^2 / (4 D t)) / sqrt(4 pi D t), which is a Gaussian
    with mean 0 and variance 2 D t (Berg 1993, Eq. 1.22).
    """
    return np.exp(-(x**2) / (4 * D * t)) / np.sqrt(4 * np.pi * D * t)


def sample_pareto(
    n: int,
    alpha: float,
    xmin: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a continuous power law p(x) = C x^(-alpha) for x >= xmin.

    Uses inverse transform sampling, x = xmin (1 - u)^(-1 / (alpha - 1))
    (Newman 2005, Clauset et al. 2009). Requires alpha > 1.

    Note that ``alpha`` is the exponent of the density. The complementary
    cumulative distribution decays as x^(-(alpha - 1)).
    """
    if alpha <= 1:
        raise ValueError("alpha must be larger than 1 for a normalizable power law")
    rng = _rng(rng)
    u = rng.random(n)
    return xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))


def mle_power_law_exponent(x: np.ndarray, xmin: float) -> tuple[float, float]:
    """Maximum likelihood estimate of the density exponent of a power law.

    alpha_hat = 1 + n / sum(ln(x_i / xmin)), with standard error
    (alpha_hat - 1) / sqrt(n), using only the data with x_i >= xmin
    (Newman 2005; Clauset et al. 2009).

    Returns
    -------
    alpha_hat : float
        Estimated exponent of the density p(x) ~ x^(-alpha).
    stderr : float
        Asymptotic standard error.
    """
    x = np.asarray(x, dtype=float)
    tail = x[x >= xmin]
    n = tail.size
    if n == 0:
        raise ValueError("no data above xmin")
    alpha_hat = 1.0 + n / np.sum(np.log(tail / xmin))
    return alpha_hat, (alpha_hat - 1.0) / np.sqrt(n)


def sample_symmetric_stable(
    alpha: float,
    size: int | tuple[int, ...],
    scale: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a symmetric alpha-stable law with the Chambers-Mallows-Stuck method.

    The characteristic function is E[exp(i k X)] = exp(-|scale * k|^alpha),
    with 0 < alpha <= 2. alpha = 2 gives a Gaussian with variance
    2 * scale^2 and alpha = 1 gives a Cauchy law with scale ``scale``.

    With V uniform on (-pi/2, pi/2) and W exponential with mean 1,

        X = sin(alpha V) / cos(V)^(1/alpha) * (cos((1 - alpha) V) / W)^((1 - alpha) / alpha)

    (Chambers, Mallows and Stuck 1976, symmetric case beta = 0). This
    matches ``scipy.stats.levy_stable(alpha, 0, scale=scale)``.
    """
    if not 0 < alpha <= 2:
        raise ValueError("alpha must satisfy 0 < alpha <= 2")
    rng = _rng(rng)
    v = rng.uniform(-np.pi / 2, np.pi / 2, size)
    w = rng.exponential(1.0, size)
    if alpha == 1:
        x = np.tan(v)
    else:
        x = (
            np.sin(alpha * v)
            / np.cos(v) ** (1.0 / alpha)
            * (np.cos((1.0 - alpha) * v) / w) ** ((1.0 - alpha) / alpha)
        )
    return scale * x


def levy_flight_2d(
    n_steps: int,
    mu: float = 2.0,
    lmin: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Planar random walk with isotropic directions and power-law step lengths.

    Step lengths follow p(l) ~ l^(-mu) for l >= lmin (the parametrization of
    Viswanathan et al. 1999). For 1 < mu <= 3 the step variance is infinite.
    Setting a large ``mu`` (for example 10) gives steps with finite variance.

    Returns
    -------
    numpy.ndarray
        Positions with shape (n_steps + 1, 2), starting at the origin.
    """
    rng = _rng(rng)
    lengths = sample_pareto(n_steps, mu, lmin, rng)
    angles = rng.uniform(0.0, 2.0 * np.pi, n_steps)
    steps = np.column_stack([lengths * np.cos(angles), lengths * np.sin(angles)])
    xy = np.zeros((n_steps + 1, 2))
    xy[1:] = np.cumsum(steps, axis=0)
    return xy
