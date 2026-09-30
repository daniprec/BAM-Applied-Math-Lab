"""Epidemic incidence data: real file if present, synthetic otherwise.

The expected file is ``data/epidemic/cases.csv`` with columns

    date,new_cases

where ``date`` is an ISO date (YYYY-MM-DD), one row per day in order, and
``new_cases`` is the number of newly reported cases on that day. See
``data/epidemic/README.md``.
"""

import os
import warnings

import numpy as np
from scipy.integrate import solve_ivp

from amlab.epidemics.models import sir_rhs

DEFAULT_PATH = os.path.join("data", "epidemic", "cases.csv")


def synthetic_sir_cases(
    n_days: int = 120,
    population: float = 1e6,
    beta: float = 0.3,
    gamma: float = 0.1,
    i0: float = 1e-5,
    seed: int = 2020,
    sigma: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Daily incidence of a deterministic SIR epidemic with reporting noise.

    The expected count on day d is m_d = population * (S(d-1) - S(d)).
    The reported count is Poisson with mean m_d * exp(sigma * Z_d), where
    Z_d are independent standard normal variables. The lognormal factor
    mimics day-to-day reporting fluctuations larger than Poisson noise.

    Parameters
    ----------
    n_days : int, optional
        Number of days, by default 120.
    population : float, optional
        Population size, by default one million.
    beta, gamma : float, optional
        SIR rates per day, by default 0.3 and 0.1 (R0 = 3).
    i0 : float, optional
        Initial infected fraction, by default 1e-5.
    seed : int, optional
        Seed for ``np.random.default_rng``, by default 2020.
    sigma : float, optional
        Standard deviation of the log reporting factor, by default 0.2.
        Use 0 for pure Poisson noise.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Days 1..n_days and the noisy daily counts.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, n_days + 1)
    sol = solve_ivp(
        sir_rhs, (0, n_days), [1 - i0, i0, 0.0], args=(beta, gamma), t_eval=t,
        rtol=1e-8, atol=1e-10,
    )
    mean = population * -np.diff(sol.y[0])
    factor = np.exp(sigma * rng.standard_normal(mean.size))
    cases = rng.poisson(np.clip(mean, 0, None) * factor)
    return t[1:], cases


def load_cases(path: str = DEFAULT_PATH) -> tuple[np.ndarray, np.ndarray, bool]:
    """Load daily incidence from ``path`` or fall back to synthetic data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, bool]
        Day index (1, 2, ...), daily new cases, and a flag that is True when
        the data are synthetic.
    """
    if os.path.exists(path):
        import pandas as pd

        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        cases = df["new_cases"].to_numpy(dtype=float)
        days = np.arange(1, len(cases) + 1)
        return days, cases, False
    warnings.warn(
        f"{path} not found: using synthetic SIR data (R0 = 3, seed 2020, sigma 0.2).",
        stacklevel=2,
    )
    days, cases = synthetic_sir_cases()
    return days, cases.astype(float), True
