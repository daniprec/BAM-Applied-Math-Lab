"""Fit the SIR model to daily incidence with nonlinear least squares."""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares

from amlab.epidemics.models import sir_rhs


def sir_incidence(
    days: np.ndarray, beta: float, gamma: float, i0: float, population: float
) -> np.ndarray:
    """Expected new cases per day predicted by the SIR model.

    Parameters
    ----------
    days : np.ndarray
        Integer days 1..n at which incidence is reported.
    beta, gamma : float
        SIR rates per day.
    i0 : float
        Initial infected fraction at day 0.
    population : float
        Population size.
    """
    t = np.arange(0, int(days[-1]) + 1)
    # odeint (LSODA) is faster than solve_ivp for many repeated small solves
    y = odeint(sir_rhs, [1 - i0, i0, 0.0], t, args=(beta, gamma), tfirst=True)
    inc = population * -np.diff(y[:, 0])
    return inc[np.asarray(days, dtype=int) - 1]


def fit_sir_incidence(
    days: np.ndarray,
    cases: np.ndarray,
    population: float,
    fit_population: bool = False,
    x0: tuple = (0.25, 0.12, 10.0),
) -> dict:
    """Estimate SIR parameters from daily incidence.

    Unknowns are beta, gamma and the initial number of infected people
    I0. With ``fit_population=True`` the effective population size N is
    also unknown, and ``population`` is only the starting guess. This is
    the realistic case: the number of people who can actually be reached
    by the epidemic is rarely known in advance.

    The residuals are sqrt(model) - sqrt(data), which roughly stabilises
    the variance of count data. Parameters are searched on a log scale to
    keep them positive.

    Parameters
    ----------
    days : np.ndarray
        Integer days 1..n.
    cases : np.ndarray
        Observed new cases on those days.
    population : float
        Population size (fixed), or initial guess if ``fit_population``.
    fit_population : bool, optional
        Whether to estimate N as well, by default False.
    x0 : tuple, optional
        Initial guess for (beta, gamma, I0 in people).

    Returns
    -------
    dict
        Keys ``beta``, ``gamma``, ``i0`` (fraction), ``population``, ``r0``,
        ``cost`` and ``success``.
    """
    cases = np.asarray(cases, dtype=float)

    def unpack(logp):
        p = np.exp(logp)
        n = p[3] if fit_population else population
        return p[0], p[1], p[2] / n, n

    def residuals(logp):
        beta, gamma, i0, n = unpack(logp)
        model = sir_incidence(days, beta, gamma, i0, n)
        return np.sqrt(np.clip(model, 0, None)) - np.sqrt(cases)

    guess = list(x0)
    lower = [1e-3, 1e-3, 1e-2]
    upper = [5.0, 2.0, 1e5]
    if fit_population:
        guess.append(0.3 * population)
        lower.append(1e3)
        upper.append(1e2 * population)
    res = least_squares(
        residuals, np.log(guess), bounds=(np.log(lower), np.log(upper))
    )
    beta, gamma, i0, n = unpack(res.x)
    return {
        "beta": beta,
        "gamma": gamma,
        "i0": i0,
        "population": n,
        "r0": beta / gamma,
        "cost": res.cost,
        "success": res.success,
    }
