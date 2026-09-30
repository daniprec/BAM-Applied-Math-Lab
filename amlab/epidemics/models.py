"""SIR and SIS models in normalized form (fractions of a closed population).

References: Kermack and McKendrick (1927); Hethcote (2000), SIAM Review 42.
"""

import numpy as np
from scipy.optimize import brentq


def sir_rhs(t: float, y: np.ndarray, beta: float, gamma: float) -> list:
    """Right-hand side of the SIR model.

    dS/dt = -beta S I,  dI/dt = beta S I - gamma I,  dR/dt = gamma I.

    Parameters
    ----------
    t : float
        Time (unused, required by solve_ivp).
    y : np.ndarray
        State (S, I, R) as fractions of the population.
    beta : float
        Transmission rate (1/time).
    gamma : float
        Recovery rate (1/time).
    """
    s, i, _ = y
    new = beta * s * i
    return [-new, new - gamma * i, gamma * i]


def sis_rhs(t: float, i: np.ndarray, beta: float, gamma: float) -> list:
    """Right-hand side of the SIS model with S = 1 - I.

    dI/dt = beta (1 - I) I - gamma I.
    """
    i = np.atleast_1d(i)[0]
    return [beta * (1 - i) * i - gamma * i]


def sis_endemic_level(r0: float) -> float:
    """Endemic equilibrium I* = 1 - 1/R0 of the SIS model (0 if R0 <= 1)."""
    return max(0.0, 1.0 - 1.0 / r0)


def final_size(r0: float, s0: float = 1.0) -> float:
    """Fraction of the population infected over the whole SIR epidemic.

    Solves the final size relation ln(s0 / s_inf) = r0 (1 - s_inf) for
    s_inf in (0, 1/r0) and returns z = s0 - s_inf. Assumes R(0) = 0 and
    I(0) = 1 - s0.

    Parameters
    ----------
    r0 : float
        Basic reproduction number beta / gamma.
    s0 : float, optional
        Initial susceptible fraction, by default 1 (limit of a tiny seed).
    """
    if r0 * s0 <= 1.0 and s0 == 1.0:
        return 0.0

    def g(s_inf):
        return np.log(s0 / s_inf) - r0 * (1 - s_inf)

    s_inf = brentq(g, 1e-12, min(s0, 1.0 / r0) - 1e-12)
    return s0 - s_inf


def peak_prevalence(r0: float, s0: float, i0: float) -> float:
    """Maximum infected fraction of the SIR model.

    I_max = i0 + s0 - 1/r0 - (1/r0) ln(r0 s0), valid when r0 s0 > 1.
    """
    if r0 * s0 <= 1.0:
        return i0
    return i0 + s0 - 1.0 / r0 - np.log(r0 * s0) / r0
