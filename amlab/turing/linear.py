"""Linear stability analysis of the Gierer-Meinhardt reaction-diffusion model.

The model is written in the course scaling (Murray, Mathematical Biology II,
Section 2.3):

    u_t = Laplacian(u) + gamma * (a - b u + u**2 / v)
    v_t = d * Laplacian(v) + gamma * (u**2 - v)

All functions accept NumPy arrays where it makes sense, so they can be used to
draw Turing spaces and dispersion curves.
"""

from typing import Tuple

import numpy as np


def gierer_meinhardt_steady_state(a: float = 0.40, b: float = 1.00) -> Tuple[float, float]:
    """Homogeneous steady state of the Gierer-Meinhardt kinetics.

    Parameters
    ----------
    a : float
        Source term of the activator.
    b : float
        Linear decay rate of the activator.

    Returns
    -------
    tuple of float
        (u_star, v_star) with u_star = (a + 1) / b and v_star = u_star**2.
    """
    u_star = (a + 1) / b
    return u_star, u_star**2


def gierer_meinhardt_jacobian(
    a: float = 0.40, b: float = 1.00
) -> Tuple[float, float, float, float]:
    """Jacobian entries (f_u, f_v, g_u, g_v) of the kinetics at the steady state.

    The factor gamma is not included. Multiply by gamma to get the reaction
    Jacobian of the scaled model.
    """
    u_star, v_star = gierer_meinhardt_steady_state(a, b)
    fu = -b + 2 * u_star / v_star
    fv = -(u_star**2) / v_star**2
    gu = 2 * u_star
    gv = -1.0 + 0.0 * u_star
    return fu, fv, gu, gv


def turing_conditions(fu, fv, gu, gv, d):
    """Check the four Turing conditions (Murray, Mathematical Biology II, Section 2.3).

    Parameters
    ----------
    fu, fv, gu, gv : float or np.ndarray
        Jacobian entries of the kinetics at the homogeneous steady state.
    d : float or np.ndarray
        Ratio of the inhibitor diffusion coefficient to the activator one.

    Returns
    -------
    bool or np.ndarray
        True where all four conditions hold.
    """
    det = fu * gv - fv * gu
    cond1 = (fu + gv) < 0
    cond2 = det > 0
    cond3 = (d * fu + gv) > 0
    cond4 = (d * fu + gv) ** 2 - 4 * d * det > 0
    return cond1 & cond2 & cond3 & cond4


def critical_diffusion_ratio(fu, fv, gu, gv):
    """Critical diffusion ratio d_c at which a Turing instability first appears.

    d_c is the root of d**2 fu**2 + 2 (2 fv gu - fu gv) d + gv**2 = 0 that
    satisfies d fu + gv > 0 (Murray, Mathematical Biology II, Section 2.3).
    Returns NaN when no admissible root exists.
    """
    qa = fu**2
    qb = 2 * (2 * fv * gu - fu * gv)
    qc = gv**2
    disc = qb**2 - 4 * qa * qc
    with np.errstate(invalid="ignore", divide="ignore"):
        root = (-qb + np.sqrt(disc)) / (2 * qa)
    ok = (disc >= 0) & (root * fu + gv > 0)
    return np.where(ok, root, np.nan)


def dispersion_relation(k2, fu, fv, gu, gv, d, gamma: float = 1.0):
    """Growth rate Re(sigma) of the mode with squared wavenumber k2.

    sigma solves det(gamma J - k2 diag(1, d) - sigma I) = 0. The function
    returns the largest real part of the two roots.
    """
    k2 = np.asarray(k2, dtype=float)
    tr = gamma * (fu + gv) - k2 * (1 + d)
    h = d * k2**2 - gamma * (d * fu + gv) * k2 + gamma**2 * (fu * gv - fv * gu)
    disc = tr**2 - 4 * h
    sqrt_disc = np.sqrt(np.abs(disc))
    return np.where(disc >= 0, 0.5 * (tr + sqrt_disc), 0.5 * tr)


def unstable_band(fu, fv, gu, gv, d, gamma: float = 1.0):
    """Interval (k1**2, k2**2) of squared wavenumbers with h(k2) < 0.

    Returns (nan, nan) when the band is empty.
    """
    det = fu * gv - fv * gu
    s = d * fu + gv
    disc = s**2 - 4 * d * det
    if s <= 0 or disc <= 0:
        return np.nan, np.nan
    k2_low = gamma * (s - np.sqrt(disc)) / (2 * d)
    k2_high = gamma * (s + np.sqrt(disc)) / (2 * d)
    return k2_low, k2_high


def laplacian_eigenvalues_1d(length: float, num_modes: int, bc: str = "neumann"):
    """Squared wavenumbers k_n**2 = -lambda_n of the Laplacian on (0, length).

    Parameters
    ----------
    length : float
        Length of the interval.
    num_modes : int
        Number of modes to return.
    bc : str
        'neumann' (n = 0, 1, ...), 'dirichlet' (n = 1, 2, ...), or
        'periodic' (n = 0, 1, ..., each n > 0 has a cosine and a sine mode).

    Returns
    -------
    n : np.ndarray
        Mode indices.
    k2 : np.ndarray
        Squared wavenumbers.
    """
    if bc == "neumann":
        n = np.arange(num_modes)
        k2 = (n * np.pi / length) ** 2
    elif bc == "dirichlet":
        n = np.arange(1, num_modes + 1)
        k2 = (n * np.pi / length) ** 2
    elif bc == "periodic":
        n = np.arange(num_modes)
        k2 = (2 * n * np.pi / length) ** 2
    else:
        raise ValueError("bc must be 'neumann', 'dirichlet' or 'periodic'")
    return n, k2


def unstable_modes_1d(
    a: float = 0.40,
    b: float = 1.00,
    d: float = 30.0,
    gamma: float = 1.0,
    length: float = 40.0,
    num_modes: int = 20,
    bc: str = "neumann",
):
    """List the unstable modes on (0, length), sorted from fastest to slowest.

    Returns
    -------
    list of tuple
        (n, growth_rate) for every mode with positive growth rate.
    """
    fu, fv, gu, gv = gierer_meinhardt_jacobian(a, b)
    n, k2 = laplacian_eigenvalues_1d(length, num_modes, bc)
    rates = dispersion_relation(k2, fu, fv, gu, gv, d, gamma)
    modes = [(int(ni), float(ri)) for ni, ri in zip(n, rates) if ri > 0]
    modes.sort(key=lambda item: item[1], reverse=True)
    return modes


def unstable_modes_2d(
    a: float = 0.40,
    b: float = 1.00,
    d: float = 30.0,
    gamma: float = 1.0,
    length_x: float = 20.0,
    length_y: float = 50.0,
    num_modes: int = 20,
    bc: str = "neumann",
):
    """List the unstable modes (n, m) on a rectangle, fastest first.

    The squared wavenumber of mode (n, m) is k_n**2 + k_m**2, with the 1D
    wavenumbers of `laplacian_eigenvalues_1d` in each direction. The flat
    mode (0, 0) is excluded.

    Returns
    -------
    list of tuple
        ((n, m), growth_rate) for every mode with positive growth rate.
    """
    fu, fv, gu, gv = gierer_meinhardt_jacobian(a, b)
    n, k2x = laplacian_eigenvalues_1d(length_x, num_modes, bc)
    m, k2y = laplacian_eigenvalues_1d(length_y, num_modes, bc)
    k2 = k2x[:, None] + k2y[None, :]
    rates = dispersion_relation(k2, fu, fv, gu, gv, d, gamma)
    modes = []
    for i in range(len(n)):
        for j in range(len(m)):
            if k2[i, j] > 0 and rates[i, j] > 0:
                modes.append(((int(n[i]), int(m[j])), float(rates[i, j])))
    modes.sort(key=lambda item: item[1], reverse=True)
    return modes
