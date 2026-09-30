"""Vectorized integration of the Kuramoto model for many couplings at once.

Each row of the phase array is an independent population that shares the
same natural frequencies but has its own coupling strength K. This makes a
scan of the order parameter r against K fast enough to run inside a page.

The model is the mean-field form of Kuramoto (1975):

    d(theta_i)/dt = omega_i + K r sin(psi - theta_i),
    r exp(i psi) = (1/N) sum_j exp(i theta_j).
"""

import numpy as np


def _rhs(theta: np.ndarray, omega: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Right-hand side for a stack of populations, shape (num_k, N)."""
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # r cos(psi) and r sin(psi), one value per population
    rc = cos_t.mean(axis=1, keepdims=True)
    rs = sin_t.mean(axis=1, keepdims=True)
    # K r sin(psi - theta) = K (r sin(psi) cos(theta) - r cos(psi) sin(theta))
    return omega[None, :] + k[:, None] * (rs * cos_t - rc * sin_t)


def order_parameter_sweep(
    k_values: np.ndarray,
    omega: np.ndarray,
    theta0: np.ndarray,
    t_end: float = 100.0,
    dt: float = 0.05,
    t_average: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Time-averaged order parameter for each coupling strength.

    Uses the classical fourth-order Runge-Kutta method with a fixed step.
    Choose ``dt`` so that ``dt * max(|omega| + K)`` stays well below 1.

    Parameters
    ----------
    k_values : np.ndarray
        Coupling strengths, shape (num_k,).
    omega : np.ndarray
        Natural frequencies, shape (N,).
    theta0 : np.ndarray
        Initial phases, shape (N,). The same initial condition is used for
        every K.
    t_end : float, optional
        Final time, by default 100.
    dt : float, optional
        Time step, by default 0.05.
    t_average : float, optional
        Length of the final window over which r(t) is averaged, by default 50.

    Returns
    -------
    r_mean : np.ndarray
        Mean of r(t) over the averaging window, shape (num_k,).
    r_std : np.ndarray
        Standard deviation of r(t) over the same window, shape (num_k,).
    """
    k = np.asarray(k_values, dtype=float)
    omega = np.asarray(omega, dtype=float)
    theta = np.tile(np.asarray(theta0, dtype=float), (k.size, 1))
    num_steps = int(round(t_end / dt))
    start_avg = num_steps - int(round(t_average / dt))
    r_hist = []
    for n in range(num_steps):
        k1 = _rhs(theta, omega, k)
        k2 = _rhs(theta + 0.5 * dt * k1, omega, k)
        k3 = _rhs(theta + 0.5 * dt * k2, omega, k)
        k4 = _rhs(theta + dt * k3, omega, k)
        theta = theta + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if n >= start_avg:
            r_hist.append(np.hypot(np.cos(theta).mean(axis=1), np.sin(theta).mean(axis=1)))
    r_hist = np.array(r_hist)
    return r_hist.mean(axis=0), r_hist.std(axis=0)


def critical_coupling(g0: float) -> float:
    """Kuramoto critical coupling K_c = 2 / (pi g(0)) (Strogatz 2000).

    Parameters
    ----------
    g0 : float
        Value at zero of the (symmetric, unimodal) frequency density.

    Returns
    -------
    float
        Critical coupling strength.
    """
    return 2.0 / (np.pi * g0)
