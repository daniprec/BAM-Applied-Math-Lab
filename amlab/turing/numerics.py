"""Explicit Euler bounds and a compact 1D Gierer-Meinhardt solver.

These helpers support the time-step stability and convergence checks of the
reaction-diffusion laboratories. See LeVeque (2007), Chapter 9, for the
stability analysis of explicit schemes for the diffusion equation.
"""

from typing import Callable, Optional, Tuple

import numpy as np


def euler_dt_max(dx: float, diffusion: float, ndim: int = 1) -> float:
    """Largest stable explicit Euler step for pure diffusion.

    For the centered second-difference Laplacian (3-point stencil in 1D,
    5-point stencil in 2D) the forward Euler scheme is stable when

        dt <= dx**2 / (2 * ndim * diffusion).

    Parameters
    ----------
    dx : float
        Grid spacing.
    diffusion : float
        Largest diffusion coefficient in the system.
    ndim : int
        Number of space dimensions.

    Returns
    -------
    float
        The bound on dt.
    """
    return dx**2 / (2 * ndim * diffusion)


def _laplacian_1d(uv: np.ndarray, dx: float) -> np.ndarray:
    """Three-point Laplacian with periodic wrapping via np.roll."""
    lap = -2 * uv + np.roll(uv, 1, axis=-1) + np.roll(uv, -1, axis=-1)
    return lap / dx**2


def simulate_gm_1d(
    a: float = 0.40,
    b: float = 1.00,
    d: float = 30.0,
    gamma: float = 1.0,
    length: float = 40.0,
    dx: float = 1.0,
    dt: float = 0.01,
    t_end: float = 100.0,
    bc: str = "periodic",
    initial: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the 1D Gierer-Meinhardt model with explicit Euler.

    Parameters
    ----------
    a, b, d, gamma : float
        Model parameters in the course scaling.
    length : float
        Domain length L. The grid has N = round(L / dx) points.
    dx, dt : float
        Grid spacing and time step.
    t_end : float
        Final time. The number of steps is round(t_end / dt).
    bc : str
        'periodic' (grid x_i = i dx) or 'neumann' (copy the nearest
        interior value after every step).
    initial : callable, optional
        Function of the grid x that returns the initial perturbation of u
        and v, shape (2, N). If None, 1% Gaussian noise with the given seed.
    seed : int
        Seed for the default noisy initial condition.

    Returns
    -------
    x : np.ndarray
        Grid points, shape (N,).
    uv : np.ndarray
        Final state, shape (2, N). Contains NaN or inf if the scheme blew up.
    """
    num_points = int(round(length / dx))
    x = np.arange(num_points) * dx
    u_star = (a + 1) / b
    base = np.array([[u_star], [u_star**2]]) * np.ones((2, num_points))
    if initial is None:
        rng = np.random.default_rng(seed)
        uv = base * (1 + 0.01 * rng.standard_normal((2, num_points)))
    else:
        uv = base * (1 + initial(x))

    num_steps = int(round(t_end / dt))
    with np.errstate(all="ignore"):
        for _ in range(num_steps):
            u, v = uv
            lap = _laplacian_1d(uv, dx)
            f = a - b * u + u**2 / v
            g = u**2 - v
            uv = uv + dt * np.array([lap[0] + gamma * f, d * lap[1] + gamma * g])
            if bc == "neumann":
                uv[:, 0] = uv[:, 1]
                uv[:, -1] = uv[:, -2]
            elif bc != "periodic":
                raise ValueError("bc must be 'periodic' or 'neumann'")
            if not np.all(np.isfinite(uv)):
                break
    return x, uv
