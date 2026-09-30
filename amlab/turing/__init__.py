"""Linear Turing analysis and numerical checks for reaction-diffusion models.

The notation follows Murray, Mathematical Biology II, Chapter 2, and the
scaling used across the course:

    u_t = Laplacian(u) + gamma * f(u, v)
    v_t = d * Laplacian(v) + gamma * g(u, v)

Modules
-------
linear
    Steady state, Jacobian, Turing conditions, dispersion relation and
    Laplacian eigenvalues for the Gierer-Meinhardt model.
numerics
    Explicit Euler time-step bounds and a small 1D solver used for
    stability and convergence checks.
"""

from .linear import (
    critical_diffusion_ratio,
    dispersion_relation,
    gierer_meinhardt_jacobian,
    gierer_meinhardt_steady_state,
    laplacian_eigenvalues_1d,
    turing_conditions,
    unstable_band,
    unstable_modes_1d,
    unstable_modes_2d,
)
from .numerics import euler_dt_max, simulate_gm_1d

__all__ = [
    "critical_diffusion_ratio",
    "dispersion_relation",
    "euler_dt_max",
    "gierer_meinhardt_jacobian",
    "gierer_meinhardt_steady_state",
    "laplacian_eigenvalues_1d",
    "simulate_gm_1d",
    "turing_conditions",
    "unstable_band",
    "unstable_modes_1d",
    "unstable_modes_2d",
]
