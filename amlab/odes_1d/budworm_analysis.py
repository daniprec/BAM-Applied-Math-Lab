"""Analysis tools for the spruce budworm model.

The dimensionless model is (Strogatz 2024, Section 3.7; Ludwig, Jones and
Holling 1978, eqn 9)

    dx/dtau = r x (1 - x / k) - x**2 / (1 + x**2).

This module collects equilibria, their stability, the saddle-node
bifurcation curves in the (k, r) plane, a quasi-static sweep used to draw
hysteresis loops, and the three-variable budworm-forest model of Ludwig,
Jones and Holling (1978), eqns (3), (10), (11), (13), (14).
"""

import numpy as np

from amlab.odes_1d.spruce_budworm import spruce_budworm


def budworm_equilibria(r: float, k: float) -> np.ndarray:
    """Return the positive equilibria of the dimensionless budworm model.

    Nonzero equilibria satisfy r (1 - x/k) (1 + x^2) - x = 0, which is the
    cubic  -(r/k) x^3 + r x^2 - (1 + r/k) x + r = 0.

    Parameters
    ----------
    r : float
        Dimensionless growth rate.
    k : float
        Dimensionless carrying capacity.

    Returns
    -------
    np.ndarray
        Sorted array with the one or three positive real roots.
    """
    coeffs = [-r / k, r, -(1 + r / k), r]
    roots = np.roots(coeffs)
    real = roots[np.abs(roots.imag) < 1e-9].real
    return np.sort(real[real > 0])


def budworm_slope(x: np.ndarray, r: float, k: float) -> np.ndarray:
    """Return f'(x) for f(x) = r x (1 - x/k) - x^2 / (1 + x^2).

    A negative value at an equilibrium means it is stable, a positive
    value means it is unstable.
    """
    x = np.asarray(x, dtype=float)
    return r * (1 - 2 * x / k) - 2 * x / (1 + x**2) ** 2


def bifurcation_curves(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parametric saddle-node curves of the budworm model.

    For x > 1 the pair (k(x), r(x)) with

        k = 2 x^3 / (x^2 - 1),   r = 2 x^3 / (1 + x^2)^2

    is a point where two equilibria collide (Strogatz 2024, Section 3.7).
    The two branches meet at the cusp x = sqrt(3), (k, r) = (3 sqrt(3),
    3 sqrt(3) / 8).

    Parameters
    ----------
    x : np.ndarray
        Values of the double root, x > 1.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays k(x) and r(x).
    """
    x = np.asarray(x, dtype=float)
    k = 2 * x**3 / (x**2 - 1)
    r = 2 * x**3 / (1 + x**2) ** 2
    return k, r


def quasi_static_sweep(
    r: float, k_values: np.ndarray, x0: float, t_relax: float = 200.0
) -> np.ndarray:
    """Follow the attracting equilibrium while k changes very slowly.

    For each value of k, the model relaxes for a time t_relax starting from
    the state reached with the previous value. The result approximates the
    branch the system follows when k drifts slowly compared with the
    budworm dynamics.

    Parameters
    ----------
    r : float
        Dimensionless growth rate.
    k_values : np.ndarray
        Sequence of carrying capacities, in the order they are visited.
    x0 : float
        Initial population.
    t_relax : float, optional
        Relaxation time at each value of k, by default 200.

    Returns
    -------
    np.ndarray
        Population reached at each k.
    """
    from scipy.integrate import solve_ivp

    x = x0
    out = np.empty(len(k_values))
    for i, k in enumerate(k_values):
        sol = solve_ivp(spruce_budworm, (0, t_relax), [x], args=(r, k), rtol=1e-8)
        x = sol.y[0, -1]
        out[i] = x
    return out


# Level III parameters of Ludwig, Jones and Holling (1978), Table 1 and text.
# K' = 335 larvae/branch is the value used in the text; it gives Q = 302 and
# M = 0.71 as stated there.
LUDWIG_LEVEL_III = {
    "r_b": 1.52,  # budworm intrinsic growth rate (1/year)
    "k_prime": 335.0,  # budworm carrying capacity (larvae/branch)
    "beta": 43200.0,  # maximum predation (larvae/acre/year)
    "alpha_prime": 1.11,  # half-saturation density (larvae/branch)
    "r_s": 0.095,  # branch growth rate (1/year)
    "k_s": 25440.0,  # maximum branch density (branches/acre)
    "k_e": 1.0,  # maximum energy reserve
    "r_e": 0.92,  # energy reserve growth rate (1/year)
    "p": 0.00195,  # consumption rate of E (per larva)
}


def ludwig_forest_model(t: float, y: np.ndarray, params: dict = None) -> list:
    """Budworm, branch area and energy reserve (Ludwig et al. 1978).

    dB/dt = r_b B (1 - B / (K' S)) - beta B^2 / ((alpha' S)^2 + B^2)
    dS/dt = r_s S (1 - (S / K_s) (K_e / E))
    dE/dt = r_e E (1 - E / K_e) - P B / S

    Parameters
    ----------
    t : float
        Time in years (unused, required by solve_ivp).
    y : np.ndarray
        State (B, S, E): larvae per acre, branches per acre, energy reserve.
    params : dict, optional
        Parameter dictionary, by default LUDWIG_LEVEL_III.

    Returns
    -------
    list
        Derivatives [dB/dt, dS/dt, dE/dt].
    """
    p = LUDWIG_LEVEL_III if params is None else params
    b, s, e = y
    db = p["r_b"] * b * (1 - b / (p["k_prime"] * s)) - p["beta"] * b**2 / (
        (p["alpha_prime"] * s) ** 2 + b**2
    )
    ds = p["r_s"] * s * (1 - (s / p["k_s"]) * (p["k_e"] / e))
    de = p["r_e"] * e * (1 - e / p["k_e"]) - p["p"] * b / s
    return [db, ds, de]
