from typing import Any, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def simulate(
    system_func: Any,
    y0: List[float],
    t_span: Tuple[float, float],
    t_eval: np.ndarray,
    *args,
) -> np.ndarray:
    """
    Performs the numerical integration of the ODEs.

    Parameters
    ----------
    system_func : callable
        Function defining the system of ODEs.
    y0 : list of float
        Initial conditions.
    t_span : tuple of float
        Tuple containing the start and end times (t0, tf).
    t_eval : ndarray
        Time points at which to store the computed solutions.
    *args
        Additional arguments to pass to the system function.

    Returns
    -------
    y : ndarray
        Array of solution values at t_eval.
    """
    sol = solve_ivp(system_func, t_span, y0, args=args, t_eval=t_eval)
    return sol.y


def compute_fixed_points(equations: callable) -> np.ndarray:
    """
    Computes the fixed points of the FitzHugh-Nagumo model for a given external current.

    Parameters
    ----------
    i_ext : float
        External stimulus current.

    Returns
    -------
    fixed_points : ndarray
        Array of fixed points [v*, w*].
    """

    # Initial guesses for fixed points
    guesses = [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)]
    fixed_points = []
    for guess in guesses:
        fixed_point, info, ier, mesg = fsolve(equations, guess, full_output=True)
        if ier == 1:
            # Check for duplicates
            if not any(np.allclose(fixed_point, fp, atol=1e-5) for fp in fixed_points):
                fixed_points.append(fixed_point)
    return np.array(fixed_points)
