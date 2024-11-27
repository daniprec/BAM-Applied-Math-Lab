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


def compute_fixed_points(
    equations: callable, t: float = 0, i_ext: float = 0.5, kwargs: dict = None
) -> np.ndarray:
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
    kwargs = kwargs if isinstance(kwargs, dict) else {}

    def func(y: np.ndarray) -> np.ndarray:
        return equations(t, y, i_ext, **kwargs)

    # Initial guesses for fixed points
    guesses = [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)]
    fixed_points = []
    for guess in guesses:
        fixed_point, info, ier, mesg = fsolve(func, guess, full_output=True)
        if ier == 1:
            # Check for duplicates
            if not any(np.allclose(fixed_point, fp, atol=1e-5) for fp in fixed_points):
                fixed_points.append(fixed_point)
    return np.array(fixed_points)


def extract_nullcline(x: np.ndarray, y: np.ndarray, f: np.ndarray) -> Tuple[np.ndarray]:
    """Find where dx/dt = 0 and dy/dt = 0
    We use contour levels at zero to find nullclines
    Function to extract nullcline points where derivative crosses zero

    Parameters
    ----------
    x : np.ndarray
        Array of x values.
    y : np.ndarray
        Array of y values.
    f : np.ndarray
        Array of function values.
    Returns
    -------
    x_nc : np.ndarray
        Array of x values where f crosses zero.
    y_nc : np.ndarray
        Array of y values where f crosses zero.
    """
    # Find where f changes sign (zero crossings)
    zero_crossings = np.where(np.diff(np.sign(f), axis=0))
    x_nc = x[zero_crossings]
    y_nc = y[zero_crossings]
    return x_nc, y_nc


def compute_nullclines(
    equations: callable,
    t: float = 0,
    i_ext: float = 0.5,
    limits: tuple[float] = (-3, -3, 3, 3),
    num_points: int = 1000,
    kwargs: dict = None,
) -> List[Tuple[np.ndarray]]:
    """
    Computes the nullclines for a 2D dynamical system without plotting.

    Parameters
    ----------
    equations : callable
        Function defining the system of ODEs. Should accept arguments (t, y, *args)
        and return [dx/dt, dy/dt].
    limits : tuple, optional
        Tuple containing the x and y limits (x_min, y_min, x_max, y_max).
    args : tuple, optional
        Additional arguments to pass to the system function.
    num_points : int, optional
        Number of points to use in each variable range.

    Returns
    -------
    nullclines : list[tuple[np.ndarray]]
        List of tuples containing the x and y coordinates of the nullclines.
    """

    x_min, y_min, x_max, y_max = limits
    kwargs = kwargs if isinstance(kwargs, dict) else {}

    x_values = np.linspace(x_min, x_max, num_points)
    y_values = np.linspace(y_min, y_max, num_points)

    x, y = np.meshgrid(x_values, y_values)

    # Flatten the arrays to compute derivatives at each point
    x_flat = x.flatten()
    y_flat = y.flatten()
    dx_dt = np.zeros_like(x_flat)
    dy_dt = np.zeros_like(y_flat)

    # Evaluate the derivatives at each point
    for idx, (xi, yi) in enumerate(zip(x_flat, y_flat)):
        dydt = equations(t, [xi, yi], i_ext, **kwargs)
        dx_dt[idx] = dydt[0]
        dy_dt[idx] = dydt[1]

    # Reshape to the grid shape
    dx_dt = dx_dt.reshape(x.shape)
    dy_dt = dy_dt.reshape(y.shape)

    # Extract nullcline data
    x_nc_x, x_nc_y = extract_nullcline(x, y, dx_dt)
    y_nc_x, y_nc_y = extract_nullcline(x, y, dy_dt)

    return [(x_nc_x, x_nc_y), (y_nc_x, y_nc_y)]
