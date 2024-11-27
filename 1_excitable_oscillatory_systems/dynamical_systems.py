from typing import Any, Callable, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def simulate(
    system_func: Callable,
    y0: List[float],
    t_span: Tuple[float, float],
    t_eval: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    """
    Performs the numerical integration of the ODEs.

    Parameters
    ----------
    system_func : Callable
        Function defining the system of ODEs.
    y0 : list of float
        Initial conditions.
    t_span : tuple of float
        Tuple containing the start and end times (t0, tf).
    t_eval : ndarray
        Time points at which to store the computed solutions.
    **kwargs
        Additional keyword arguments to pass to the system function.

    Returns
    -------
    y : ndarray
        Array of solution values at t_eval.
    """

    def func(t: float, y: np.ndarray) -> np.ndarray:
        return system_func(t, y, **kwargs)

    sol = solve_ivp(func, t_span, y0, t_eval=t_eval)
    return sol.y


def compute_fixed_points(
    system_func: Callable,
    t: float = 0.0,
    **kwargs: Any,
) -> np.ndarray:
    """
    Computes the fixed points of a 2D dynamical system for a given external current.

    Parameters
    ----------
    system_func : Callable
        Function defining the system of ODEs.
    t : float, optional
        Time variable (default is 0.0).
    **kwargs
        Additional keyword arguments to pass to the system function.

    Returns
    -------
    fixed_points : ndarray
        Array of fixed points [x*, y*].
    """

    def func(y: np.ndarray) -> np.ndarray:
        return system_func(t, y, **kwargs)

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


def extract_nullcline(
    x: np.ndarray, y: np.ndarray, f: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts nullcline points where the function crosses zero.

    Parameters
    ----------
    x : ndarray
        Array of x values.
    y : ndarray
        Array of y values.
    f : ndarray
        Array of function values corresponding to (x, y).

    Returns
    -------
    x_nc : ndarray
        Array of x values where f crosses zero.
    y_nc : ndarray
        Array of y values where f crosses zero.
    """
    # Find where f changes sign (zero crossings)
    zero_crossings = np.where(np.diff(np.sign(f), axis=0))
    x_nc = x[zero_crossings]
    y_nc = y[zero_crossings]
    return x_nc, y_nc


def compute_nullclines(
    system_func: Callable,
    t: float = 0.0,
    limits: Tuple[float, float, float, float] = (-3.0, -3.0, 3.0, 3.0),
    num_points: int = 1000,
    **kwargs: Any,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Computes the nullclines for a 2D dynamical system without plotting.

    Parameters
    ----------
    system_func : Callable
        Function defining the system of ODEs.
    t : float, optional
        Time variable (default is 0.0).
    limits : tuple of float, optional
        Tuple containing the x and y limits (x_min, y_min, x_max, y_max).
    num_points : int, optional
        Number of points to use in each variable range (default is 1000).
    **kwargs
        Additional keyword arguments to pass to the system function.

    Returns
    -------
    nullclines : list of tuple of ndarray
        List containing tuples of x and y coordinates of the nullclines:
        - nullclines[0]: (x values, y values) where dx/dt = 0.
        - nullclines[1]: (x values, y values) where dy/dt = 0.
    """
    x_min, y_min, x_max, y_max = limits

    x_values = np.linspace(x_min, x_max, num_points)
    y_values = np.linspace(y_min, y_max, num_points)

    x_grid, y_grid = np.meshgrid(x_values, y_values)

    # Flatten the arrays to compute derivatives at each point
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    dx_dt = np.zeros_like(x_flat)
    dy_dt = np.zeros_like(y_flat)

    # Evaluate the derivatives at each point
    for idx, (xi, yi) in enumerate(zip(x_flat, y_flat)):
        dydt = system_func(t, [xi, yi], **kwargs)
        dx_dt[idx] = dydt[0]
        dy_dt[idx] = dydt[1]

    # Reshape to the grid shape
    dx_dt_grid = dx_dt.reshape(x_grid.shape)
    dy_dt_grid = dy_dt.reshape(y_grid.shape)

    # Extract nullcline data
    x_nc_x, x_nc_y = extract_nullcline(x_grid, y_grid, dx_dt_grid)
    y_nc_x, y_nc_y = extract_nullcline(x_grid, y_grid, dy_dt_grid)

    return [(x_nc_x, x_nc_y), (y_nc_x, y_nc_y)]
