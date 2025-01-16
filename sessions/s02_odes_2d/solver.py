from typing import Any, Callable, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def solve_ode(
    system_func: Callable,
    y0: np.ndarray,
    t_step: float = 1.0,
    t_show: float = 50.0,
    solver: str = "rk",
    **kwargs: Any,
) -> np.ndarray:
    t_eval = np.arange(0, t_show + t_step, t_step)
    if "eu" in solver.lower():
        return solve_ode_euler(system_func, y0, t_eval, **kwargs)
    elif "rk" in solver.lower():
        return solve_ode_rk(system_func, y0, t_eval, **kwargs)
    else:
        raise ValueError(f"Invalid solver: {solver}")


def solve_ode_euler(
    system_func: Callable,
    y0: np.ndarray,
    t_eval: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    """
    Performs the numerical integration of the ODEs.
    The function uses the Runge-Kutta method.

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

    # Solve an initial value problem for a system of ODEs
    ls_y = [y0.copy()]
    for idx in range(1, len(t_eval)):
        y0 = y0 + func(t_eval[idx], y0) * (t_eval[idx] - t_eval[idx - 1])
        ls_y.append(y0.copy())
    return np.stack(ls_y, axis=-1)


def solve_ode_rk(
    system_func: Callable,
    y0: np.ndarray,
    t_eval: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    """
    Performs the numerical integration of the ODEs.
    The function uses the Runge-Kutta method.

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

    t_span = (t_eval[0], t_eval[-1])
    # Solve an initial value problem for a system of ODEs
    sol = solve_ivp(func, t_span, y0, t_eval=t_eval, method="RK45")
    return sol.y


def compute_fixed_points(
    system_func: Callable,
    t: float = 0.0,
    **kwargs: Any,
) -> np.ndarray:
    """
    Computes the fixed points of a 2D dynamical system.

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
        # Find the roots of the (non-linear) equations defined by func(x) = 0
        # given a starting estimate (guess)
        fixed_point, info, ier, mesg = fsolve(func, guess, full_output=True)
        # ier: An integer flag. Set to 1 if a solution was found
        if ier == 1:
            # Check for duplicates
            if not any(np.allclose(fixed_point, fp, atol=1e-5) for fp in fixed_points):
                fixed_points.append(fixed_point)
    return np.array(fixed_points)


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

    # Create a grid of points
    x_values = np.linspace(x_min, x_max, num_points)
    y_values = np.linspace(y_min, y_max, num_points)
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    # Evaluate the derivatives at each point
    dx_dt: np.ndarray
    dy_dt: np.ndarray
    dx_dt, dy_dt = system_func(t, [x_grid, y_grid], **kwargs)

    # Extract nullcline data - Find where dx_dt changes sign (zero crossings)
    zero_crossings = np.where(np.diff(np.sign(dx_dt), axis=0))
    x_nc_x = x_grid[zero_crossings]
    x_nc_y = y_grid[zero_crossings]
    # Extract nullcline data - Find where dy_dt changes sign (zero crossings)
    zero_crossings = np.where(np.diff(np.sign(dy_dt), axis=1))
    y_nc_x = x_grid[zero_crossings]
    y_nc_y = y_grid[zero_crossings]

    return [(x_nc_x, x_nc_y), (y_nc_x, y_nc_y)]


def laplacian(uv: np.ndarray) -> np.ndarray:
    """
    Compute the Laplacian of the u and v fields using a 5-point finite
    difference scheme: considering each point and its immediate neighbors in
    the up, down, left, and right directions.

    Reference: https://en.wikipedia.org/wiki/Five-point_stencil

    Parameters
    ----------
    uv : np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the values of
        u and v.
    Returns
    -------
    np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the Laplacian
        of u and v.
    """
    lap = -4 * uv

    # Immediate neighbors (up, down, left, right)
    lap += np.roll(uv, shift=1, axis=0)  # up
    lap += np.roll(uv, shift=-1, axis=0)  # down
    lap += np.roll(uv, shift=1, axis=1)  # left
    lap += np.roll(uv, shift=-1, axis=1)  # right
    return lap


def laplacian_9pt(uv: np.ndarray) -> np.ndarray:
    """
    Compute the Laplacian of the u and v fields using a 9-point finite
    difference scheme (Patra-Karttunen), considering each point and its
    immediate neighbors, including diagonals.

    Reference: https://en.wikipedia.org/wiki/Nine-point_stencil

    Parameters
    ----------
    uv : np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the values of u and v.

    Returns
    -------
    np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the Laplacian of u and v.
    """
    # Weights for the 9-point stencil (Patra-Karttunen)
    center_weight = -20 / 6
    neighbor_weight = 4 / 6
    diagonal_weight = 1 / 6

    lap = center_weight * uv

    # Shifted arrays for immediate neighbors
    up = np.roll(uv, shift=1, axis=0)
    down = np.roll(uv, shift=-1, axis=0)

    # Immediate neighbors (up, down, left, right)
    lap += neighbor_weight * up  # up
    lap += neighbor_weight * down  # down
    lap += neighbor_weight * np.roll(uv, shift=1, axis=1)  # left
    lap += neighbor_weight * np.roll(uv, shift=-1, axis=1)  # right

    # Diagonal neighbors
    lap += diagonal_weight * np.roll(up, shift=1, axis=1)  # up-left
    lap += diagonal_weight * np.roll(up, shift=-1, axis=1)  # up-right
    lap += diagonal_weight * np.roll(down, shift=1, axis=1)  # down-left
    lap += diagonal_weight * np.roll(down, shift=-1, axis=1)  # down-right

    return lap
