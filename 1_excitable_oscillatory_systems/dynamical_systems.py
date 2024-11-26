# dynamical_systems.py

from typing import Any, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp


def simulate(
    system_func: Any,
    y0: List[float],
    t_span: Tuple[float, float],
    t_eval: np.ndarray,
    *args
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


def animate(i: int, y: np.ndarray, line: Any) -> Tuple[Any]:
    """
    Updates the line object for each frame in the animation.

    Parameters
    ----------
    i : int
        Frame index.
    y : ndarray
        Array of solution values.
    line : Line2D
        Line object to update.

    Returns
    -------
    line : tuple of Line2D
        Updated line object.
    """
    line.set_data(y[0][:i], y[1][:i])
    return (line,)


def update_simulation(
    event: Any,
    system_func: Any,
    t_span: Tuple[float, float],
    t_eval: np.ndarray,
    line: Any,
    ani: Any,
    *args
) -> None:
    """
    Updates the simulation with new initial conditions from a mouse click.

    Parameters
    ----------
    event : MouseEvent
        Matplotlib mouse event.
    system_func : callable
        Function defining the system of ODEs.
    t_span : tuple of float
        Tuple containing the start and end times (t0, tf).
    t_eval : ndarray
        Time points at which to store the computed solutions.
    line : Line2D
        Line object to update.
    ani : FuncAnimation
        Animation object to update.
    *args
        Additional arguments to pass to the system function.

    Returns
    -------
    None
    """
    y0 = [event.xdata, event.ydata]
    if None in y0:
        return
    y = simulate(system_func, y0, t_span, t_eval, *args)
    ani.event_source.stop()
    ani.new_frame_seq()
    ani.frame_seq = ani.new_frame_seq()
    ani._args = (y, line)
    ani.event_source.start()
    ani._args = (y, line)
    ani.event_source.start()
