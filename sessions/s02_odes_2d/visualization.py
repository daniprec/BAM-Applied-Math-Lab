import sys
from typing import Any, Callable, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import MouseEvent
from matplotlib.lines import Line2D

# Add the path to the sys module
# (allowing the import of the utils module)
sys.path.append(".")

from sessions.s02_odes_2d.solver import (
    compute_fixed_points,
    compute_nullclines,
    solve_ode,
)


def update_simulation(
    event: MouseEvent,
    system_func: Callable,
    line: Line2D,
    ani: FuncAnimation,
    *args: Any,
    **kwargs: Any,
):
    """
    Updates the simulation with new initial conditions from a mouse click.

    Parameters
    ----------
    event : MouseEvent
        Matplotlib mouse event.
    system_func : Callable
        Function defining the system of ODEs.
    line : Line2D
        Line object to update.
    ani : FuncAnimation
        Animation object to update.
    *args
        Additional arguments to pass to the system function.
    **kwargs
        Additional keyword arguments to pass to the system function.
    """
    y0 = [event.xdata, event.ydata]
    if None in y0:
        return
    y = solve_ode(system_func, y0, *args, **kwargs)
    ani.event_source.stop()
    ani.new_frame_seq()
    ani.frame_seq = ani.new_frame_seq()
    ani._args = (y, line)
    ani.event_source.start()


def plot_phase_plane(
    system_func: Callable,
    limits: Tuple[float, float, float, float] = (-3.0, -3.0, 3.0, 3.0),
    ax: Optional[plt.Axes] = None,
    t_step: float = 0.0,
    t_show: float = 0.0,
    solver: str = "rk",
    **kwargs: Any,
):
    """
    Plots the phase plane of any excitable-oscillatory model.

    Parameters
    ----------
    system_func : Callable
        Function that defines the model equations.
    limits : tuple of float, optional
        Tuple containing the x and y limits (x_min, y_min, x_max, y_max).
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses current axes.
    **kwargs
        Additional keyword arguments to pass to the system function.
    """
    # Set up the axes if not provided
    if ax is None:
        ax = plt.gca()

    # Create a grid of points
    v_values = np.linspace(limits[0], limits[2], 20)
    w_values = np.linspace(limits[1], limits[3], 20)
    v_grid, w_grid = np.meshgrid(v_values, w_values)

    # Compute derivatives
    dvdt, dwdt = system_func(0.0, [v_grid, w_grid], **kwargs)

    # Plot vector field
    ax.quiver(v_grid, w_grid, dvdt, dwdt, color="gray", alpha=0.5)

    # Compute nullclines
    nullclines = compute_nullclines(system_func, t=0.0, limits=limits, **kwargs)
    v_nullcline = nullclines[0]
    w_nullcline = nullclines[1]

    # Plot nullclines
    ax.scatter(v_nullcline[0], v_nullcline[1], c="b", s=1, label="dv/dt = 0 Nullcline")
    ax.scatter(w_nullcline[0], w_nullcline[1], c="r", s=1, label="dw/dt = 0 Nullcline")

    # Compute and plot fixed points
    fixed_points = compute_fixed_points(system_func, t=0.0, **kwargs)
    for fp in fixed_points:
        ax.plot(fp[0], fp[1], "ko", markersize=8)
        ax.text(fp[0] + 0.1, fp[1] + 0.1, f"({fp[0]:.2f}, {fp[1]:.2f})")


def plot_bifurcation(
    system_func: Callable,
    param_name: str,
    param_limits: Tuple[float, float],
    y0: List[float],
    ax: Optional[plt.Axes] = None,
    **kwargs: Any,
):
    """
    Plots the bifurcation diagram by varying a specified parameter.

    Parameters
    ----------
    system_func : Callable
        Function that defines the model equations.
    param_name : str
        Name of the parameter to vary.
    param_values : ndarray
        Array of parameter values.
    y0 : list of float
        Initial conditions.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses current axes.
    **kwargs
        Additional keyword arguments to pass to the system function.
    """
    if ax is None:
        ax = plt.gca()

    max_v = []
    min_v = []

    param_values = np.linspace(*param_limits, 100)
    for param_value in param_values:
        kwargs[param_name] = param_value
        y = solve_ode(system_func, y0, **kwargs)
        v = y[0, -int(y.shape[-1] / 2) :]  # Last half of data
        max_v.append(np.max(v))
        min_v.append(np.min(v))

    ax.plot(param_values, max_v, "r", label="Max v")
    ax.plot(param_values, min_v, "b", label="Min v")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Membrane Potential v")
    ax.set_title("Bifurcation Diagram")
    ax.legend()
    ax.grid(True)


def run_interactive_plot(
    system_func: Callable,
    v0: float = 0.0,
    w0: float = 0.0,
    limits: Tuple[float, float, float, float] = (-3.0, -3.0, 3.0, 3.0),
    param_name: Optional[str] = None,
    param_limits: Tuple[float, float] = (0.0, 2.0),
    **kwargs: Any,
):
    """
    Runs an interactive simulation of a dynamical system with the ability to update initial conditions.

    Parameters
    ----------
    system_func : Callable
        Function that defines the model equations.
    v0 : float, optional
        Initial value of the first variable (default is 0.0).
    w0 : float, optional
        Initial value of the second variable (default is 0.0).
    limits : tuple of float, optional
        Tuple containing the x and y limits (x_min, y_min, x_max, y_max).
    param_name : str, optional
        Name of the parameter to vary in bifurcation diagram.
    param_values : ndarray, optional
        Array of parameter values for bifurcation diagram.
    **kwargs
        Additional keyword arguments to pass to the system function.
    """
    y0: List[float] = [v0, w0]

    # Initial simulation
    y = solve_ode(system_func, y0, **kwargs)

    # Determine the number of subplots
    if param_name and param_limits is not None:
        size = 2
    else:
        size = 1

    # Set up the figure with subplots
    fig, (ax_phase, ax_bifurcation) = plt.subplots(1, size, figsize=(8 * size, 6))

    # Plot phase plane
    plot_phase_plane(system_func, limits=limits, ax=ax_phase, **kwargs)
    # Set up the plot parameters
    ax_phase.set_xlabel("Membrane Potential (v)")
    ax_phase.set_ylabel("Recovery Variable (w)")
    ax_phase.set_title("Phase Plane Analysis")
    ax_phase.legend()
    ax_phase.set_xlim(limits[0], limits[2])
    ax_phase.set_ylim(limits[1], limits[3])
    ax_phase.grid(True)

    # Plot bifurcation diagram if parameters are provided
    if param_name and param_limits is not None:
        plot_bifurcation(
            system_func,
            param_name=param_name,
            param_limits=param_limits,
            y0=y0,
            ax=ax_bifurcation,
            **kwargs,
        )

    # Initialize the line object for animation on phase plane
    (line,) = ax_phase.plot([], [], lw=2)

    def animate(i: int, y: np.ndarray, line: Line2D) -> Tuple[Line2D]:
        """
        Animation function to update the line object.
        """
        line.set_data(y[0][:i], y[1][:i])
        return (line,)

    # Create the animation
    ani = animation.FuncAnimation(fig, animate, fargs=(y, line), interval=20, blit=True)

    # Connect the click event to the update function
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda event: update_simulation(event, system_func, line, ani, **kwargs),
    )

    # Show the interactive plot
    plt.show()
