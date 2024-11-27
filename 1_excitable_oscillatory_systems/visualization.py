from typing import Any, Callable, List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from dynamical_systems import compute_fixed_points, compute_nullclines, simulate
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import MouseEvent
from matplotlib.lines import Line2D


def animate(i: int, y: np.ndarray, line: Line2D) -> Tuple[Line2D]:
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
    event: MouseEvent,
    system_func: Callable,
    t_span: Tuple[float, float],
    t_eval: np.ndarray,
    line: Line2D,
    ani: FuncAnimation,
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Updates the simulation with new initial conditions from a mouse click.

    Parameters
    ----------
    event : MouseEvent
        Matplotlib mouse event.
    system_func : Callable
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
    **kwargs
        Additional keyword arguments to pass to the system function.

    Returns
    -------
    None
    """
    y0 = [event.xdata, event.ydata]
    if None in y0:
        return
    y = simulate(system_func, y0, t_span, t_eval, *args, **kwargs)
    ani.event_source.stop()
    ani.new_frame_seq()
    ani.frame_seq = ani.new_frame_seq()
    ani._args = (y, line)
    ani.event_source.start()


def plot_phase_plane(
    equations: Callable,
    limits: Tuple[float, float, float, float] = (-3.0, -3.0, 3.0, 3.0),
    **kwargs: Any,
) -> None:
    """
    Plots the phase plane of any excitable-oscillatory model.

    Parameters
    ----------
    equations : Callable
        Function that defines the model equations.
    limits : tuple of float, optional
        Tuple containing the x and y limits (x_min, y_min, x_max, y_max).
    **kwargs
        Additional keyword arguments to pass to the system function.

    Returns
    -------
    None
    """
    # Create a grid of points
    v_values = np.linspace(limits[0], limits[2], 20)
    w_values = np.linspace(limits[1], limits[3], 20)
    v_grid, w_grid = np.meshgrid(v_values, w_values)

    # Compute derivatives
    dvdt, dwdt = equations(0.0, [v_grid, w_grid], **kwargs)

    # Plot vector field
    plt.quiver(v_grid, w_grid, dvdt, dwdt, color="gray", alpha=0.5)

    # Compute nullclines
    nullclines = compute_nullclines(equations, t=0.0, limits=limits, **kwargs)
    v_nullcline = nullclines[0]
    w_nullcline = nullclines[1]

    # Plot nullclines
    plt.scatter(v_nullcline[0], v_nullcline[1], c="b", s=1, label="dv/dt = 0 Nullcline")
    plt.scatter(w_nullcline[0], w_nullcline[1], c="r", s=1, label="dw/dt = 0 Nullcline")

    # Compute and plot fixed points
    fixed_points = compute_fixed_points(equations, t=0.0, **kwargs)
    for fp in fixed_points:
        plt.plot(fp[0], fp[1], "ko", markersize=8)
        plt.text(fp[0] + 0.1, fp[1] + 0.1, f"({fp[0]:.2f}, {fp[1]:.2f})")

    plt.xlabel("Membrane Potential (v)")
    plt.ylabel("Recovery Variable (w)")
    plt.title("Phase Plane Analysis")
    plt.legend()
    plt.xlim(limits[0], limits[2])
    plt.ylim(limits[1], limits[3])
    plt.grid(True)


def run_interactive_plot(
    equations: Callable,
    t_end: float = 100.0,
    num_points: int = 1000,
    v0: float = 0.0,
    w0: float = 0.0,
    limits: Tuple[float, float, float, float] = (-3.0, -3.0, 3.0, 3.0),
    **kwargs: Any,
) -> None:
    """
    Runs an interactive simulation of a dynamical system with the ability to update initial conditions.

    Parameters
    ----------
    equations : Callable
        Function that defines the model equations.
    t_span : float, optional
        End time for the simulation (default is 100.0).
    num_points : int, optional
        Number of time points to evaluate (default is 1000).
    v0 : float, optional
        Initial value of the first variable (default is 0.0).
    w0 : float, optional
        Initial value of the second variable (default is 0.0).
    limits : tuple of float, optional
        Tuple containing the x and y limits (x_min, y_min, x_max, y_max).
    **kwargs
        Additional keyword arguments to pass to the system function.

    Returns
    -------
    None
    """
    t_span: Tuple[float, float] = (0.0, t_end)  # Time span [t0, tf]
    t_eval: np.ndarray = np.linspace(*t_span, num_points)  # Time points to evaluate
    y0: List[float] = [v0, w0]  # Initial conditions [v0, w0]

    # Initial simulation
    y = simulate(equations, y0, t_span, t_eval, **kwargs)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_phase_plane(equations, limits=limits, **kwargs)

    # Initialize the line object
    (line,) = ax.plot([], [], lw=2)

    # Create the animation
    ani = animation.FuncAnimation(
        fig, animate, frames=len(t_eval), fargs=(y, line), interval=20, blit=True
    )

    # Connect the click event to the update function
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda event: update_simulation(
            event, equations, t_span, t_eval, line, ani, **kwargs
        ),
    )

    # Show the interactive plot
    plt.show()
