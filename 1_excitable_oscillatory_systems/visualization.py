from typing import Any, List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from dynamical_systems import compute_fixed_points, compute_nullclines, simulate


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
    *args,
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


def plot_phase_plane(
    equations: callable,
    i_ext: float = 0.5,
    limits: tuple[float] = (-3, -3, 3, 3),
    kwargs: dict = None,
) -> None:
    """
    Plots the phase plane of any excitable-oscillatory model.

    Parameters
    ----------
    equations : callable
        Function that defines the model equations.
    i_ext : float
        External stimulus current.
    """
    kwargs = kwargs if isinstance(kwargs, dict) else {}

    # Create a grid of points
    v = np.linspace(limits[0], limits[2], 20)
    w = np.linspace(limits[1], limits[3], 20)
    V, W = np.meshgrid(v, w)

    # Compute derivatives
    dv, dw = equations(0, [V, W], i_ext)

    # Plot vector field
    plt.quiver(V, W, dv, dw, color="gray", alpha=0.5)

    # Compute nullclines
    v_nullcline, w_nullcline = compute_nullclines(
        equations, t=0, i_ext=i_ext, limits=limits, kwargs=kwargs
    )

    # Plot nullclines
    plt.scatter(v_nullcline[0], v_nullcline[1], c="b", s=1, label="v nullcline")
    plt.scatter(w_nullcline[0], w_nullcline[1], c="r", s=1, label="w nullcline")

    # This is a nonlinear equation; we'll use numerical methods to find fixed points.
    # Compute and plot fixed points
    fixed_points = compute_fixed_points(equations, t=0, i_ext=i_ext, kwargs=kwargs)
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
    equations: callable,
    i_ext: float = 0.5,
    t_span: float = 100,
    t_eval: float = 1000,
    v0: float = 0.0,
    w0: float = 0.0,
    limits: tuple[float] = (-3, -3, 3, 3),
) -> None:
    """
    Main function to run the interactive model simulation.
    """
    t_span: Tuple[float, float] = (0.0, t_span)
    t_eval: np.ndarray = np.linspace(*t_span, t_eval)
    y0: List[float] = [v0, w0]  # Initial conditions [v0, w0]

    # Initial simulation
    y = simulate(equations, y0, t_span, t_eval, i_ext)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_phase_plane(equations, i_ext=i_ext, limits=limits)

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
            event, equations, t_span, t_eval, line, ani, i_ext
        ),
    )

    # Show the interactive plot
    plt.show()
