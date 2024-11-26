from typing import Any, List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def fitzhugh_nagumo(t: float, y: np.ndarray, i_ext: float) -> List[float]:
    """
    Defines the FitzHugh-Nagumo model equations.

    Parameters
    ----------
    t : float
        Time variable.
    y : ndarray
        Array containing the variables [v, w] at time t.
    i_ext : float
        External stimulus current.

    Returns
    -------
    dydt : list of float
        Derivatives [dv/dt, dw/dt] at time t.
    """
    v, w = y
    dvdt = v - (v**3) / 3 - w + i_ext
    dwdt = 0.08 * (v + 0.7 - 0.8 * w)
    return [dvdt, dwdt]


def simulate(
    y0: List[float], t_span: Tuple[float, float], t_eval: np.ndarray, i_ext: float
) -> np.ndarray:
    """
    Performs the numerical integration of the ODEs.

    Parameters
    ----------
    y0 : list of float
        Initial conditions [v0, w0].
    t_span : tuple of float
        Tuple containing the start and end times (t0, tf).
    t_eval : ndarray
        Time points at which to store the computed solutions.
    i_ext : float
        External stimulus current.

    Returns
    -------
    y : ndarray
        Array of solution values at t_eval.
    """
    sol = solve_ivp(fitzhugh_nagumo, t_span, y0, args=(i_ext,), t_eval=t_eval)
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
    t_span: Tuple[float, float],
    t_eval: np.ndarray,
    i_ext: float,
    line: Any,
    ani: Any,
) -> None:
    """
    Updates the simulation with new initial conditions from a mouse click.

    Parameters
    ----------
    event : MouseEvent
        Matplotlib mouse event.
    t_span : tuple of float
        Tuple containing the start and end times (t0, tf).
    t_eval : ndarray
        Time points at which to store the computed solutions.
    i_ext : float
        External stimulus current.
    line : Line2D
        Line object to update.
    ani : FuncAnimation
        Animation object to update.

    Returns
    -------
    None
    """
    v0, w0 = event.xdata, event.ydata
    if v0 is None or w0 is None:
        return
    y0 = [v0, w0]
    y = simulate(y0, t_span, t_eval, i_ext)
    ani.event_source.stop()
    ani.new_frame_seq()
    ani.frame_seq = ani.new_frame_seq()
    ani._args = (y, line)
    ani.event_source.start()


def main() -> None:
    """
    Main function to run the interactive FitzHugh-Nagumo model simulation.

    Returns
    -------
    None
    """
    # Parameters
    i_ext: float = 0.5  # External stimulus
    t_span: Tuple[float, float] = (0.0, 100.0)
    t_eval: np.ndarray = np.linspace(*t_span, 1000)
    y0: List[float] = [0.0, 0.0]  # Initial conditions

    # Initial simulation
    y = simulate(y0, t_span, t_eval, i_ext)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("Membrane Potential (v)")
    ax.set_ylabel("Recovery Variable (w)")
    ax.set_title("Interactive FitzHugh-Nagumo Model")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    # Initialize the line object
    (line,) = ax.plot([], [], lw=2)

    # Create the animation
    ani = animation.FuncAnimation(
        fig, animate, frames=len(t_eval), fargs=(y, line), interval=20, blit=True
    )

    # Connect the click event to the update function
    fig.canvas.mpl_connect(
        "button_press_event",
        lambda event: update_simulation(event, t_span, t_eval, i_ext, line, ani),
    )

    # Show the interactive plot
    plt.show()


if __name__ == "__main__":
    main()
