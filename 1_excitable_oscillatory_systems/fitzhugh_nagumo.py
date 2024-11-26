from typing import List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from dynamical_systems import animate, simulate, update_simulation


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


def main() -> None:
    """
    Main function to run the interactive FitzHugh-Nagumo model simulation.
    """
    # Parameters
    i_ext: float = 0.5  # External stimulus
    t_span: Tuple[float, float] = (0.0, 100.0)
    t_eval: np.ndarray = np.linspace(*t_span, 1000)
    y0: List[float] = [0.0, 0.0]  # Initial conditions [v0, w0]

    # Initial simulation
    y = simulate(fitzhugh_nagumo, y0, t_span, t_eval, i_ext)

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
        lambda event: update_simulation(
            event, fitzhugh_nagumo, t_span, t_eval, line, ani, i_ext
        ),
    )

    # Show the interactive plot
    plt.show()


if __name__ == "__main__":
    main()
