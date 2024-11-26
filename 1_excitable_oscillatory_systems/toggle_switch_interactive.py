from typing import List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from dynamical_systems import animate, simulate, update_simulation


def toggle_switch(
    t: float, y: np.ndarray, alpha1: float, alpha2: float, beta: float
) -> List[float]:
    """
    Defines the toggle switch model equations.

    Parameters
    ----------
    t : float
        Time variable.
    y : ndarray
        Array containing the variables [u, v] at time t.
    alpha1 : float
        Maximum expression rate of gene 1.
    alpha2 : float
        Maximum expression rate of gene 2.
    beta : float
        Cooperativity (Hill coefficient).

    Returns
    -------
    dydt : list of float
        Derivatives [du/dt, dv/dt] at time t.
    """
    u, v = y
    du_dt = alpha1 / (1 + v**beta) - u
    dv_dt = alpha2 / (1 + u**beta) - v
    return [du_dt, dv_dt]


def main() -> None:
    """
    Main function to run the interactive toggle switch simulation.
    """
    # Parameters
    alpha1: float = 5.0
    alpha2: float = 5.0
    beta: float = 2.0
    t_span: Tuple[float, float] = (0.0, 50.0)
    t_eval: np.ndarray = np.linspace(*t_span, 1000)
    y0: List[float] = [0.1, 0.1]  # Initial conditions [u0, v0]

    # Initial simulation
    y = simulate(toggle_switch, y0, t_span, t_eval, alpha1, alpha2, beta)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("Protein Concentration u")
    ax.set_ylabel("Protein Concentration v")
    ax.set_title("Interactive Toggle Switch Model")
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)

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
            event, toggle_switch, t_span, t_eval, line, ani, alpha1, alpha2, beta
        ),
    )

    # Show the interactive plot
    plt.show()


if __name__ == "__main__":
    main()
