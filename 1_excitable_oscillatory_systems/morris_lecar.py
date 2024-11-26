from typing import List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from dynamical_systems import animate, simulate, update_simulation


def morris_lecar(t: float, y: np.ndarray, i_ext: float) -> List[float]:
    """
    Defines the Morris-Lecar model equations.

    Parameters
    ----------
    t : float
        Time variable.
    y : ndarray
        Array containing the variables [V, w] at time t.
    i_ext : float
        External current.

    Returns
    -------
    dydt : list of float
        Derivatives [dV/dt, dw/dt] at time t.
    """
    V, w = y
    # Parameters for the Morris-Lecar model
    C = 20.0  # Membrane capacitance (μF/cm²)
    g_Ca = 4.0  # Maximal Ca²⁺ conductance (mS/cm²)
    g_K = 8.0  # Maximal K⁺ conductance (mS/cm²)
    g_L = 2.0  # Leak conductance (mS/cm²)
    V_Ca = 120.0  # Ca²⁺ reversal potential (mV)
    V_K = -84.0  # K⁺ reversal potential (mV)
    V_L = -60.0  # Leak reversal potential (mV)
    V1 = -1.2  # Parameters for steady-state functions
    V2 = 18.0
    V3 = 2.0
    V4 = 30.0
    phi = 0.04  # Temperature-like parameter

    # Steady-state functions
    m_inf = 0.5 * (1 + np.tanh((V - V1) / V2))
    w_inf = 0.5 * (1 + np.tanh((V - V3) / V4))
    tau_w = 1 / (phi * np.cosh((V - V3) / (2 * V4)))

    # Differential equations
    dVdt = (
        i_ext - g_L * (V - V_L) - g_Ca * m_inf * (V - V_Ca) - g_K * w * (V - V_K)
    ) / C
    dwdt = (w_inf - w) / tau_w

    return [dVdt, dwdt]


def main() -> None:
    """
    Main function to run the interactive Morris-Lecar model simulation.

    Returns
    -------
    None
    """
    # Parameters
    i_ext: float = 40.0  # External current (μA/cm²)
    t_span: Tuple[float, float] = (0.0, 200.0)
    t_eval: np.ndarray = np.linspace(*t_span, 2000)
    y0: List[float] = [-60.0, 0.0]  # Initial conditions [V0, w0]

    # Initial simulation
    y = simulate(morris_lecar, y0, t_span, t_eval, i_ext)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("Membrane Potential (V)")
    ax.set_ylabel("Recovery Variable (w)")
    ax.set_title("Interactive Morris-Lecar Model")
    ax.set_xlim(-80, 60)
    ax.set_ylim(0, 1)

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
            event, morris_lecar, t_span, t_eval, line, ani, i_ext
        ),
    )

    # Show the interactive plot
    plt.show()


if __name__ == "__main__":
    main()
