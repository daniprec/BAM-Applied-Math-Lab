from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp


def michaelis_menten(t: float, s: float, vmax: float = 1.0, km: float = 0.5) -> float:
    """Model for the Michaelis-Menten enzyme kinetics.

    Parameters
    ----------
    t : float
        Time variable. Not used but required by scipy.integrate.solve_ivp().
    s : float
        Substrate concentration at time t.
    vmax : float, optional
        Maximum reaction rate, by default 1.0.
    km : float, optional
        Michaelis constant, by default 0.5.

    Returns
    -------
    float
        Rate of change of the substrate concentration.
    """
    dsdt = -(vmax * s) / (km + s)
    return dsdt


def plot_solution(solution: Any) -> tuple[Figure, Axes]:
    """Plot the solution of the Michaelis-Menten ODE.

    Parameters
    ----------
    solution : Any
        Solution object returned by scipy.integrate.solve_ivp
    """
    fig, ax = plt.subplots()
    plt.plot(solution.t, solution.y[0])
    plt.ylabel("Substrate Concentration")
    plt.title("Michaelis-Menten Kinetics")
    plt.xlabel("Time")
    plt.grid()
    plt.show()
    return fig, ax


def main(vmax=1.0, km=0.5, s0=10) -> None:
    """Main function to run the Michaelis-Menten simulation.

    Parameters
    ----------
    vmax : float, optional
        Maximum reaction rate, by default 1.0.
    km : float, optional
        Michaelis constant, by default 0.5.
    s0 : float, optional
        Initial substrate concentration, by default 10.
    """
    # Time span
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 100)

    # Solve the ODE
    solution = solve_ivp(michaelis_menten, t_span, [s0], args=(vmax, km), t_eval=t_eval)

    # Plot the results
    plot_solution(solution)


if __name__ == "__main__":
    main()
