from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def sir_model(
    t: float, y: np.ndarray, beta: float = 0.3, gamma: float = 0.1
) -> np.ndarray:
    """Model for the SIR disease dynamics.

    Parameters
    ----------
    t : float
        Time variable. Not used but required by scipy.integrate.solve_ivp().
    y : np.ndarray
        Array containing the susceptible, infected, and recovered populations.
    beta : float, optional
        Transmission rate, by default 0.3.
    gamma : float, optional
        Recovery rate, by default 0.1.

    Returns
    -------
    np.ndarray
        Rate of change of the susceptible, infected, and recovered populations.
    """
    s, i, r = y
    dsdt = -beta * s * i
    didt = beta * s * i - gamma * i
    drdt = gamma * i
    return np.array([dsdt, didt, drdt])


def plot_sir_model(solution: Any) -> None:
    """Plot the solution of the SIR model ODE.

    Parameters
    ----------
    solution : Any
        Solution object returned by scipy.integrate.solve_ivp().
    """
    fig, ax = plt.subplots()
    plt.plot(solution.t, solution.y[0], label="Susceptible")
    plt.plot(solution.t, solution.y[1], label="Infected")
    plt.plot(solution.t, solution.y[2], label="Recovered")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("SIR Model Disease Dynamics")
    plt.legend()
    plt.grid()
    plt.show()
    return fig, ax


def main(beta=0.3, gamma=0.1, s0=0.99, i0=0.01, r0=0.0) -> None:
    """Main function to run the SIR model simulation.

    Parameters
    ----------
    beta : float, optional
        Transmission rate, by default 0.3.
    gamma : float, optional
        Recovery rate, by default 0.1.
    s0 : float, optional
        Initial susceptible population, by default 0.99.
    i0 : float, optional
        Initial infected population, by default 0.01.
    r0 : float, optional
        Initial recovered population, by default 0.0.
    """
    # Time span
    t_span = (0, 160)
    t_eval = np.linspace(0, 160, 1000)

    # Initial conditions
    y0 = [s0, i0, r0]

    # Solve the ODE
    solution = solve_ivp(sir_model, t_span, y0, args=(beta, gamma), t_eval=t_eval)

    # Plot the results
    plot_sir_model(solution)


if __name__ == "__main__":
    main()
