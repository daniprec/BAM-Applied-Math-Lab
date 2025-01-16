from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def spruce_budworm(
    t: float, n: float, r: float = 0.1, k: float = 100, b: float = 0.1
) -> float:
    """Model for the spruce budworm population dynamics.

    Parameters
    ----------
    t : float
        Time variable. Not used but required by scipy.integrate.solve_ivp().
    n : float
        Budworm population at time t.
    r : float, optional
        Intrinsic growth rate, by default 0.1.
    k : float, optional
        Carrying capacity of the forest, by default 100.
    b : float, optional
        Predation rate, by default 0.1.

    Returns
    -------
    float
        Rate of change of the budworm population.
    """
    dndt = r * n * (1 - n / k) - (b * n**2) / (1 + n**2)
    return dndt


def plot_spruce_budworm(solution: Any) -> None:
    """Plot the solution of the spruce budworm ODE.

    Parameters
    ----------
    solution : Any
        Solution object returned by scipy.integrate.solve_ivp().
    """
    fig, ax = plt.subplots()
    plt.plot(solution.t, solution.y[0])
    plt.xlabel("Time")
    plt.ylabel("Budworm Population")
    plt.title("Spruce budworm Population Dynamics")
    plt.grid()
    plt.show()
    return fig, ax


def main(r=0.1, k=100, b=0.1, n0=10) -> None:
    """Main function to run the spruce budworm simulation.

    Parameters
    ----------
    r : float, optional
        Intrinsic growth rate, by default 0.1.
    k : float, optional
        Carrying capacity of the forest, by default 100.
    b : float, optional
        Predation rate, by default 0.1.
    n0 : int, optional
        Initial budworm population, by default 10.
    """
    # Time span
    t_span = (0, 200)
    t_eval = np.linspace(0, 200, 1000)

    # Solve the ODE
    solution = solve_ivp(
        spruce_budworm,
        t_span,
        [n0],
        args=(r, k, b),
        t_eval=t_eval,
        method="RK45",
    )

    # Plot the results
    plot_spruce_budworm(solution)


if __name__ == "__main__":
    main()
