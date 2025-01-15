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
        Time variable.
    n : float
        Budworm population at time t.
    r : float, optional
        Intrinsic growth rate, by default 0.1.
    k : float, optional
        Carrying capacity, by default 100.
    b : float, optional
        Predation rate, by default 0.1.

    Returns
    -------
    float
        Rate of change of the budworm population.
    """
    dndt = r * n * (1 - n / k) - (b * n**2) / (1 + n**2)
    return dndt


def plot_solution(solution: Any) -> None:
    """Plot the solution of the spruce budworm ODE.

    Parameters
    ----------
    solution : Any
        Solution object returned by solve_ivp.
    """
    plt.plot(solution.t, solution.y[0])
    plt.xlabel("Time")
    plt.ylabel("Budworm Population")
    plt.title("Spruce Budworm Population Dynamics")
    plt.grid()
    plt.show()


def main() -> None:
    """Main function to run the spruce budworm simulation."""
    # Parameters
    r = 0.1  # intrinsic growth rate
    k = 100  # carrying capacity
    b = 0.1  # predation rate

    # Initial condition
    n0 = 10

    # Time span
    t_span = (0, 200)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Solve the ODE
    solution = solve_ivp(spruce_budworm, t_span, [n0], args=(r, k, b), t_eval=t_eval)

    # Plot the results
    plot_solution(solution)


if __name__ == "__main__":
    main()
