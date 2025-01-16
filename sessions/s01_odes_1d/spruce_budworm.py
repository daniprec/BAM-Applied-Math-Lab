import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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


def plot_spruce_budworm(
    r: float = 0.1, k: int = 100, b: float = 0.1, n0: int = 10, t_show: float = 200
) -> tuple[Figure, Axes]:
    """Plot the solution of the spruce budworm ODE.

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
    t_span = (0, t_show)
    t_eval = np.linspace(0, t_show, 1000)

    # Solve the ODE
    solution = solve_ivp(
        spruce_budworm,
        t_span,
        [n0],
        args=(r, k, b),
        t_eval=t_eval,
        method="RK45",
    )

    fig, ax = plt.subplots()
    plt.plot(solution.t, solution.y[0])
    plt.xlabel("Time")
    plt.ylabel("Budworm Population")
    plt.title("Spruce budworm Population Dynamics")
    plt.grid()
    return fig, ax


def main() -> None:
    """Main function to run the spruce budworm simulation."""
    plot_spruce_budworm()
    plt.show()


if __name__ == "__main__":
    main()
