import matplotlib.pyplot as plt
import numpy as np
from fitzhugh_nagumo import fitzhugh_nagumo
from scipy.integrate import solve_ivp


def simulate_system(
    i_ext: float, y0: list, t_span: tuple, t_eval: np.ndarray
) -> np.ndarray:
    """
    Simulates the FitzHugh-Nagumo model for a given external current.

    Parameters
    ----------
    i_ext : float
        External stimulus current.
    y0 : list
        Initial conditions.
    t_span : tuple
        Time span for the simulation.
    t_eval : ndarray
        Time points to evaluate the solution.

    Returns
    -------
    y : ndarray
        Solution array.
    """
    sol = solve_ivp(fitzhugh_nagumo, t_span, y0, args=(i_ext,), t_eval=t_eval)
    return sol.y


def plot_bifurcation() -> None:
    """
    Plots the bifurcation diagram by varying the external current.

    Returns
    -------
    None
    """
    i_ext_values = np.linspace(0.0, 1.0, 100)
    max_v = []
    min_v = []

    y0 = [0.0, 0.0]
    t_span = (0.0, 200.0)
    t_eval = np.linspace(*t_span, 2000)

    for i_ext in i_ext_values:
        y = simulate_system(i_ext, y0, t_span, t_eval)
        # Ignore initial transient by taking the last half of the data
        v = y[0][-1000:]
        max_v.append(np.max(v))
        min_v.append(np.min(v))

    plt.figure(figsize=(8, 6))
    plt.plot(i_ext_values, max_v, "r.", label="Max v")
    plt.plot(i_ext_values, min_v, "b.", label="Min v")
    plt.xlabel("External Stimulus Current (i_ext)")
    plt.ylabel("Membrane Potential v")
    plt.title("Bifurcation Diagram of FitzHugh-Nagumo Model")
    plt.legend()
    plt.grid(True)
    plt.show()


def main() -> None:
    """
    Main function to plot the bifurcation diagram.
    """
    plot_bifurcation()


if __name__ == "__main__":
    main()
