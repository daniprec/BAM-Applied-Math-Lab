import matplotlib.pyplot as plt
import numpy as np
from dynamical_systems import simulate
from fitzhugh_nagumo import fitzhugh_nagumo


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
        y = simulate(
            fitzhugh_nagumo,
            y0,
            t_span,
            t_eval,
            i_ext=i_ext,
        )
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
