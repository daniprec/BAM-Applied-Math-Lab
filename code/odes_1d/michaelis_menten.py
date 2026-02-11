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
    dsdt = (vmax * s) / (km + s)
    return dsdt


def plot_michaelis_menten(
    s0: float = 1, vmax: float = 1.0, km: float = 0.5, t_show: int = 2
) -> tuple[Figure, Axes]:
    """Plot the solution of the Michaelis-Menten ODE.

    Parameters
    ----------
    solution : Any
        Solution object returned by scipy.integrate.solve_ivp
    """
    # Time span
    t_span = (0, t_show)
    t_eval = np.linspace(0, t_show, 1000)

    # Solve the ODE
    solution = solve_ivp(
        michaelis_menten,
        t_span,
        [s0],
        args=(vmax, km),
        t_eval=t_eval,
        method="RK45",
    )

    fig, axs = plt.subplots(2, 1)
    # Leave some vertical space (avoids xlabel overlap)
    fig.tight_layout(pad=2.0, h_pad=2.0)

    axs[0].plot(solution.t, solution.y[0])
    axs[0].set_ylabel("Substrate Concentration")
    axs[0].set_xlabel("Time")
    axs[0].grid()

    s = np.linspace(0, max(solution.y[0]), 100)
    v = michaelis_menten(0, s, vmax, km)
    axs[1].plot(s, v)
    axs[1].hlines(vmax / 2, 0, km, linestyles="--", colors="r")
    axs[1].vlines(km, 0, vmax / 2, linestyles="--", colors="r")
    axs[1].set_ylabel("Reaction Rate")
    axs[1].set_xlabel("Substrate Concentration")
    axs[1].grid()

    return fig, axs


def main() -> None:
    """Main function to run the Michaelis-Menten simulation."""
    plot_michaelis_menten()
    plt.show()


if __name__ == "__main__":
    main()
