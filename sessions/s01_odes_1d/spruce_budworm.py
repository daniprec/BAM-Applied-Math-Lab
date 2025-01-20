import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp


def spruce_budworm(t: float, x: float, r: float = 0.5, k: float = 10) -> float:
    """Model for the spruce budworm population dynamics.

    Reference: Chapter 3.7 from Strogatz, S. H. (2018).
    Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering. CRC Press.

    Parameters
    ----------
    t : float
        Time variable. Not used but required by scipy.integrate.solve_ivp().
    x : float
        Budworm population at time t (adimensional).
    r : float, optional
        Intrinsic growth rate (adimensional), by default 0.5.
    k : float, optional
        Carrying capacity of the forest (adimensional), by default 10.

    Returns
    -------
    float
        Rate of change of the budworm population.
    """
    dxdt = r * x * (1 - x / k) - (x**2) / (1 + x**2)
    return dxdt


def plot_spruce_budworm_rate(
    xt: float, r: float = 0.5, k: float = 10
) -> tuple[Figure, Axes]:
    """Plot the rate of change of the spruce budworm ODE.

    Parameters
    ----------
    xt: float
        Budworm population at current t.
    r : float, optional
        Intrinsic growth rate, by default 0.5.
    k : float, optional
        Carrying capacity of the forest, by default 10.
    """
    x = np.linspace(0, k, 1000)
    dxdt = spruce_budworm(0, x, r, k)

    # Find the zeros, and classify them as stable or unstable
    mask_zerocross = np.diff(np.sign(dxdt)).astype(bool)
    mask_stable = mask_zerocross & (dxdt[1:] < 0)
    mask_unstable = mask_zerocross & (dxdt[1:] > 0)
    is_stable = np.where(mask_stable)[0]
    is_unstable = np.where(mask_unstable)[0]

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(x, dxdt)
    # Mark the equilibrium points
    for xc in x[is_unstable]:
        plt.scatter(xc, 0, color="red")
    for xc in x[is_stable]:
        plt.scatter(xc, 0, color="blue")
    # Plot horizontal line at y=0
    plt.axhline(0, color="red", lw=1)
    # Mark the current population
    plt.axvline(xt, color="green", lw=1, linestyle="--")
    plt.xlabel("Budworm Population")
    plt.ylabel("Rate of Change")
    plt.title("Spruce budworm Rate of Change")
    plt.grid()
    return fig, ax


def evolve_spruce_budworm(
    t: np.ndarray, x: np.ndarray, r: float = 0.5, k: float = 10, t_eval: float = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve the spruce budworm ODE and append the new values to the arrays.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    x : np.ndarray
        Budworm population array.
    r : float, optional
        Intrinsic growth rate, by default 0.5.
    k : float, optional
        Carrying capacity of the forest, by default 10.
    t_eval : float, optional
        Time to evaluate the ODE, by default 50.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Time and budworm population arrays.
    """
    t_span = (t[-1], t[-1] + t_eval)
    t_eval = np.linspace(t[-1], t[-1] + t_eval, 1000)

    # Solve the ODE
    solution = solve_ivp(
        spruce_budworm,
        t_span,
        [x[-1]],
        args=(r, k),
        t_eval=t_eval,
        method="RK45",
    )

    t = np.concatenate((t, solution.t))
    x = np.concatenate((x, solution.y[0]))
    x = np.clip(x, a_min=0, a_max=None)
    return t, x


def plot_spruce_budworm(t: np.ndarray, x: np.ndarray) -> tuple[Figure, Axes]:
    """Plot the solution of the spruce budworm ODE.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    x : np.ndarray
        Budworm population array.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(t, x, "green")
    plt.xlabel("Time")
    plt.ylabel("Budworm Population")
    plt.title("Spruce budworm Population Dynamics")
    plt.grid()
    return fig, ax


def main(r: float = 0.5, k: float = 10, t_eval: float = 50):
    """Main function to run the spruce budworm simulation."""
    t = np.array([0])
    x = np.array([k / 10])

    plot_spruce_budworm_rate(x[-1], r=r, k=k)
    plt.show()

    for _ in range(5):
        t, x = evolve_spruce_budworm(t, x, r=r, k=k, t_eval=t_eval)

    plot_spruce_budworm(t, x)
    plt.show()


if __name__ == "__main__":
    main()
