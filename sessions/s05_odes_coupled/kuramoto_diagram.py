import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# I will import several functions from the script "kuramoto.py"
sys.path.append(".")

from sessions.s05_odes_coupled.kuramoto import (
    initialize_oscillators,
    kuramoto_ode_meanfield,
    kuramoto_order_parameter,
)


def kuramoto_critical_coupling(k: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Compute the theoretical order parameter for the Kuramoto model
    given the coupling strength k.

    Parameters
    ----------
    k : numpy.ndarray
        Coupling strength.
    sigma : float, optional
        Standard deviation of the Gaussian distribution of the
        natural frequencies, default is 1.0.

    Returns
    -------
    r : numpy.ndarray
        Order parameter.
    """
    # The probability density function g(omega) is given by the Gaussian
    # distribution, and thus g(0) is
    g0 = 1 / ((2 * np.pi * (sigma**2)) ** (1 / 2))
    # Critical coupling strength
    kc = 2.0 / (g0 * np.pi)
    # Theoretical order parameter
    r = np.zeros_like(k)
    r[k >= kc] = np.sqrt(1 - (kc / k[k > kc]))
    return r


def draw_kuramoto_diagram(
    num_oscillators: int = 1000,
    sigma: float = 1.0,
    dt: float = 0.01,
    t_end: float = 100.0,
    kmin: float = 0.0,
    kmax: float = 5.0,
    knum: int = 50,
    seed: int = 1,
):
    """
    Draw the Kuramoto diagram, showing the order parameter as a function
    of the coupling strength. Theoretical and empirical order parameters
    are plotted.

    Parameters
    ----------
    num_oscillators : int, optional
        Number of oscillators, default is 1000.
    sigma : float, optional
        Standard deviation of the Gaussian distribution of the
        natural frequencies, default is 0.01.
    dt : float, optional
        Time step for the numerical integration, default is 0.01.
    t_end : float, optional
        End time for the numerical integration, default is 100.0.
    kmin : float, optional
        Minimum coupling strength, default is 0.0.
    kmax : float, optional
        Maximum coupling strength, default is 5.0.
    knum : int, optional
        Number of coupling strengths, default is 50.
    seed : int, optional
        Seed for the random number generator, default is 1.
    """
    # Time span and time points relevant for the numerical integration
    t_span = (0, t_end)
    t_eval = np.arange(0, t_end, dt)
    # We will take the last X% of the time points to compute the order parameter
    idx_end = int(len(t_eval) * 0.25)
    t_eval = t_eval[-idx_end:]
    # Initialize the coupling strength and the empirical order parameter lists
    ls_k = np.linspace(kmin, kmax, knum)
    ls_r_q10 = np.zeros_like(ls_k)
    ls_r_q50 = np.zeros_like(ls_k)
    ls_r_q90 = np.zeros_like(ls_k)

    # Theoretical order parameter
    r_theoretical = kuramoto_critical_coupling(ls_k, sigma=sigma)

    # Initialize the oscillators
    theta, omega = initialize_oscillators(num_oscillators, sigma=sigma, seed=seed)

    # Empirical order parameter
    for idx, coupling_strength in enumerate(ls_k):
        sol = solve_ivp(
            kuramoto_ode_meanfield,
            t_span,
            theta,
            t_eval=t_eval,
            args=(omega, coupling_strength),
        )
        theta = sol.y
        # Keep theta within [0, 2 * pi]
        theta = np.mod(theta, 2 * np.pi)

        # Compute the order parameter
        r, phi, rcosphi, rsinphi = kuramoto_order_parameter(theta)

        # Append the mean order parameter of the last X% of the time points
        ls_r_q10[idx] = np.percentile(r, 10)
        ls_r_q50[idx] = np.percentile(r, 50)
        ls_r_q90[idx] = np.percentile(r, 90)

        print(
            f"K = {coupling_strength:.2f}, r (theory) = {r_theoretical[idx]:.2f}"
            f", r (empirical) = {ls_r_q50[idx]:.2f}"
        )

        # Take the last state as the initial condition for the next iteration
        theta = theta[:, -1]

    # Plot the order parameter as a function of time
    fig, ax = plt.subplots()
    ax.plot(ls_k, r_theoretical, label="Theoretical", color="blue")
    # Plot the empirical order parameter as points with error bars
    ax.errorbar(
        ls_k,
        ls_r_q50,
        yerr=[ls_r_q50 - ls_r_q10, ls_r_q90 - ls_r_q50],
        fmt="o",
        label="Empirical",
        color="red",
    )
    ax.set_xlabel("Coupling strength (K)")
    ax.set_ylabel("Order parameter (r)")
    ax.set_title("Kuramoto model")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    draw_kuramoto_diagram()
