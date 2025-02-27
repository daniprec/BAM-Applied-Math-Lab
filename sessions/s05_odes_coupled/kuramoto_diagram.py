import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import cauchy

# I will import several functions from the script "kuramoto.py"
sys.path.append(".")

from sessions.s05_odes_coupled.kuramoto import (
    initialize_oscillators,
    kuramoto_ode_meanfield,
    kuramoto_order_parameter,
)


def kuramoto_critical_coupling(
    k: np.ndarray, scale: float = 1.0, distribution: str = "cauchy"
) -> np.ndarray:
    """
    Compute the theoretical order parameter for the Kuramoto model
    given the coupling strength k.

    Parameters
    ----------
    k : numpy.ndarray
        Coupling strength.
    scale : float, optional
        Standard deviation of the Gaussian distribution of the
        natural frequencies, default is 1.0.
    distribution : str, optional
        Distribution of the natural frequencies, default is "cauchy".

    Returns
    -------
    r : numpy.ndarray
        Order parameter.
    """
    # The probability density function g(omega) is given by the Gaussian
    # distribution, and thus g(0) is
    if distribution == "cauchy":
        g0 = cauchy.pdf(0, loc=0, scale=scale)
    elif distribution == "normal":
        g0 = 1 / (scale * np.sqrt(2 * np.pi))
        g20 = -g0 / scale**2  # Second derivative of the Gaussian distribution
    else:
        raise ValueError(f"Invalid distribution {distribution}")
    # Critical coupling strength
    kc = 2.0 / (g0 * np.pi)
    # Theoretical order parameter
    r = np.zeros_like(k)
    if distribution == "cauchy":
        # This formula is only valid when using Cauchy distribution
        r[k >= kc] = np.sqrt(1 - (kc / k[k > kc]))
    else:
        # Approximation for the Gaussian distribution
        mu = (k[k >= kc] - kc) / kc
        r[k >= kc] = np.sqrt((16 / (np.pi * kc**3)) * (mu / (-g20)))
        # Only works near onset! For large k, r = 1
        r = np.minimum(r, 1)
    return r


def draw_kuramoto_diagram(
    num_oscillators: int = 5000,
    distribution: str = "cauchy",
    scale: float = 1.0,
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
    distribution : str, optional
        Distribution of the natural frequencies, default is "cauchy".
    scale : float, optional
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
    r_theoretical = kuramoto_critical_coupling(
        ls_k, scale=scale, distribution=distribution
    )

    # Initialize the oscillators
    theta, omega = initialize_oscillators(
        num_oscillators, distribution=distribution, scale_omega=scale, seed=seed
    )

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
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw_kuramoto_diagram()
