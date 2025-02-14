import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

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
    g0 = 1 / (2 * np.pi * sigma**2) ** (1 / 2)
    # Critical coupling strength
    kc = 2.0 / (g0 * np.pi)
    # Theoretical order parameter
    r = np.zeros_like(k)
    r[k < kc] = 0
    r[k >= kc] = np.sqrt(1 - kc / k[k >= kc])
    return r


def main():
    num_oscillators = 1000
    sigma = 1.0
    dt = 1.0
    t_end = 100.0
    idx_end = int(t_end / dt / 10)
    t_span = (0, t_end)
    t_eval = np.arange(0, t_end, dt)
    ls_k = np.linspace(0, 5, 20)
    ls_r = []

    # Theoretical order parameter
    r_theoretical = kuramoto_critical_coupling(ls_k, sigma=sigma)

    # Empirical order parameter
    for coupling_strength in ls_k:
        theta, omega = initialize_oscillators(num_oscillators, sigma=sigma)
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

        ls_r.append(np.mean(r[-idx_end:]))

    # Plot the order parameter as a function of time
    fig, ax = plt.subplots()
    ax.plot(ls_k, r_theoretical, label="Theoretical")
    ax.plot(ls_k, ls_r, label="Empirical")
    ax.set_xlabel("Coupling strength (K)")
    ax.set_ylabel("Order parameter (r)")
    ax.set_title("Kuramoto model")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
