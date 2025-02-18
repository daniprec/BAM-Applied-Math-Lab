import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

"""
Copy or import the following functions from the kuramoto.py file:
- initialize_oscillators
- kuramoto_ode
- kuramoto_order_parameter
"""


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
    # Compute the critical coupling strength kc

    # Given an array of k, compute the theoretical order parameter
    # Remember its value differs for k < kc and k >= kc

    return r


def draw_kuramoto_diagram(
    num_oscillators: int = 1000,
    sigma: float = 1.0,
    dt: float = 1.0,
    t_end: float = 100.0,
    kmin: float = 0.0,
    kmax: float = 5.0,
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
        natural frequencies, default is 1.0.
    dt : float, optional
        Time step for the numerical integration, default is 1.0.
    t_end : float, optional
        End time for the numerical integration, default is 100.0.
    kmin : float, optional
        Minimum coupling strength, default is 0.0.
    kmax : float, optional
        Maximum coupling strength, default is 5.0.
    """
    # Time span and time points relevant for the numerical integration
    t_span = (0, t_end)
    t_eval = np.arange(0, t_end, dt)
    # We will take the last 10% of the time points to compute the order parameter
    idx_end = int(t_end / dt / 10)
    # Initialize the coupling strength and the empirical order parameter lists
    ls_k = np.linspace(kmin, kmax, 100)
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
        # Append the mean order parameter of the last 10% of the time points
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
    draw_kuramoto_diagram()
