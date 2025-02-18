from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from scipy.integrate import solve_ivp


def initialize_oscillators(
    num_oscillators: int, distribution: str = "normal", sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initializes the phases and natural frequencies of the oscillators.

    Parameters
    ----------
    num_oscillators : int
        Number of oscillators.
    distribution : str, optional
        Distribution of natural frequencies ('uniform' or 'normal').
        Kuramoto uses unimodal distributions, such as the normal distribution.
    sigma : float, optional
        Standard deviation of the normal distribution, by default 1.0.

    Returns
    -------
    theta : ndarray
        Initial phases of the oscillators.
    omega : ndarray
        Natural frequencies of the oscillators.
    """
    # Assign a random initial phase to each oscillator
    # (position in the unit circle)
    theta = np.random.uniform(0, 2 * np.pi, num_oscillators)

    # Assign a random natural frequency to each oscillator (angular velocity)
    if distribution == "uniform":
        omega = np.random.uniform(-1.0, 1.0, num_oscillators)
    elif distribution == "normal":
        omega = np.random.normal(0, sigma, num_oscillators)
    else:
        raise ValueError("Distribution must be 'uniform' or 'normal'.")

    return theta, omega


def kuramoto_ode_pairwise(
    t: float, theta: np.ndarray, omega: np.ndarray = 1, coupling_strength: float = 1.0
) -> np.ndarray:
    """
    Computes the time derivative of the phase for each oscillator in the
    Kuramoto model. Uses the pairwise interactions: the coupling term is
    the average of the sine of the pairwise differences between phases.

    Reference: https://en.wikipedia.org/wiki/Kuramoto_model


    Parameters
    ----------
    t : float
        Time (not used in the Kuramoto model).
    theta : np.ndarray
        Phases of the oscillators.
    omega : np.ndarray
        Natural frequencies of the oscillators.
    coupling_strength : float
        Coupling strength (K), which determines the strength of synchronization.

    Returns
    -------
    np.ndarray
        Time derivative of the phase for each oscillator.
    """
    # Keep theta within [0, 2 * pi]
    theta = np.mod(theta, 2 * np.pi)
    # Compute the pairwise differences between phases
    theta_diff = theta[:, None] - theta
    # Average over all oscillators
    coupling_term = coupling_strength * np.mean(np.sin(theta_diff), axis=0)
    # Compute the time derivative
    dtheta_dt = omega + coupling_term
    return dtheta_dt


def kuramoto_order_parameter(theta: np.ndarray) -> tuple:
    """
    Computes the order parameter of the Kuramoto model.

    Parameters
    ----------
    theta : np.ndarray
        Phases of the oscillators, in radians. Shape is (N, T).

    Returns
    -------
    r : float
        Order parameter (synchronization index).
    phi : float
        Phase of the order parameter.
    rcosphi : float
        Real part of the order parameter, r * cos(phi).
    rsinphi : float
        Imaginary part of the order parameter, r * sin(phi).
    """
    # Compute the order parameter as r * exp(i * phi)
    order_param = np.mean(np.exp(1j * theta), axis=0)
    # The absolute value of the order parameter is the synchronization index
    r = np.abs(order_param)
    # The angle of the order parameter is the phase of the synchronization
    phi = np.angle(order_param)
    # The real part of the order parameter is r * cos(phi)
    rcosphi = np.real(order_param)
    # The imaginary part of the order parameter is r * sin(phi)
    rsinphi = np.imag(order_param)
    return r, phi, rcosphi, rsinphi


def kuramoto_ode_meanfield(
    t: float,
    theta: np.ndarray,
    omega: np.ndarray = None,
    coupling_strength: float = 1.0,
) -> np.ndarray:
    """
    Computes the time derivative of the phase for each oscillator in the
    Kuramoto model. Uses the mean-field approximation: the coupling term is
    the sine of the difference between the phase centroid and
    the phase of each oscillator.

    Reference: https://en.wikipedia.org/wiki/Kuramoto_model


    Parameters
    ----------
    t : float
        Time (not used in the Kuramoto model).
    theta : np.ndarray
        Phases of the oscillators, in radians.
    omega : np.ndarray
        Natural frequencies of the oscillators.
    coupling_strength : float
        Coupling strength (K), which determines the strength of synchronization.

    Returns
    -------
    np.ndarray
        Time derivative of the phase for each oscillator.
    """
    # Ensure omega is an array and matches the shape of theta
    if omega is None:
        omega = np.ones_like(theta)
    # Keep theta within [0, 2 * pi]
    theta = np.mod(theta, 2 * np.pi)
    # Compute the order parameter
    r, phi, _, _ = kuramoto_order_parameter(theta)
    # Compute the coupling term
    coupling_term = coupling_strength * r * np.sin(phi - theta)
    # Compute the time derivative
    dtheta_dt = omega + coupling_term
    return dtheta_dt


def run_simulation(dt: float = 0.01, interval: int = 1):
    """
    Animates the Kuramoto model simulation on the unit circle with the phase centroid.

    Parameters
    ----------
    dt : float, optional
        Time step for the integration time, by default 0.01.
    interval : int, optional
        Interval between frames in milliseconds, by default 10.
    """
    # ------------------------------------------------------------------------#
    # PARAMETERS
    # ------------------------------------------------------------------------#
    coupling_strength = 1.0  # K
    max_k = 5.0  # Maximum K allowed in the slider
    num_oscillators = 100  # Number of oscillators
    max_oscillators = 500  # Maximum number of oscillators allowed in the slider
    sigma = 1.0  # Standard deviation of the natural frequencies

    # Initialize oscillators (phase and natural frequency)
    theta, omega = initialize_oscillators(num_oscillators, sigma=sigma)
    t_span = (0, dt)

    # Order parameter (phase centroid)
    ls_order_param = [0] * 500
    ls_t = np.arange(0, 500) * dt
    dict_kr = {0: [0]}

    # ------------------------------------------------------------------------#
    # INITIALIZE THE PLOT
    # ------------------------------------------------------------------------#

    # We will do a grid of three plots: one square on the right,
    # and two rectangles on the left
    fig, axs = plt.subplots(2, 2, figsize=(12, 6), height_ratios=[1, 0.2])

    # Left plot: Phase space
    ax_phase: Axes = axs[0, 0]
    ax_phase.set_title("Kuramoto Model")
    ax_phase.set_xlabel("Cos(theta)")
    ax_phase.set_ylabel("Sin(theta)")
    ax_phase.set_xlim(-1.1, 1.1)
    ax_phase.set_ylim(-1.1, 1.1)
    ax_phase.set_aspect("equal")
    ax_phase.grid(True)

    # Draw unit circle
    circle = plt.Circle((0, 0), 1, color="lightgray", fill=False)
    ax_phase.add_artist(circle)

    # Initialize scatter plot for oscillators
    scatter = ax_phase.scatter([], [], s=50, color="blue", alpha=0.5)

    # Initialize line for the phase centroid (order parameter)
    (centroid_line,) = ax_phase.plot([], [], color="red", linewidth=2)
    (centroid_point,) = ax_phase.plot([], [], "ro", markersize=8)

    # Right plot: r vs t
    ax_order_param: Axes = axs[1, 1]
    ax_order_param.set_title("Order Parameter (r)")
    ax_order_param.set_xlabel("Time")
    ax_order_param.set_ylabel("r")
    ax_order_param.set_ylim(0, 1)
    ax_order_param.grid(True)
    (line_order_param,) = ax_order_param.plot(ls_t, ls_order_param, color="red")

    # Right plot: r vs K
    ax_kr: Axes = axs[0, 1]
    ax_kr.set_title("Order Parameter (r) vs Coupling Strength (K)")
    ax_kr.set_xlabel("Coupling Strength (K)")
    ax_kr.set_ylabel("r")
    ax_kr.set_ylim(0, 1)
    ax_kr.set_xlim(0, max_k)
    ax_kr.grid(True)
    (line_kr,) = ax_kr.plot([], [], color="red", marker="o", linestyle="--")

    # ------------------------------------------------------------------------#
    # ANIMATION
    # ------------------------------------------------------------------------#

    def update(frame: int):
        # Acces the variables from the outer scope to update them
        nonlocal theta, coupling_strength, dict_kr, ls_order_param

        # Solve the ODE system
        sol = solve_ivp(
            kuramoto_ode_pairwise,
            t_span,
            theta,
            args=(omega, coupling_strength),
        )
        theta = sol.y[..., -1]
        # Keep theta within [0, 2 * pi]
        theta = np.mod(theta, 2 * np.pi)

        # Update scatter plot on the unit circle (left plot)
        x = np.cos(theta)
        y = np.sin(theta)
        data = np.vstack((x, y)).T
        scatter.set_offsets(data)

        # Compute the order parameter
        r, phi, rcosphi, rsinphi = kuramoto_order_parameter(theta)

        # Update centroid line
        centroid_line.set_data([0, rcosphi], [0, rsinphi])
        centroid_point.set_data([rcosphi], [rsinphi])

        # Update order parameter list - It will always contain the same amount of values
        ls_order_param.append(r)
        ls_order_param.pop(0)
        # Update order parameter plot
        line_order_param.set_data(ls_t, ls_order_param)

        # Update dictonary of K vs r
        if coupling_strength not in dict_kr.keys():
            dict_kr[coupling_strength] = [r]
            # Sort the dictionary by the coupling strength
            dict_kr = dict(sorted(dict_kr.items()))
        else:
            dict_kr[coupling_strength].append(r)
            # Store no more than 200 values per K to avoid memory issues
            dict_kr[coupling_strength] = dict_kr[coupling_strength][-200:]

        # Update K vs r plot by computing the mean of the r values for each K
        ls_means = [np.mean(dict_kr[k]) for k in dict_kr.keys()]
        line_kr.set_data(list(dict_kr.keys()), ls_means)

        return scatter, centroid_line, centroid_point, line_order_param, line_kr

    ani = animation.FuncAnimation(fig, update, blit=True, interval=interval)

    # ------------------------------------------------------------------------#
    # SLIDERS
    # ------------------------------------------------------------------------#

    # Create the sliders axes
    ax_sliders: Axes = axs[1, 0]
    ax_sliders.axis("off")  # Turn off the axis (the grid and numbers)

    # Coupling strength slider
    ax_coupling = ax_sliders.inset_axes([0.0, 0.4, 0.8, 0.1])
    slider_coupling = plt.Slider(
        ax_coupling,
        "Coupling Strength (K)",
        valmin=0.0,
        valmax=max_k,
        valinit=coupling_strength,
        valstep=0.05,
    )

    # Number of oscillators slider
    ax_num_oscillators = ax_sliders.inset_axes([0.0, 0.6, 0.8, 0.1])
    slider_num_oscillators = plt.Slider(
        ax_num_oscillators,
        "Number of Oscillators",
        valmin=1,
        valmax=max_oscillators,
        valinit=num_oscillators,
        valstep=1,
    )

    # Sigmas slider
    ax_sigma = ax_sliders.inset_axes([0.0, 0.8, 0.8, 0.1])
    slider_sigma = plt.Slider(
        ax_sigma,
        "Sigma",
        valmin=0.1,
        valmax=2.0,
        valinit=sigma,
        valstep=0.1,
    )

    def update_sliders(_):
        # Acces the variables from the outer scope to update them
        nonlocal coupling_strength, num_oscillators, sigma, theta, omega
        # Update the parameters according to the sliders values
        coupling_strength = slider_coupling.val
        num_oscillators = int(slider_num_oscillators.val)
        sigma = slider_sigma.val
        # Reinitalize the oscillators
        theta, omega = initialize_oscillators(num_oscillators, sigma=sigma)

    # Attach the update function to the sliders
    slider_coupling.on_changed(update_sliders)
    slider_num_oscillators.on_changed(update_sliders)
    slider_sigma.on_changed(update_sliders)

    # ------------------------------------------------------------------------#
    #  DISPLAY
    # ------------------------------------------------------------------------#

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation()
