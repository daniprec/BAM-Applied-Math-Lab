from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from scipy.integrate import solve_ivp


def initialize_oscillators(
    num_oscillators: int, distribution: str = "uniform"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initializes the phases and natural frequencies of the oscillators.

    Parameters
    ----------
    num_oscillators : int
        Number of oscillators.
    distribution : str, optional
        Distribution of natural frequencies ('uniform' or 'normal').

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
        omega = np.random.normal(0.5, 0.5, num_oscillators)
    else:
        raise ValueError("Distribution must be 'uniform' or 'normal'.")

    return theta, omega


def kuramoto_ode(
    t: float, theta: np.ndarray, omega: np.ndarray = 1, coupling_strength: float = 1.0
) -> np.ndarray:
    """
    Computes the time derivative of the phase for each oscillator in the
    Kuramoto model.

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


def animate_simulation(distribution: str = "uniform", t_show: float = 1.0):
    """
    Animates the Kuramoto model simulation on the unit circle with the phase centroid.
    The user can interactively control one oscillator using the mouse.
    Left-click adds an oscillator.
    Right-click removes a random oscillator (except the user's controlled oscillator).

    Parameters
    ----------
    distribution : str, optional
        Distribution of natural frequencies ('uniform' or 'normal').
    """
    # ------------------------------------------------------------------------#
    # PARAMETERS
    # ------------------------------------------------------------------------#
    coupling_strength = 1.0
    num_oscillators = 100
    max_oscillators = 500

    # Initialize oscillators (phase and natural frequency)
    theta, omega = initialize_oscillators(max_oscillators, distribution)
    t_span = (0, t_show)

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

    # ------------------------------------------------------------------------#
    # ANIMATION
    # ------------------------------------------------------------------------#

    def update(frame):
        # Acces the variables from the outer scope to update them
        nonlocal theta, coupling_strength, num_oscillators

        # Solve the ODE system
        sol = solve_ivp(
            kuramoto_ode,
            t_span,
            theta,
            args=(omega, coupling_strength),
        )
        theta = sol.y[..., -1]
        # Keep theta within [0, 2 * pi]
        theta = np.mod(theta, 2 * np.pi)

        # Update scatter plot
        x = np.cos(theta)
        y = np.sin(theta)
        data = np.vstack((x, y)).T
        # Take as many oscillators as chosen by the user
        scatter.set_offsets(data[:num_oscillators])

        # Compute order parameter - taking only as many as user decides
        order_param = np.mean(np.exp(1j * theta[:num_oscillators]))
        # Update centroid line
        centroid_line.set_data([0, np.real(order_param)], [0, np.imag(order_param)])
        centroid_point.set_data([np.real(order_param)], [np.imag(order_param)])

        return scatter, centroid_line, centroid_point

    ani = animation.FuncAnimation(fig, update, blit=True, interval=1)

    # ------------------------------------------------------------------------#
    # SLIDERS
    # ------------------------------------------------------------------------#

    # Create the sliders axes
    ax_sliders: Axes = axs[1, 0]
    ax_sliders.axis("off")  # Turn off the axis (the grid and numbers)

    # Coupling strength slider
    ax_coupling = ax_sliders.inset_axes([0.0, 0.8, 0.8, 0.1])
    slider_coupling = plt.Slider(
        ax_coupling,
        "Coupling Strength (K)",
        valmin=0.0,
        valmax=10.0,
        valinit=coupling_strength,
        valstep=0.1,
    )

    # Number of oscillators slider
    ax_num_oscillators = ax_sliders.inset_axes([0.0, 0.6, 0.8, 0.1])
    slider_num_oscillators = plt.Slider(
        ax_num_oscillators,
        "Number of Oscillators",
        valmin=1,
        valmax=500,
        valinit=num_oscillators,
        valstep=1,
    )

    def update_sliders(_):
        # Acces the variables from the outer scope to update them
        nonlocal coupling_strength, num_oscillators
        # Update the parameters according to the sliders values
        coupling_strength = slider_coupling.val
        num_oscillators = int(slider_num_oscillators.val)

    # Attach the update function to the sliders
    slider_coupling.on_changed(update_sliders)
    slider_num_oscillators.on_changed(update_sliders)

    # ------------------------------------------------------------------------#
    #  DISPLAY
    # ------------------------------------------------------------------------#

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    animate_simulation()
