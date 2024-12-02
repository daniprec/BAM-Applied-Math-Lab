from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import toml
from matplotlib import animation


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
    theta: np.ndarray, omega: np.ndarray, coupling_strength: float, dt: float = 0.01
) -> np.ndarray:
    """
    Computes the time derivative of the phase for each oscillator in the
    Kuramoto model. Solves the ordinary differential equation (ODE) using
    Euler's method (first-order approximation).

    Reference: https://en.wikipedia.org/wiki/Kuramoto_model


    Parameters
    ----------
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
    # Solve the ODE using Euler's method (first-order approximation)
    dtheta = omega + coupling_strength * np.sin(theta - theta[:, None]).mean(axis=1)
    theta = theta + dtheta * dt
    # Keep theta within [0, 2 pi]
    theta = np.mod(theta, 2 * np.pi)
    return theta


def animate_simulation(
    num_oscillators: int,
    coupling_strength: float,
    dt: float,
    distribution: str = "uniform",
):
    """
    Animates the Kuramoto model simulation on the unit circle with the phase centroid.
    The user can interactively control one oscillator using the mouse.
    Left-click adds an oscillator.
    Right-click removes a random oscillator (except the user's controlled oscillator).

    Parameters
    ----------
    num_oscillators : int
        Number of oscillators (N).
    coupling_strength : float
        Coupling strength (K), which determines the strength of synchronization.
    dt : float
        Time step.
    distribution : str, optional
        Distribution of natural frequencies ('uniform' or 'normal').
    """
    # Initialize oscillators (phase and natural frequency)
    theta, omega = initialize_oscillators(num_oscillators, distribution)

    # Initialize order parameter
    order_param = np.mean(np.exp(1j * theta))

    # Animate results
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Kuramoto Model (K = {coupling_strength:.1f})")
    ax.set_xlabel("Cos(theta)")
    ax.set_ylabel("Sin(theta)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw unit circle
    circle = plt.Circle((0, 0), 1, color="lightgray", fill=False)
    ax.add_artist(circle)

    # Initialize scatter plot for oscillators
    scatter = ax.scatter([], [], s=50, color="blue", alpha=0.5)

    # Initialize line for the phase centroid (order parameter)
    (centroid_line,) = ax.plot([], [], color="red", linewidth=2)
    (centroid_point,) = ax.plot([], [], "ro", markersize=8)

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        centroid_line.set_data([], [])
        centroid_point.set_data([], [])
        return scatter, centroid_line, centroid_point

    def update(frame):
        nonlocal theta, order_param, num_oscillators

        # Update phases using Euler's method
        theta = kuramoto_ode(theta, omega, coupling_strength, dt=dt)

        # Update scatter plot
        x = np.cos(theta)
        y = np.sin(theta)
        data = np.vstack((x, y)).T
        scatter.set_offsets(data)

        # Compute order parameter
        order_param = np.mean(np.exp(1j * theta))
        # Update centroid line
        centroid_line.set_data([0, np.real(order_param)], [0, np.imag(order_param)])
        centroid_point.set_data(np.real(order_param), np.imag(order_param))

        return scatter, centroid_line, centroid_point

    def on_click(event):
        nonlocal theta, omega, coupling_strength
        if event.inaxes != ax:
            return
        if event.button == 1:
            # Left click: Increase the coupling strength
            coupling_strength += 0.1

        elif event.button == 3:
            # Right click: Decrease the coupling strength
            coupling_strength -= 0.1
            coupling_strength = max(0.0, coupling_strength)

        # Force the canvas to redraw to update the title
        ax.set_title(f"Kuramoto Model (K = {coupling_strength:.1f})")
        fig.canvas.draw_idle()

    # Connect the mouse events
    fig.canvas.mpl_connect("button_press_event", on_click)

    ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=50)

    plt.tight_layout()
    plt.show()


def main(config: str = "config.toml", key: str = "kuramoto"):
    """
    Main function to run the Kuramoto model simulation.

    Parameters
    ----------
    config : str, optional
        Path to the configuration file.
    key : str, optional
        Key in the configuration file.
    """
    dict_config: Dict[str, Any] = toml.load(config)[key]
    animate_simulation(**dict_config)


if __name__ == "__main__":
    main()
