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
    theta = np.random.uniform(0, 2 * np.pi, num_oscillators)

    if distribution == "uniform":
        omega = np.random.uniform(-1.0, 1.0, num_oscillators)
    elif distribution == "normal":
        omega = np.random.normal(0.0, 0.5, num_oscillators)
    else:
        raise ValueError("Distribution must be 'uniform' or 'normal'.")

    return theta, omega


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
        Coupling strength (K).
    dt : float
        Time step.
    distribution : str, optional
        Distribution of natural frequencies ('uniform' or 'normal').

    Returns
    -------
    None
    """
    # Initialize oscillators
    theta, omega = initialize_oscillators(num_oscillators, distribution)

    # Designate the controlled oscillator (index 0)
    controlled_idx = 0

    # Initialize order parameter
    order_param = np.mean(np.exp(1j * theta))

    # Animate results
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Kuramoto Model Synchronization with Interactive Control")
    ax.set_xlabel("Cos(θ)")
    ax.set_ylabel("Sin(θ)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw unit circle
    circle = plt.Circle((0, 0), 1, color="lightgray", fill=False)
    ax.add_artist(circle)

    # Initialize scatter plot for oscillators
    scatter = ax.scatter([], [], s=50, color="blue")

    # Highlight the controlled oscillator
    (controlled_point,) = ax.plot([], [], "ro", markersize=8)

    # Initialize line for the phase centroid (order parameter)
    (centroid_line,) = ax.plot([], [], color="red", linewidth=2)

    # Time variable
    t = 0.0

    # Mouse position variable
    mouse_theta = None  # Phase corresponding to mouse position

    # Flag to check if mouse is within axes
    mouse_in_axes = False

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        controlled_point.set_data([], [])
        centroid_line.set_data([], [])
        return scatter, controlled_point, centroid_line

    def update(frame):
        nonlocal theta, t, order_param, mouse_theta, mouse_in_axes, num_oscillators

        # Update phases using Euler's method
        # Exclude the controlled oscillator if mouse is in axes
        for i in range(num_oscillators):
            if i == controlled_idx and mouse_in_axes:
                continue  # Skip updating controlled oscillator
            coupling = (coupling_strength / num_oscillators) * np.sum(
                np.sin(theta[i] - theta)
            )
            theta[i] += (omega[i] + coupling) * dt

        # Keep theta within [0, 2π]
        theta = np.mod(theta, 2 * np.pi)

        # Update controlled oscillator phase if mouse is in axes
        if mouse_in_axes and mouse_theta is not None:
            theta[controlled_idx] = mouse_theta
        else:
            # Update controlled oscillator normally
            coupling = (coupling_strength / num_oscillators) * np.sum(
                np.sin(theta[controlled_idx] - theta)
            )
            theta[controlled_idx] += (omega[controlled_idx] + coupling) * dt
            theta[controlled_idx] = np.mod(theta[controlled_idx], 2 * np.pi)

        # Compute order parameter
        order_param = np.mean(np.exp(1j * theta))

        # Update scatter plot
        x = np.cos(theta)
        y = np.sin(theta)
        data = np.vstack((x, y)).T
        scatter.set_offsets(data)

        # Update controlled oscillator point
        controlled_x = np.cos(theta[controlled_idx])
        controlled_y = np.sin(theta[controlled_idx])
        controlled_point.set_data(controlled_x, controlled_y)

        # Update centroid line
        centroid_line.set_data([0, np.real(order_param)], [0, np.imag(order_param)])

        # Update time
        t += dt

        return scatter, controlled_point, centroid_line

    def on_mouse_move(event):
        nonlocal mouse_theta, mouse_in_axes
        if event.inaxes == ax:
            mouse_in_axes = True
            # Calculate angle based on mouse position
            dx = event.xdata
            dy = event.ydata
            angle = np.arctan2(dy, dx)
            mouse_theta = np.mod(angle, 2 * np.pi)
        else:
            mouse_in_axes = False
            mouse_theta = None

    def on_click(event):
        nonlocal theta, omega, num_oscillators
        if event.inaxes != ax:
            return
        if event.button == 1:
            # Left click: Add an oscillator
            new_theta = np.random.uniform(0, 2 * np.pi)
            if distribution == "uniform":
                new_omega = np.random.uniform(-1.0, 1.0)
            elif distribution == "normal":
                new_omega = np.random.normal(0.0, 0.5)
            else:
                raise ValueError("Distribution must be 'uniform' or 'normal'.")
            # Append to theta and omega
            theta = np.append(theta, new_theta)
            omega = np.append(omega, new_omega)
            num_oscillators += 1
        elif event.button == 3:
            # Right click: Remove a random oscillator (excluding controlled oscillator)
            if num_oscillators <= 1:
                return  # Can't remove any more oscillators
            # Get indices excluding the controlled oscillator
            indices = np.arange(num_oscillators)
            indices = indices[indices != controlled_idx]
            # Choose a random oscillator to remove
            remove_idx = np.random.choice(indices)
            # Remove from theta and omega
            theta = np.delete(theta, remove_idx)
            omega = np.delete(omega, remove_idx)
            num_oscillators -= 1
            # No need to adjust controlled_idx since it's at index 0

    # Connect the mouse events
    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
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
