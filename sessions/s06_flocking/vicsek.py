import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from matplotlib.axes import Axes


def initialize_particles(n: int, d: float = 25) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize the state of the particles.

    Parameters
    ----------
    n : int
        Number of particles.
    d : float, optional
        Dimension of the space (default is 25).

    Returns
    -------
    np.ndarray
        Initial positions of the particles.
    np.ndarray
        Initial angle of the particles, in radians.
    """
    # Random initial theta
    theta = np.random.uniform(0, 2 * np.pi, n)
    # Random initial x, y xy
    xy = np.random.uniform(0, d, (2, n))
    return xy, theta


def vicsek_equations(
    xy: np.ndarray,
    theta: np.ndarray,
    dt: float = 1,
    eta: float = 0.1,
    d: float = 25,
    r: float = 1,
    v0: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Update the state of the particles based on the Vicsek model.

    Parameters
    ----------
    xy : np.ndarray
        Position of the particles.
    theta : np.ndarray
        Angle of the particles.
    dt : float, optional
        Time step, default is 1 (standard convention).
    eta : float, optional
        Noise parameter, default is 0.1.
    d : float, optional
        Dimension of the space, default is 25.
    r : float, optional
        Interaction radius, default is 1 (standard convention).
    v0 : float, optional
        Speed of the particles, default is 0.03.

    Returns
    -------
    np.ndarray
        Updated position of the particles.
    np.ndarray
        Updated angle of the particles.
    """
    # Compute distance matrix and neighbor matrix
    d_matrix = scipy.spatial.distance.pdist(xy.T)
    d_matrix = scipy.spatial.distance.squareform(d_matrix)
    neighbors = d_matrix <= r
    # Compute mean angle of neighbors
    neighbor_mean_theta = theta @ neighbors / np.sum(neighbors, axis=1)
    # Update angle
    theta = neighbor_mean_theta + eta * np.pi * np.random.uniform(-1, 1, len(theta))

    # Update position
    v = v0 * np.array([np.cos(theta), np.sin(theta)])
    xy = xy + dt * v
    # Boundary conditions: periodic
    xy = np.mod(xy, d)

    return xy, theta


def vicsek_order_parameter(theta: np.ndarray) -> float:
    """
    Compute the order parameter of the Vicsek model.

    Parameters
    ----------
    theta : np.ndarray
        Angle of the particles.

    Returns
    -------
    float
        Order parameter of the Vicsek model.
    """
    return np.abs(np.mean(np.exp(1j * theta)))


def run_animation(
    n: int = 300,
    v0: float = 0.03,
    dt: float = 1,
    r: float = 1,
):
    """
    Run the animation of the Vicsek model.
    Reference: Topaz, Chad M., Lori Ziegelmeier, and Tom Halverson. 2015.
    "Topological Data Analysis of Biological Aggregation Models."
    PLOS ONE 10 (5): e0126383. https://doi.org/10/f7mp7k

    Parameters
    ----------
    n : int, optional
        Number of particles, default is 300.
    v0 : float, optional
        Speed of the particles, default is 0.03.
    dt : float, optional
        Time step, default is 1 (standard convention).
    r : float, optional
        Interaction radius, default is 1 (standard convention).
    """
    # Initialize parameters (will be changed with sliders)
    eta = 0.1
    d = 25

    # Initialize particles
    xy, theta = initialize_particles(n, d=d)
    ls_order_param = [0] * 3000
    dict_noise = {}

    # Plot particles to the left, order parameter to the right
    fig, axs = plt.subplots(2, 2, figsize=(10, 5), height_ratios=[4, 1])
    ax_plane: Axes = axs[0, 0]
    ax_order: Axes = axs[0, 1]
    ax_sliders: Axes = axs[1, 0]
    ax_noise: Axes = axs[1, 1]

    # Initialize quiver plot
    plt_particles = ax_plane.quiver(
        xy[0],
        xy[1],
        np.cos(theta),
        np.sin(theta),
        angles="xy",
    )
    ax_plane.set_xlim(0, d)
    ax_plane.set_ylim(0, d)
    ax_plane.set_aspect("equal")

    # Initialize order parameter
    (line_order_param,) = ax_order.plot([], [])
    ax_order.set_xlim(0, 3000)
    ax_order.set_ylim(0, 1)
    ax_order.set_xlabel("Time")
    ax_order.set_ylabel("Order parameter (r)")
    ax_order.grid(True)

    # Initialize order parameter vs noise
    (line_noise,) = ax_noise.plot([], [], color="red", marker="o", linestyle="--")
    ax_noise.set_xlim(0, 1)
    ax_noise.set_ylim(0, 1)
    ax_noise.set_xlabel("Noise (eta)")
    ax_noise.set_ylabel("Order param (r)")

    # Clear axis for the sliders
    ax_sliders.axis("off")

    # --------------------------------
    # ANIMATION
    # --------------------------------

    def update_animation(frame: int):
        nonlocal xy, theta, eta, d, dict_noise
        xy, theta = vicsek_equations(xy, theta, v0=v0, dt=dt, r=r, d=d, eta=eta)

        # Update quiver plot
        plt_particles.set_offsets(xy.T)
        plt_particles.set_UVC(np.cos(theta), np.sin(theta))

        # Update order parameter
        ls_order_param.append(vicsek_order_parameter(theta))
        ls_order_param.pop(0)
        line_order_param.set_data(range(3000), ls_order_param)

        # Average the last 1000 values to get the order parameter
        order_param = np.mean(ls_order_param[-1000:])
        dict_noise[eta] = order_param
        dict_noise = dict(sorted(dict_noise.items()))
        line_noise.set_data(*zip(*dict_noise.items()))
        return plt_particles, line_order_param, line_noise

    ani = animation.FuncAnimation(fig, update_animation, interval=0, blit=True)

    # --------------------------------
    # SLIDERS
    # --------------------------------

    # Add sliders
    ax_eta = ax_sliders.inset_axes([0.0, 0.3, 0.8, 0.1])
    ax_d = ax_sliders.inset_axes([0.0, 0.6, 0.8, 0.1])

    s_eta = plt.Slider(ax_eta, "Noise", 0.0, 1.0, valinit=eta, valstep=0.05)
    s_d = plt.Slider(ax_d, "Dimension", 1, 50, valinit=d, valstep=1)

    def update_sliders(val):
        nonlocal xy, eta, d
        # Pause animation
        ani.event_source.stop()

        # Update parameters with sliders
        eta = s_eta.val
        d = s_d.val

        # Update plot limits
        ax_plane.set_xlim(0, d)
        ax_plane.set_ylim(0, d)
        ax_plane.set_aspect("equal")

        # Reinitialize the animation
        ani.event_source.start()

    # --------------------------------
    # SHOW
    # --------------------------------

    s_eta.on_changed(update_sliders)
    s_d.on_changed(update_sliders)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_animation()
