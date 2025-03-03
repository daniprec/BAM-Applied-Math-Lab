import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent


def initialize_particles(
    num_boids: int, box_size: float = 25
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize the state of the particles.

    Parameters
    ----------
    num_boids : int
        Number of particles.
    box_size : float, optional
        Dimension of the space (default is 25).

    Returns
    -------
    np.ndarray
        Initial positions of the particles.
    np.ndarray
        Initial angle of the particles, in radians.
    """
    # Random initial theta
    theta = np.random.uniform(0, 2 * np.pi, num_boids)
    # Random initial x, y xy
    xy = np.random.uniform(0, box_size, (2, num_boids))
    return xy, theta


def vicsek_equations(
    xy: np.ndarray,
    theta: np.ndarray,
    dt: float = 1,
    eta: float = 0.1,
    box_size: float = 25,
    radius_interaction: float = 1,
    v0: float = 0.03,
    xy_pred: np.ndarray = np.array([-1000, -1000]),
    radius_predator: float = 1,
    strength_predator: float = 0.1,
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
    box_size : float, optional
        Dimension of the space, default is 25.
    radius_interaction : float, optional
        Interaction radius, default is 1 (standard convention).
    v0 : float, optional
        Speed of the particles, default is 0.03.
    xy_pred : np.ndarray, optional
        Position of the predator, default is far away [-1000, -1000].
    radius_predator : float, optional
        Radius of the predator, default is 1.
    strength_predator : float, optional
        Strength of the repulsion from the predator, default is 0.1.

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
    neighbors = d_matrix <= radius_interaction
    # Compute mean angle of neighbors
    term_theta_avg = theta @ neighbors / np.sum(neighbors, axis=1)
    # Add noise
    term_noise = eta * np.pi * np.random.uniform(-1, 1, len(theta))
    # Update angle
    theta = term_theta_avg + term_noise
    theta = np.mod(theta, 2 * np.pi)

    # REPULSION FROM PREDATOR
    # Compute distances to the predator
    d_pred = np.linalg.norm(xy - xy_pred[:, np.newaxis], axis=0)
    # Mask for boids within the predator radius
    affected = d_pred <= radius_predator
    # Compute repulsion direction (away from predator)
    repulsion_angle = np.arctan2(xy[1] - xy_pred[1], xy[0] - xy_pred[0])
    term_predator = strength_predator * (repulsion_angle - theta) * affected
    # Update angle
    theta = theta + term_predator
    theta = np.mod(theta, 2 * np.pi)

    # Update position
    v = v0 * np.array([np.cos(theta), np.sin(theta)])
    xy = xy + dt * v
    # Boundary conditions: periodic
    xy = np.mod(xy, box_size)

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


def run_simulation(dt: float = 1):
    """
    Run the animation of the Vicsek model.
    Reference: Topaz, Chad M., Lori Ziegelmeier, and Tom Halverson. 2015.
    "Topological Data Analysis of Biological Aggregation Models."
    PLOS ONE 10 (5): e0126383. https://doi.org/10/f7mp7k

    Parameters
    ----------
    dt : float, optional
        Time step, default is 1 (standard convention).
    """
    # Initialize parameters (will be changed with sliders)
    num_boids = 300
    noise_eta = 0.1
    box_size = 25
    radius_interaction = 1
    v0 = 0.03
    xy_pred = np.array([-1000, -1000])
    radius_predator = 1
    strength_predator = 0.1

    # Initialize particles
    xy, theta = initialize_particles(num_boids, box_size=box_size)
    ls_order_param = [0] * 3000
    dict_noise = {}

    # Plot particles to the left, order parameter to the right
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), height_ratios=[4, 1])
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
    ax_plane.set_xlim(0, box_size)
    ax_plane.set_ylim(0, box_size)
    ax_plane.set_aspect("equal")

    # Paint circle around the predator
    circle_predator = plt.Circle(xy_pred, radius_predator, color="red", fill=False)
    ax_plane.add_artist(circle_predator)

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
        nonlocal xy, theta, noise_eta, v0, radius_interaction, box_size
        nonlocal xy_pred, radius_predator, strength_predator
        nonlocal ls_order_param, dict_noise

        xy, theta = vicsek_equations(
            xy,
            theta,
            v0=v0,
            dt=dt,
            radius_interaction=radius_interaction,
            box_size=box_size,
            eta=noise_eta,
            xy_pred=xy_pred,
            radius_predator=radius_predator,
            strength_predator=strength_predator,
        )

        # Update quiver plot
        plt_particles.set_offsets(xy.T)
        plt_particles.set_UVC(np.cos(theta), np.sin(theta))

        # Update predator circle when the predator is present
        circle_predator.set_center(xy_pred)
        circle_predator.set_radius(radius_predator)

        # Update order parameter
        ls_order_param.append(vicsek_order_parameter(theta))
        ls_order_param.pop(0)
        line_order_param.set_data(range(3000), ls_order_param)

        # Average the last 1000 values to get the order parameter
        order_param = np.mean(ls_order_param[-1000:])
        dict_noise[noise_eta] = order_param
        dict_noise = dict(sorted(dict_noise.items()))
        line_noise.set_data(*zip(*dict_noise.items()))
        return plt_particles, circle_predator, line_order_param, line_noise

    ani = animation.FuncAnimation(fig, update_animation, interval=0, blit=True)

    # --------------------------------
    # SLIDERS
    # --------------------------------

    # Create child inset axes for sliders
    ax_num_boids = ax_sliders.inset_axes([0.0, 1.2, 0.8, 0.1])
    ax_radius_interaction = ax_sliders.inset_axes([0.0, 1.0, 0.8, 0.1])
    ax_noise_eta = ax_sliders.inset_axes([0.0, 0.8, 0.8, 0.1])
    ax_v0 = ax_sliders.inset_axes([0.0, 0.6, 0.8, 0.1])
    ax_box_size = ax_sliders.inset_axes([0.0, 0.4, 0.8, 0.1])
    ax_radius_predator = ax_sliders.inset_axes([0.0, 0.2, 0.8, 0.1])
    ax_strength_predator = ax_sliders.inset_axes([0.0, 0.0, 0.8, 0.1])

    # Create sliders in the inset axes
    slider_num_boids = plt.Slider(
        ax_num_boids, "Number of boids", 100, 1000, valinit=num_boids, valstep=100
    )
    slider_radius_interaction = plt.Slider(
        ax_radius_interaction,
        "Interaction radius",
        0,
        50,
        valinit=radius_interaction,
        valstep=1,
    )
    slider_noise_eta = plt.Slider(
        ax_noise_eta, "Noise", 0.0, 0.5, valinit=noise_eta, valstep=0.01
    )
    slider_v0 = plt.Slider(ax_v0, "Speed", 0.0, 0.1, valinit=v0, valstep=0.01)
    slider_box_size = plt.Slider(
        ax_box_size, "Dimension", 1, 50, valinit=box_size, valstep=1
    )
    slider_radius_predator = plt.Slider(
        ax_radius_predator,
        "Predator radius",
        0,
        10,
        valinit=radius_predator,
        valstep=1,
    )
    slider_strength_predator = plt.Slider(
        ax_strength_predator,
        "Predator strength",
        0,
        1,
        valinit=strength_predator,
        valstep=0.1,
    )

    def update_sliders(_):
        nonlocal xy, radius_interaction, noise_eta, v0, box_size
        nonlocal radius_predator, strength_predator
        # Pause animation
        ani.event_source.stop()

        # Update parameters with sliders
        noise_eta = slider_noise_eta.val
        v0 = slider_v0.val
        box_size = slider_box_size.val
        radius_interaction = slider_radius_interaction.val
        radius_predator = slider_radius_predator.val
        strength_predator = slider_strength_predator.val

        # Update plot limits
        ax_plane.set_xlim(0, box_size)
        ax_plane.set_ylim(0, box_size)
        ax_plane.set_aspect("equal")

        # Reinitialize the animation
        ani.event_source.start()

    slider_radius_interaction.on_changed(update_sliders)
    slider_noise_eta.on_changed(update_sliders)
    slider_v0.on_changed(update_sliders)
    slider_box_size.on_changed(update_sliders)
    slider_radius_predator.on_changed(update_sliders)
    slider_strength_predator.on_changed(update_sliders)

    # A special case must be done for the number of boids as it requires to reinitialize the particles
    def update_num_boids(_):
        nonlocal xy, theta, num_boids, plt_particles
        # Pause animation
        ani.event_source.stop()

        # Update number of boids
        num_boids = int(slider_num_boids.val)
        # Reinitialize particles
        xy, theta = initialize_particles(num_boids, box_size=box_size)

        # Because the number of particles has changed, we need to redefine the quiver plot
        # This is because the number of particles on the plot is fixed at the beginning
        # and cannot be changed dynamically
        plt_particles.remove()  # Remove old quiver plot
        plt_particles = ax_plane.quiver(
            xy[0],
            xy[1],
            np.cos(theta),
            np.sin(theta),
            angles="xy",
        )

        # Reinitialize the animation
        ani.event_source.start()

    slider_num_boids.on_changed(update_num_boids)

    # --------------------------------
    # MOUSE
    # --------------------------------

    # Track the mouse position on ax_plane as the predator

    def on_mouse_move(event: MouseEvent):
        nonlocal xy_pred
        if event.inaxes == ax_plane:
            # Update predator position
            xy_pred = np.array([event.xdata, event.ydata])
        else:
            xy_pred = np.array([-1000, -1000])

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    # --------------------------------
    # SHOW
    # --------------------------------

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation()
