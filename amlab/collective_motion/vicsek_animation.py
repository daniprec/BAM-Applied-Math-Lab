import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

sys.path.append(".")  # Add parent directory to path to import vicsek.py
from amlab.collective_motion.vicsek import (
    initialize_particles,
    vicsek_equations,
    vicsek_order_parameter,
)


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

    # Sliding window length for order parameter
    ORDER_WINDOW = 3000
    # Tail length for trajectory visualization
    TAIL_LEN = 20

    # Initialize particles
    xy, theta = initialize_particles(num_boids, box_size=box_size)
    ls_order_param = [0] * ORDER_WINDOW
    dict_noise = {}

    # Plot particles to the left, order parameter to the right
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), height_ratios=[4, 1])
    ax_plane: Axes = axs[0, 0]
    ax_order: Axes = axs[0, 1]
    ax_sliders: Axes = axs[1, 0]
    ax_noise: Axes = axs[1, 1]

    # Show last TAIL_LEN positions of each particle (tail effect)
    xy_tail = np.repeat(xy[:, :, np.newaxis], TAIL_LEN, axis=2)

    (plt_particles,) = ax_plane.plot(
        xy_tail[0].flatten(),
        xy_tail[1].flatten(),
        color="grey",
        linestyle="",
        marker=".",
        markersize=2,
    )
    (plt_current,) = ax_plane.plot(
        xy[0],
        xy[1],
        linestyle="",
        marker="o",
        color="black",
        markersize=3,
    )
    ax_plane.set_xlim(0, box_size)
    ax_plane.set_ylim(0, box_size)
    ax_plane.set_aspect("equal")

    # Initialize order parameter
    (line_order_param,) = ax_order.plot([], [])
    ax_order.set_xlim(0, ORDER_WINDOW - 1)
    ax_order.set_ylim(0, 1)
    ax_order.set_xlabel("Time")
    ax_order.set_ylabel("Order parameter (r)")
    ax_order.grid(True)

    # Initialize order parameter vs noise
    (line_noise,) = ax_noise.plot([], [], color="red", marker="o", linestyle="--")
    ax_noise.set_xlim(0, 5)
    ax_noise.set_ylim(0, 1)
    ax_noise.set_xlabel("Noise (eta)")
    ax_noise.set_ylabel("Order param (r)")

    # Clear axis for the sliders
    ax_sliders.axis("off")

    # --------------------------------
    # ANIMATION
    # --------------------------------

    def update_animation(frame: int):
        nonlocal \
            xy, \
            xy_tail, \
            theta, \
            noise_eta, \
            v0, \
            radius_interaction, \
            box_size, \
            dict_noise, \
            ls_order_param
        xy, theta = vicsek_equations(
            xy,
            theta,
            v0=v0,
            dt=dt,
            radius_interaction=radius_interaction,
            box_size=box_size,
            noise=noise_eta,
        )

        # Update tails
        xy_tail = np.roll(xy_tail, shift=-1, axis=2)
        xy_tail[:, :, -1] = xy
        plt_particles.set_data(xy_tail[0].flatten(), xy_tail[1].flatten())
        plt_current.set_data(xy[0], xy[1])

        # Update order parameter
        ls_order_param.append(vicsek_order_parameter(theta))
        ls_order_param = ls_order_param[-ORDER_WINDOW:]
        x_vals = np.arange(len(ls_order_param))
        line_order_param.set_data(x_vals, ls_order_param)

        # Average the last ORDER_WINDOW//3 values to get the order parameter (similar to Couzin)
        order_param = np.mean(ls_order_param[-ORDER_WINDOW // 3 :])
        dict_noise[noise_eta] = order_param
        dict_noise = dict(sorted(dict_noise.items()))
        if dict_noise:
            line_noise.set_data(*zip(*dict_noise.items()))
        else:
            line_noise.set_data([], [])
        return (plt_particles, plt_current, line_order_param, line_noise)

    ani = animation.FuncAnimation(fig, update_animation, interval=0, blit=True)

    # --------------------------------
    # SLIDERS
    # --------------------------------

    # Add sliders
    ax_num_boids = ax_sliders.inset_axes([0.0, 1.2, 0.8, 0.1])
    ax_radius_interaction = ax_sliders.inset_axes([0.0, 1.0, 0.8, 0.1])
    ax_noise_eta = ax_sliders.inset_axes([0.0, 0.8, 0.8, 0.1])
    ax_v0 = ax_sliders.inset_axes([0.0, 0.6, 0.8, 0.1])
    ax_box_size = ax_sliders.inset_axes([0.0, 0.4, 0.8, 0.1])

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
        ax_noise_eta, "Noise", 0.0, 5.0, valinit=noise_eta, valstep=0.01
    )
    slider_v0 = plt.Slider(ax_v0, "Speed", 0.0, 0.1, valinit=v0, valstep=0.01)
    slider_box_size = plt.Slider(
        ax_box_size, "Dimension", 1, 50, valinit=box_size, valstep=1
    )

    def update_sliders(_):
        nonlocal xy, radius_interaction, noise_eta, v0, box_size
        # Pause animation
        ani.event_source.stop()

        # Update parameters with sliders
        noise_eta = slider_noise_eta.val
        v0 = slider_v0.val
        box_size = slider_box_size.val
        radius_interaction = slider_radius_interaction.val

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

    # A special case must be done for the number of boids as it requires to reinitialize the particles
    def update_num_boids(_):
        nonlocal xy, theta, num_boids, plt_particles, xy_tail
        # Pause animation
        ani.event_source.stop()

        # Update number of boids
        num_boids = int(slider_num_boids.val)
        # Reinitialize particles
        xy, theta = initialize_particles(num_boids, box_size=box_size)
        xy_tail = np.repeat(xy[:, :, np.newaxis], TAIL_LEN, axis=2)
        # Update the Line2D object to match new number of particles
        plt_particles.set_data(xy_tail[0].flatten(), xy_tail[1].flatten())

        # Reinitialize the animation
        ani.event_source.start()

    slider_num_boids.on_changed(update_num_boids)

    # --------------------------------
    # SHOW
    # --------------------------------

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation()
