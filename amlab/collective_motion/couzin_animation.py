import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

sys.path.append(".")  # Add parent directory to path to import couzin.py
from amlab.collective_motion.couzin import couzin_equations, initialize_particles_couzin


def run_simulation(dt: float = 0.1):
    """
    Run the animation of the Couzin model.
    Reference: Couzin et al. 2002.

    Parameters
    ----------
    dt : float, optional
        Time step, default is 0.1.
    """
    # Default parameters (consistent with couzin_equations defaults)
    num_boids = 50
    box_size = 25.0
    v0 = 3.0
    radius_repulsion = 1.0
    radius_alignment = 6.0
    radius_attraction = 14.0
    noise = 0.05

    # Sliding window length for order parameter
    ORDER_WINDOW = 3000
    # Tail length for trajectory visualization
    TAIL_LEN = 5

    # Initialize particles
    xy, theta = initialize_particles_couzin(num_boids, box_size=box_size)
    ls_order_param = [0.0] * ORDER_WINDOW

    # Plot particles to the left, order parameter to the right
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), height_ratios=[4, 1])
    ax_plane: Axes = axs[0, 0]
    ax_order: Axes = axs[0, 1]
    ax_sliders: Axes = axs[1, 0]
    ax_dummy: Axes = axs[1, 1]

    # Show last TAIL_LEN positions of each particle (tail effect)
    xy_tail = np.repeat(xy[:, :, np.newaxis], TAIL_LEN, axis=2)

    (plt_particles,) = ax_plane.plot(
        xy_tail[0].flatten(),
        xy_tail[1].flatten(),
        color="cyan",
        linestyle="",
        marker=".",
        markersize=2,
    )
    # Current positions (black, bigger markers)
    (plt_current,) = ax_plane.plot(
        xy[0],
        xy[1],
        linestyle="",
        marker="o",
        color="blue",
        markersize=4,  # twice as big as 2
    )
    ax_plane.set_xlim(0, box_size)
    ax_plane.set_ylim(0, box_size)
    ax_plane.set_aspect("equal")

    (line_order_param,) = ax_order.plot([], [])
    ax_order.set_xlim(0, ORDER_WINDOW - 1)
    ax_order.set_ylim(0, 1)
    ax_order.set_xlabel("Time (frames)")
    ax_order.set_ylabel("Order parameter (r)")
    ax_order.grid(True)

    ax_sliders.axis("off")
    ax_dummy.axis("off")

    def update_animation(frame: int):
        nonlocal xy, xy_tail, theta, v0
        nonlocal radius_repulsion, radius_alignment, radius_attraction
        nonlocal box_size, ls_order_param

        # Enforce zone ordering (zor < zoo < zoa) for the model,
        # regardless of slider abuse
        radii = np.sort(
            np.array(
                [radius_repulsion, radius_alignment, radius_attraction], dtype=float
            )
        )
        rr, ro, ra = radii

        xy_new, theta_new = couzin_equations(
            xy,
            theta,
            v0=v0,
            dt=dt,
            box_size=box_size,
            radius_repulsion=rr,
            radius_alignment=ro,
            radius_attraction=ra,
            noise_std=noise,
        )
        xy[:] = xy_new
        theta[:] = theta_new

        # Update tails
        xy_tail = np.roll(xy_tail, shift=-1, axis=2)
        xy_tail[:, :, -1] = xy
        plt_particles.set_data(xy_tail[0].flatten(), xy_tail[1].flatten())

        # Update current positions
        plt_current.set_data(xy[0], xy[1])

        # Proper polarization / order parameter:
        # r = | (1/N) sum_i v_i / |v_i| |, here |v_i| = 1
        vx = np.cos(theta)
        vy = np.sin(theta)
        order_param = np.sqrt(vx.mean() ** 2 + vy.mean() ** 2)

        ls_order_param.append(order_param)
        ls_order_param = ls_order_param[-ORDER_WINDOW:]

        x_vals = np.arange(len(ls_order_param))
        line_order_param.set_data(x_vals, ls_order_param)

        return (plt_particles, plt_current, line_order_param)

    ani = animation.FuncAnimation(
        fig,
        update_animation,
        interval=dt * 500.0,  # ms between frames
        blit=True,
    )

    # --- Sliders ---

    ax_num_boids = ax_sliders.inset_axes([0.0, 1.2, 0.8, 0.1])
    ax_box_size = ax_sliders.inset_axes([0.0, 1.0, 0.8, 0.1])
    ax_v0 = ax_sliders.inset_axes([0.0, 0.8, 0.8, 0.1])
    ax_radius_repulsion = ax_sliders.inset_axes([0.0, 0.6, 0.8, 0.1])
    ax_radius_alignment = ax_sliders.inset_axes([0.0, 0.4, 0.8, 0.1])
    ax_radius_attraction = ax_sliders.inset_axes([0.0, 0.2, 0.8, 0.1])

    # Make valinit consistent with slider ranges
    slider_num_boids = plt.Slider(
        ax_num_boids,
        "Number of boids",
        10,
        500,
        valinit=num_boids,
        valstep=10,
    )
    slider_box_size = plt.Slider(
        ax_box_size,
        "Dimension",
        5,
        50,
        valinit=box_size,
        valstep=1,
    )
    slider_v0 = plt.Slider(
        ax_v0,
        "Speed",
        0.0,
        10.0,
        valinit=v0,
        valstep=0.1,
    )
    slider_radius_repulsion = plt.Slider(
        ax_radius_repulsion,
        "Repulsion radius",
        0.0,
        5.0,
        valinit=radius_repulsion,
        valstep=0.1,
    )
    slider_radius_alignment = plt.Slider(
        ax_radius_alignment,
        "Alignment radius",
        0.0,
        15.0,
        valinit=radius_alignment,
        valstep=0.1,
    )
    slider_radius_attraction = plt.Slider(
        ax_radius_attraction,
        "Attraction radius",
        0.0,
        30.0,
        valinit=radius_attraction,
        valstep=0.1,
    )

    def update_sliders(_):
        nonlocal v0, box_size
        nonlocal radius_repulsion, radius_alignment, radius_attraction

        ani.event_source.stop()

        v0 = slider_v0.val
        box_size = slider_box_size.val
        radius_repulsion = slider_radius_repulsion.val
        radius_alignment = slider_radius_alignment.val
        radius_attraction = slider_radius_attraction.val

        # Update axes limits for new box size
        ax_plane.set_xlim(0, box_size)
        ax_plane.set_ylim(0, box_size)
        ax_plane.set_aspect("equal")

        ani.event_source.start()

    slider_box_size.on_changed(update_sliders)
    slider_v0.on_changed(update_sliders)
    slider_radius_repulsion.on_changed(update_sliders)
    slider_radius_alignment.on_changed(update_sliders)
    slider_radius_attraction.on_changed(update_sliders)

    def update_num_boids(_):
        nonlocal xy, theta, num_boids, plt_particles, xy_tail

        ani.event_source.stop()

        num_boids = int(slider_num_boids.val)
        xy, theta = initialize_particles_couzin(num_boids, box_size=box_size)
        xy_tail = np.repeat(xy[:, :, np.newaxis], TAIL_LEN, axis=2)
        plt_particles.set_data(xy_tail[0].flatten(), xy_tail[1].flatten())

        ani.event_source.start()

    slider_num_boids.on_changed(update_num_boids)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation()
