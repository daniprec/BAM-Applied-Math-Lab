from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp


def initialize_bridge_and_pedestrians(
    num_pedestrians: int, seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates initial state and parameter arrays for:
      - The bridge: y(0) = 0, dy(0) = 0.
      - N pedestrians: phases theta_i(0) random in [0, 2pi).
        Also generate (omega_i, c_i, xi_i) for each pedestrian.

    Parameters
    ----------
    num_pedestrians : int
        Number of pedestrians on the bridge.
    seed : int, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    state0 : ndarray
        The initial state = [y, dy, theta_1, ..., theta_N].
    freq_pedestrians : ndarray
        Pedestrians' natural frequencies (omega_i).
    coupling_pedestrians : ndarray
        Pedestrians' coupling constants (c_i).
    x_positions : ndarray
        Where each pedestrian sits horizontally on the bridge (for plotting).
    """
    if seed is not None:
        np.random.seed(seed)

    # Bridge initial conditions
    y0, dy0 = 0.0, 0.0

    # Pedestrians' initial phases, e.g. random in [0, 2 pi)
    theta_init = np.random.uniform(0, 2 * np.pi, num_pedestrians)

    # Sigma measured in Ref [14]
    freq_pedestrians = np.random.normal(loc=1.0, scale=0.086, size=num_pedestrians)

    # For a simple left-to-right arrangement of pedestrians on the bridge:
    x_positions = np.linspace(0, num_pedestrians - 1, num_pedestrians)

    # Combine into one initial state vector
    state0 = np.concatenate(([y0, dy0], theta_init))

    return state0, freq_pedestrians, x_positions


def bridge_with_pedestrians_ode(
    t: float,
    state: np.ndarray,
    damping: float,
    freq_pedestrians: np.ndarray,
    coupling_strength: float,
) -> np.ndarray:
    """Reference: Eckhardt, eq. 66, 67"""
    # Unpack the current state
    y = state[0]
    dy = state[1]
    thetas = state[2:]

    # Bridge acceleration
    ddy = np.sum(np.cos(thetas)) - 2 * damping * dy - y

    # Pedestrians
    dthetas = freq_pedestrians - coupling_strength * ddy * np.cos(thetas)

    # Build the derivative of the full state
    dstate_dt = np.empty_like(state)
    dstate_dt[0] = dy  # y'
    dstate_dt[1] = ddy  # y''
    dstate_dt[2:] = dthetas  # each pedestrian's phase derivative

    return dstate_dt


def pedestrian_order_parameter(thetas: np.ndarray) -> float:
    return np.abs(np.mean(np.exp(1j * thetas)))


def run_simulation(dt: float = 0.01, interval: int = 1, seed: int = 1):
    # ------------------------------------------------------------------------
    # PARAMETERS
    # ------------------------------------------------------------------------
    num_pedestrians = 200  # number of pedestrians
    damping = 0.01  # damping ratio
    t_span = (0, dt)  # time span for each integration step
    ylim = 1  # y-axis limit for the bridge plot
    coupling_strength = 0.03  # maximum coupling constant

    # We will keep a rolling history of length 500
    max_history = 500
    time_data = np.linspace(0, (max_history - 1) * dt, max_history).tolist()
    y_data = [0.0] * max_history
    order_data = [0.0] * max_history

    # Initialize everything
    state, freq_pedestrians, x_positions = initialize_bridge_and_pedestrians(
        num_pedestrians, seed=seed
    )

    # ------------------------------------------------------------------------
    # SET UP FIGURE (2 x 2)
    # ------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 6), height_ratios=[1, 0.2])

    # Top-left: Bridge + pedestrians
    ax_bridge: Axes = axs[0, 0]
    ax_bridge.set_title("Bridge + Pedestrians")
    ax_bridge.set_xlim(-1, num_pedestrians)
    ax_bridge.set_ylim(-ylim, ylim)
    ax_bridge.set_xlabel("Bridge Span")
    ax_bridge.set_ylabel("Displacement")
    ax_bridge.grid(True)

    # A line representing the bridge
    (bridge_line,) = ax_bridge.plot([], [], "k-", lw=3)
    # A scatter for the pedestrians
    scatter_peds = ax_bridge.scatter(
        [], [], s=80, c=[], cmap="hsv", vmin=0, vmax=2 * np.pi
    )

    # Top-right: y(t) vs time
    ax_ytime: Axes = axs[0, 1]
    ax_ytime.set_title("Bridge displacement over time")
    ax_ytime.set_xlabel("Time")
    ax_ytime.set_ylabel("y(t)")
    ax_ytime.set_xlim(time_data[0], time_data[-1])
    ax_ytime.set_ylim(-ylim, ylim)
    ax_ytime.grid(True)
    (line_y,) = ax_ytime.plot(time_data, y_data, "b-")

    # Bottom-right: order parameter vs time
    ax_order: Axes = axs[1, 1]
    ax_order.set_title("Pedestrian Order Parameter")
    ax_order.set_xlabel("Time")
    ax_order.set_ylabel("r(t)")
    ax_order.set_xlim(time_data[0], time_data[-1])
    ax_order.set_ylim(0, 1)
    ax_order.grid(True)
    (line_order,) = ax_order.plot(time_data, order_data, "r-")

    # Bottom-left: area for sliders
    ax_sliders: Axes = axs[1, 0]
    ax_sliders.axis("off")

    # We create a couple of sliders: damping ratio (zeta), number of pedestrians
    # (You can add more as needed.)

    ax_damping = ax_sliders.inset_axes([0.1, 0.8, 0.8, 0.2])
    slider_damping = Slider(
        ax_damping,
        "Damping (zeta)",
        valmin=0.0,
        valmax=1.0,
        valinit=damping,
        valstep=0.01,
    )

    ax_nped = ax_sliders.inset_axes([0.1, 0.6, 0.8, 0.2])
    slider_nped = Slider(
        ax_nped,
        "N. Pedestrians",
        valmin=1,
        valmax=200,
        valinit=num_pedestrians,
        valstep=1,
    )

    ax_coupling = ax_sliders.inset_axes([0.1, 0.4, 0.8, 0.2])
    slider_coupling = Slider(
        ax_coupling,
        "Coupling strength",
        valmin=0.0,
        valmax=1.0,
        valinit=coupling_strength,
        valstep=0.01,
    )

    # ------------------------------------------------------------------------
    # ANIMATION UPDATE FUNCTION
    # ------------------------------------------------------------------------
    def update(frame: int):
        nonlocal state, damping, coupling_strength

        # Integrate from sim_time to sim_time + dt
        sol = solve_ivp(
            bridge_with_pedestrians_ode,
            t_span,
            state,
            args=(damping, freq_pedestrians, coupling_strength),
        )
        state = sol.y[:, -1]

        # Shift the rolling data
        y_data.pop(0)
        order_data.pop(0)

        # Append new
        y_data.append(state[0])
        # Compute current order param
        thetas = state[2:]
        r_now = pedestrian_order_parameter(thetas)
        order_data.append(r_now)

        # Update lines
        line_y.set_data(time_data, y_data)
        line_order.set_data(time_data, order_data)

        # Update the bridge line: from x=0..N-1, all at height y(t)
        # We'll just draw a straight line across:
        bridge_line.set_data([-1, num_pedestrians], [state[0], state[0]])

        # Update pedestrian scatter
        xvals = x_positions
        yvals = np.sin(thetas) * np.abs(state[0])
        # We'll color them by (theta mod 2pi)
        cvals = thetas % (2 * np.pi)

        scatter_peds.set_offsets(np.column_stack([xvals, yvals]))
        scatter_peds.set_array(cvals)

        return (line_y, line_order, bridge_line, scatter_peds)

    ani = animation.FuncAnimation(fig, update, blit=True, interval=interval)

    # ------------------------------------------------------------------------
    # SLIDERS UPDATE
    # ------------------------------------------------------------------------

    def on_slider_change(_):
        nonlocal damping, coupling_strength
        ani.event_source.stop()
        damping = slider_damping.val
        coupling_strength = slider_coupling.val
        ani.event_source.start()

    slider_damping.on_changed(on_slider_change)
    slider_coupling.on_changed(on_slider_change)

    def on_slider_nped_change(_):
        """
        If we change zeta or N, we should re-initialize the system
        so the changes take effect. This is similar to the Kuramoto code.
        """
        nonlocal state, y_data, order_data, time_data, num_pedestrians
        nonlocal freq_pedestrians, x_positions

        ani.event_source.stop()

        # We read the new N (int) and zeta (float). modal_mass, eigenfreq remain same for now
        num_pedestrians = int(slider_nped.val)

        # Re-initialize everything
        state, freq_pedestrians, x_positions = initialize_bridge_and_pedestrians(
            num_pedestrians, seed
        )

        # Reset rolling arrays
        time_data[:] = np.linspace(0, (max_history - 1) * dt, max_history)
        y_data[:] = [0.0] * max_history
        order_data[:] = [0.0] * max_history

        # Reset plot limits
        ax_bridge.set_xlim(-1, num_pedestrians)

        ani.event_source.start()

    slider_nped.on_changed(on_slider_nped_change)

    # ------------------------------------------------------------------------
    # RESTART ON SPACE KEY
    # ------------------------------------------------------------------------
    def on_key_press(event):
        if event.key == " ":
            on_slider_nped_change(None)

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    ax_sliders.text(
        0.0, 0.0, "Press SPACE to restart the simulation", fontsize=12, color="red"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation()
