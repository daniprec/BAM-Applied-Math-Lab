import sys
from typing import Callable, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# I will import the "compute_nullclines" from the script "cdima.py"
sys.path.append(".")
from sessions.s02_odes_2d.cdima import compute_nullclines


def fitzhugh_nagumo(
    t: float,
    y: np.ndarray,
    i_app: float = 0.5,
    gamma: float = 0.5,
    alpha: float = 0.1,
    epsilon: float = 0.01,
) -> np.ndarray:
    """
    Defines the FitzHugh-Nagumo (FHN) model equations.
    The FHN model describes a prototype of an excitable system (e.g., a neuron).
    It is an example of a relaxation oscillator because, if the external
    stimulus i_app exceeds a certain threshold value, the system will exhibit a
    characteristic excursion in phase space, before the variables v and w
    relax back to their rest values.
    https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model

    Parameters
    ----------
    t : float
        Time variable.
    y : ndarray
        Array containing the variables [v, w] at time t.
    i_app : float
        External stimulus current.
    a : float
        Recovery time constant.
    b : float
        Recovery time constant.
    tau : float
        Recovery time scale.
    r : float
        Recovery time scale.

    Returns
    -------
    np.ndarray
        Derivatives [dv/dt, dw/dt] at time t.
    """
    v, w = y
    fv = v * (1 - v) * (v - alpha)
    dvdt = (fv - w + i_app) / epsilon
    dwdt = v - gamma * w
    return np.array([dvdt, dwdt])


def run_interactive_plot(
    system_func: Callable,
    t_span: Tuple[float, float] = (0.0, 10.0),
    t_step: float = 0.01,
    limits: Tuple[float, float, float, float] = (-1, -0.05, 1, 1),
):
    """
    Runs an interactive simulation of a dynamical system with the ability to update initial conditions.

    Parameters
    ----------
    system_func : Callable
        Function that defines the model equations.
    t_span : tuple of float, optional
        Tuple containing the initial and final time (default is (0.0, 100.0)).
    t_step : float, optional
        Time step for the simulation (default is 0.01).
    limits : tuple of float, optional
        Tuple containing the x and y limits (x_min, y_min, x_max, y_max).
    """
    # Initialize the systems with lists
    # We will use its mutable properties to update the initial conditions
    y0 = [0.0, 0.0]
    args = [0.1, 0.5, 0.1, 0.01]

    t_eval = np.arange(t_span[0], t_span[1], t_step)

    # ------------------------------------------------------------------------ #
    # INITIALIZE PLOT
    # ------------------------------------------------------------------------ #

    # Create a canvas
    fig, axs = plt.subplots(figsize=(10, 5), nrows=2, ncols=2, height_ratios=[5, 1])
    plt.tight_layout()  # Avoid overlapping subplots
    ax_phase: Axes = axs[0, 0]  # Phase plane x vs y
    ax_xt: Axes = axs[0, 1]  # x vs t
    ax_stability: Axes = axs[1, 0]  # Stability diagram a vs b
    axs[1, 1].axis("off")

    # ------------------------------------------------------------------------ #
    # PHASE PLANE
    # ------------------------------------------------------------------------ #

    # Many of the lines we want to plot will be updated during the animation
    # We initialize them as empty lists (similar to what we do in Streamlit)

    # Initialize the line object for animation on phase plane
    (plot_trajectory,) = ax_phase.plot([], [], lw=2)

    # A nullcline is a curve in the phase plane where one of the variables is constant
    # dy/dt = 0 nullcline: w = f(v)
    (plot_xnull,) = ax_phase.plot(
        [],
        [],
        linestyle="None",
        marker="o",
        markersize=0.1,
        color="blue",
        label="dx/dt = 0 Nullcline",
    )
    (plot_ynull,) = ax_phase.plot(
        [],
        [],
        linestyle="None",
        marker="o",
        markersize=0.1,
        color="red",
        label="dy/dt = 0 Nullcline",
    )

    # A fixed point is a point where the system is at equilibrium (dy/dt = 0)
    (plot_fixedpoint,) = ax_phase.plot([], [], "ko", markersize=8)

    # Set up the plot parameters
    ax_phase.set_xlabel("x")
    ax_phase.set_ylabel("y")
    ax_phase.set_title("Phase Plane Analysis")
    ax_phase.legend()
    ax_phase.set_xlim(limits[0], limits[2])
    ax_phase.set_ylim(limits[1], limits[3])

    # ------------------------------------------------------------------------ #
    # X VS T
    # ------------------------------------------------------------------------ #

    # Initialize the line object for animation on x vs t
    (plot_xt,) = ax_xt.plot([], [], lw=2)

    ax_xt.set_title("Time Series")
    ax_xt.set_xlabel("Time (t)")
    ax_xt.set_ylabel("x")
    ax_xt.set_xlim(t_span)
    ax_xt.set_ylim(limits[0], limits[2])

    # ------------------------------------------------------------------------ #
    # STABILITY DIAGRAM
    # ------------------------------------------------------------------------ #

    # Initialize the line object for animation on stability diagram
    (plot_stabpoint,) = ax_stability.plot([], [], color="red")
    ax_stability.set_xlabel("I app")
    ax_stability.set_xlim(0, 0.75)
    ax_stability.set_ylim(0, 1)

    # ------------------------------------------------------------------------ #
    # ANIMATION
    # ------------------------------------------------------------------------ #

    # First we define the animation function
    # This function will be called at each frame of the animation, updating the line objects

    def animate(
        i: int,
        y: np.ndarray,
        x_nullcline: np.ndarray,
        y_nullcline: np.ndarray,
        fp: np.ndarray,
    ) -> Tuple[Line2D]:
        """
        Animation function to update the line object.

        Inputs are the frame index and the data to update the line object.

        Output are the updated line objects.
        """
        if y is None:
            return ()

        plot_xnull.set_data(x_nullcline[0], x_nullcline[1])
        plot_ynull.set_data(y_nullcline[0], y_nullcline[1])
        plot_fixedpoint.set_data([fp[0]], [fp[1]])
        plot_trajectory.set_data(y[0][:i], y[1][:i])
        plot_xt.set_data(t_eval[: i + 1], y[0][: i + 1])
        plot_stabpoint.set_data([args[0], args[0]], [0, 1])
        return (
            plot_trajectory,
            plot_xnull,
            plot_ynull,
            plot_fixedpoint,
            plot_xt,
            plot_stabpoint,
        )

    # Create an animation object, which calls the animate function at each frame
    ani = animation.FuncAnimation(
        fig,
        animate,
        fargs=(None, None, None, None),
        interval=1,
        blit=True,
    )
    ani.event_source.stop()

    # ------------------------------------------------------------------------ #
    # INTERACTION
    # ------------------------------------------------------------------------ #

    # Second, we define a function that will be called when the user clicks on the plot
    # It will update the initial conditions and restart the animation

    def update_simulation(event: MouseEvent):
        # The click only works if it is inside the phase plane or stability diagram
        if event.inaxes == ax_phase:
            y0[0] = event.xdata
            y0[1] = event.ydata
        elif event.inaxes == ax_stability:
            args[0] = event.xdata
        else:
            return

        # Solve the ODEs with the new initial condition
        sol = solve_ivp(
            system_func, t_span, y0, t_eval=t_eval, method="RK45", args=args
        )
        y = sol.y

        # Compute nullclines
        nullclines = compute_nullclines(
            system_func, args=args, t=t_span[0], limits=limits
        )
        x_nullcline = nullclines[0]
        y_nullcline = nullclines[1]

        # To find the fixed point, we will use the fsolve function from scipy
        # fsolve expects a function with a single input (y)
        # Our function has two inputs (t, y), so we will create a new function
        # that only takes y as input and calls the original function with t=None
        def func(x: np.ndarray) -> np.ndarray:
            return system_func(None, x, *args)

        # Find the roots of the ODEs
        fp = fsolve(func, y0)

        # Stop the current animation, reset the frame sequence, and start a new animation
        ani.event_source.stop()
        ani.frame_seq = ani.new_frame_seq()
        ani._args = (y, x_nullcline, y_nullcline, fp)
        ani.event_source.start()

    # Connect the click event to the update function
    fig.canvas.mpl_connect("button_press_event", update_simulation)

    # Show the interactive plot
    plt.show()


def main():
    """
    Main function to run the interactive FitzHugh-Nagumo model simulation.
    """
    # Run interactive plot
    run_interactive_plot(fitzhugh_nagumo)


if __name__ == "__main__":
    main()
