from typing import Callable, List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


def cdima(t: float, xy: np.ndarray, a: float = 10, b: float = 6) -> np.ndarray:
    """
    Defines model equations of Chlorine Dioxide-lodine-Malonic Acid Reaction.
    Reference: Strogatz, chapter 8.3, Oscillating Chemical Reactions

    Parameters
    ----------
    t : float
        Time variable.
    y : ndarray
        Array containing the variables [v, w] at time t.
    a : float
        a > 0 depends on the empirical rate constants and on the concentrations
        assumed for the slow reactants.
    b : float
        b > 0 (see a).

    Returns
    -------
    np.ndarray
        Derivatives [dx/dt, dy/dt] at time t.
    """
    x, y = xy
    dx_dt = a - x - 4 * x * y / (1 + x**2)
    dy_dt = b * x * (1 - y / (1 + x**2))
    return np.array([dx_dt, dy_dt])


def compute_nullclines(
    system_func: Callable,
    t: float = 0.0,
    limits: Tuple[float, float, float, float] = (-3.0, -3.0, 3.0, 3.0),
    num_points: int = 1000,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Computes the nullclines for a 2D dynamical system.

    Parameters
    ----------
    system_func : Callable
        Function defining the system of ODEs.
    t : float, optional
        Time variable (default is 0.0).
    limits : tuple of float, optional
        Tuple containing the x and y limits (x_min, y_min, x_max, y_max).
    num_points : int, optional
        Number of points to use in each variable range (default is 1000).
    **kwargs
        Additional keyword arguments to pass to the system function.

    Returns
    -------
    nullclines : list of tuple of ndarray
        List containing tuples of x and y coordinates of the nullclines:
        - nullclines[0]: (x values, y values) where dx/dt = 0.
        - nullclines[1]: (x values, y values) where dy/dt = 0.
    """
    x_min, y_min, x_max, y_max = limits

    # Create a grid of points
    x_values = np.linspace(x_min, x_max, num_points)
    y_values = np.linspace(y_min, y_max, num_points)
    x_grid, y_grid = np.meshgrid(x_values, y_values)

    # Evaluate the derivatives at each point
    dx_dt: np.ndarray
    dy_dt: np.ndarray
    dx_dt, dy_dt = system_func(t, [x_grid, y_grid])

    # Extract nullcline data - Find where dx_dt changes sign (zero crossings)
    zero_crossings = np.where(np.diff(np.sign(dx_dt), axis=0))
    x_nc_x = x_grid[zero_crossings]
    x_nc_y = y_grid[zero_crossings]
    # Extract nullcline data - Find where dy_dt changes sign (zero crossings)
    zero_crossings = np.where(np.diff(np.sign(dy_dt), axis=1))
    y_nc_x = x_grid[zero_crossings]
    y_nc_y = y_grid[zero_crossings]

    return [(x_nc_x, x_nc_y), (y_nc_x, y_nc_y)]


def run_interactive_plot(
    system_func: Callable,
    y0: Tuple[float, float] = (0.0, 0.0),
    t_span: Tuple[float, float] = (0.0, 100.0),
    t_step: float = 0.01,
    limits: Tuple[float, float, float, float] = (0, 0, 5, 10),
):
    """
    Runs an interactive simulation of a dynamical system with the ability to update initial conditions.

    Parameters
    ----------
    system_func : Callable
        Function that defines the model equations.
    y0 : float, optional
        Initial values of the variables (default is 0.0).
    limits : tuple of float, optional
        Tuple containing the x and y limits (x_min, y_min, x_max, y_max).
    ls_args : list of any, optional
        Additional arguments to pass to the system function
    """

    # Solve an initial value problem for a system of ODEs
    t_eval = np.arange(t_span[0], t_span[1] + t_step, t_step)
    sol = solve_ivp(system_func, t_span, y0, t_eval=t_eval, method="RK45")
    y = sol.y

    # ------------------------------------------------------------------------ #
    # INITIALIZE PLOT
    # ------------------------------------------------------------------------ #

    # Create a canvas
    fig, axs = plt.subplots(figsize=(10, 5), nrows=2, ncols=2)
    plt.tight_layout()  # Avoid overlapping subplots
    ax_phase: Axes = axs[0, 0]  # Phase plane x vs y
    ax_xt: Axes = axs[0, 1]  # x vs t
    ax_stability: Axes = axs[1, 0]  # Stability diagram a vs b

    # ------------------------------------------------------------------------ #
    # PHASE PLANE - STATIC PARTS - VECTOR FIELD
    # ------------------------------------------------------------------------ #

    # Plot phase plane -  Create a grid of points
    v_values = np.linspace(limits[0], limits[2], 20)
    w_values = np.linspace(limits[1], limits[3], 20)
    v_grid, w_grid = np.meshgrid(v_values, w_values)

    # Compute derivatives
    dvdt, dwdt = system_func(0.0, [v_grid, w_grid])

    # Plot vector field
    ax_phase.quiver(v_grid, w_grid, dvdt, dwdt, color="gray", alpha=0.5)

    # Set up the plot parameters
    ax_phase.set_xlabel("x")
    ax_phase.set_ylabel("y")
    ax_phase.set_title("Phase Plane Analysis")
    ax_phase.legend()
    ax_phase.set_xlim(limits[0], limits[2])
    ax_phase.set_ylim(limits[1], limits[3])
    ax_phase.grid(True)

    # ------------------------------------------------------------------------ #
    # PHASE PLANE - STATIC PARTS - NULLCLINES
    # ------------------------------------------------------------------------ #

    # A nullcline is a curve in the phase plane where one of the variables is constant
    # dy/dt = 0 nullcline: w = f(v)

    # Compute nullclines
    nullclines = compute_nullclines(system_func, t=0.0, limits=limits)
    v_nullcline = nullclines[0]
    w_nullcline = nullclines[1]

    # Plot nullclines
    ax_phase.scatter(
        v_nullcline[0], v_nullcline[1], c="b", s=1, label="dv/dt = 0 Nullcline"
    )
    ax_phase.scatter(
        w_nullcline[0], w_nullcline[1], c="r", s=1, label="dw/dt = 0 Nullcline"
    )

    # ------------------------------------------------------------------------ #
    # PHASE PLANE - BACKGROUND - FIXED POINT
    # ------------------------------------------------------------------------ #

    # A fixed point is a point where the system is at equilibrium (dy/dt = 0)
    # To find the fixed point, we will use the fsolve function from scipy
    # fsolve expects a function with a single input (y)
    # Our function has two inputs (t, y), so we will create a new function
    # that only takes y as input and calls the original function with t=None
    def func(x: np.ndarray) -> np.ndarray:
        return system_func(None, x)

    # Initial guesses for fixed points
    x0 = (0.0, 0.0)
    # Find the roots of the (non-linear) equations defined by func(x) = 0
    fp = fsolve(func, x0)
    ax_phase.plot(fp[0], fp[1], "ko", markersize=8)
    ax_phase.text(fp[0] + 0.1, fp[1] + 0.1, f"({fp[0]:.2f}, {fp[1]:.2f})")

    # ------------------------------------------------------------------------ #
    # X VS T - STATIC PARTS
    # ------------------------------------------------------------------------ #

    ax_xt.set_title("Time Series")
    ax_xt.set_xlabel("Time (t)")
    ax_xt.set_ylabel("y")
    ax_xt.set_xlim(t_span)
    ax_xt.set_ylim(limits[1], limits[3])

    # ------------------------------------------------------------------------ #
    # ANIMATION
    # ------------------------------------------------------------------------ #

    # Initialize the line object for animation on phase plane
    (line_phase,) = ax_phase.plot([], [], lw=2)

    # Initialize the line object for animation on x vs t
    (line_xt,) = ax_xt.plot([], [], lw=2)

    def animate(
        i: int,
        y: np.ndarray,
        line_phase: Line2D,
        line_xt: Line2D,
    ) -> Tuple[Line2D]:
        """
        Animation function to update the line object.
        """
        line_phase.set_data(y[0][:i], y[1][:i])
        line_xt.set_data(t_eval[: i + 1], y[1][: i + 1])
        return (line_phase, line_xt)

    # Create the animation
    ani = animation.FuncAnimation(
        fig, animate, fargs=(y, line_phase, line_xt), interval=1, blit=True
    )

    # ------------------------------------------------------------------------ #
    # PHASE PLANE - INTERACTION
    # ------------------------------------------------------------------------ #

    def update_simulation(event: MouseEvent):
        # The initial condition is set to the mouse click position
        y0 = [event.xdata, event.ydata]
        # The click only works if it is inside the phase plane
        if event.inaxes == ax_phase:
            # Solve the ODEs with the new initial condition
            sol = solve_ivp(system_func, t_span, y0, t_eval=t_eval, method="RK45")
            y = sol.y
            # Stop the current animation, reset the frame sequence, and start a new animation
            ani.event_source.stop()
            ani.new_frame_seq()
            ani.frame_seq = ani.new_frame_seq()
            ani._args = (y, line_phase, line_xt)
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
    run_interactive_plot(cdima)


if __name__ == "__main__":
    main()
