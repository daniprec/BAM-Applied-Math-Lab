import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent


def gierer_meinhardt_ode(
    t: float,
    uv: np.ndarray,
    gamma: float = 1,
    a: float = 0.40,
    b: float = 1.00,
    d: float = 30,
    dx: float = 1,
) -> np.ndarray:
    """ODEs for the Gierer-Meinhardt model in 1D

    Parameters
    ----------
    t : float
        Time variable (not used in the function)
    uv : np.ndarray
        Array containing the concentrations of u and v
    gamma : float
        Reaction rate parameter, by default 1
    a : float
        Reaction rate parameter, by default 0.40
    b : float
        Reaction rate parameter, by default 1.00
    d : float
        Diffusion rate parameter of v, by default 30
    dx : float
        Spatial step size, by default 1

    Returns
    -------
    np.ndarray
        Array containing the time derivatives of u and v
    """
    # Compute the Laplacian via finite differences
    lap = -2 * uv
    lap += np.roll(uv, shift=1, axis=1)  # left
    lap += np.roll(uv, shift=-1, axis=1)  # right
    lap /= dx**2

    # Extract the matrices for u and v
    u, v = uv
    lu, lv = lap

    # Compute the ODEs
    f = a - b * u + u**2 / (v)
    g = u**2 - v
    du_dt = lu + gamma * f  # D1 = 1
    dv_dt = d * lv + gamma * g  # D2= d
    return np.array([du_dt, dv_dt])


def is_turing_instability(a: float = 0.40, b: float = 1.00, d: float = 30) -> bool:
    """Check if the conditions for Turing instability are met."""
    # The Turing instability is checked in the fixed point
    # f(u, v) = g(u, v) = 0
    # By solving the system of equations, we find the fixed point in:
    u = (1 + a) / b
    v = u**2
    # Compute the necessary derivatives
    fu = -b + 2 * u / v
    fv = -(u**2) / v**2
    gu = 2 * u
    gv = -1
    # Compute the determinant of the Jacobian
    nabla = fu * gv - fv * gu
    d1d2 = 2 * np.sqrt(d) * np.sqrt(nabla)
    # Check the conditions
    cond1 = (fu + gv) < 0  # Trace of the Jacobian
    cond2 = nabla > 0  # Determinant of the Jacobian
    cond3 = (gv + d * fu) > d1d2
    cond4 = d1d2 > 0
    return cond1 & cond2 & cond3 & cond4


def animate_simulation(
    gamma: float = 1,
    b: float = 1.00,
    dx: float = 0.5,
    dt: float = 0.001,
    anim_speed: int = 100,
    length: int = 40,
    seed: int = 0,
    boundary_conditions: str = "neumann",
):
    """Animate the simulation of the Gierer-Meinhardt model in 1D

    Parameters
    ----------
    dt : float, optional
        Time step size, by default 0.001
    dx : float, optional
        Spatial step size, by default 0.5
    anim_speed : int, optional
        Number of iterations per frame, by default 10
    length : int, optional
        Length of the 1D domain, by default 40
    seed : int, optional
        Random seed for reproducibility, by default 0
    boundary_conditions : str, optional
        Boundary conditions to apply, by default "neumann"
    """
    # Initialize some variables - Can be modified by the user
    a = 0.40
    d = 20

    # Fix the random seed for reproducibility
    np.random.seed(seed)

    # Compute the number of points in the 1D domain
    lenx = int(length / dx)

    # Initialize the u and v fields
    uv = np.ones((2, lenx))
    # Add 1% amplitude additive noise, to break the symmetry
    uv += uv * np.random.randn(2, lenx) / 100

    # Initialize the x-axis
    x = np.linspace(0, length, lenx)

    # ------------------------------------------------------------------------ #
    # INITIALIZE PLOT
    # ------------------------------------------------------------------------ #

    # Create a canvas
    fig, axs = plt.subplots(figsize=(10, 5), nrows=1, ncols=2)
    ax_ad: Axes = axs[0]  # a vs d
    ax_uv: Axes = axs[1]  # U vs V

    # ------------------------------------------------------------------------ #
    # A-D PLANE
    # ------------------------------------------------------------------------ #

    # In this plane, we will plot the Turing instability conditions
    # The function we designed can work with arrays, so we will create a
    # meshgrid to compute the Turing instability in the entire plane
    arr_a = np.linspace(0, 1, 1000)
    arr_d = np.linspace(0, 100, 1000)
    mesh_a, mesh_d = np.meshgrid(arr_a, arr_d)
    mask_turing = is_turing_instability(mesh_a, b, mesh_d)

    # The following matplotlib function will plot the Turing instability region
    # We use a contour plot to show the region where the conditions are met
    ax_ad.contourf(mesh_a, mesh_d, mask_turing, cmap="coolwarm", alpha=0.5)

    # We also plot a point, that can be moved by the user
    (plot_adpoint,) = ax_ad.plot([a], [d], color="black", marker="o")

    ax_ad.set_xlabel("a")
    ax_ad.set_ylabel("d")
    ax_ad.set_title("Turing Space")

    # ------------------------------------------------------------------------ #
    # U-V PLANE
    # ------------------------------------------------------------------------ #

    ax_uv.set_xlim(0, length)
    ax_uv.set_ylim(0, 5)
    (plot_vline,) = ax_uv.plot(x, uv[1])
    ax_uv.set_xlabel("x")
    ax_uv.set_ylabel("v(x)")
    ax_uv.set_title("Gierer-Meinhardt Model (1D)")

    # ------------------------------------------------------------------------ #
    # ANIMATION
    # ------------------------------------------------------------------------ #

    # This function will be called at each frame of the animation, updating the line objects

    def animate(frame: int):
        # Access the variables from the outer scope
        nonlocal a, d, uv

        # Iterate the simulation as many times as the animation speed
        # We use the Euler's method to integrate the ODEs
        # We cannot use solve_ivp because we must impose the boundary conditions
        # at each iteration
        for _ in range(anim_speed):
            dudt = gierer_meinhardt_ode(0, uv, gamma=gamma, a=a, b=b, d=d, dx=dx)
            uv = uv + dudt * dt
            # The simulation may explode if the time step is too large
            # When this happens, we raise an error and stop the simulation
            if np.isnan(uv).any():
                raise ValueError("Simulation exploded. Reduce dt or increase dx.")
            # Apply boundary conditions
            if boundary_conditions == "neumann":
                # Neumann - zero flux boundary conditions
                uv[:, 0] = uv[:, 1]
                uv[:, -1] = uv[:, -2]
            elif boundary_conditions == "periodic":
                # Periodic conditions are already implemented in the laplacian function
                pass
            else:
                raise ValueError(
                    "Invalid boundary_conditions value. Use 'neumann' or 'periodic'."
                )

        # Update the plot
        plot_vline.set_ydata(uv[1])
        plot_adpoint.set_data([a], [d])

        # The function must return an iterable with all the artists that have changed
        return [plot_vline, plot_adpoint]

    ani = animation.FuncAnimation(fig, animate, interval=1, blit=True)

    # ------------------------------------------------------------------------ #
    # INTERACTION
    # ------------------------------------------------------------------------ #

    # We define a function that will be called when the user clicks on the plot
    # It will update the initial conditions and restart the animation

    def update_simulation(event: MouseEvent):
        # Access the a and d variables from the outer scope and modify them
        nonlocal a, d, uv

        # The click only works if it is inside the phase plane or stability diagram
        if event.inaxes == ax_ad:
            a = event.xdata
            d = event.ydata
        else:
            return

        # Add 1% amplitude additive noise, to break the symmetry
        uv += uv * np.random.randn(2, lenx) / 100

        # Stop the current animation, reset the frame sequence, and start a new animation
        ani.event_source.stop()
        ani.frame_seq = ani.new_frame_seq()
        ani.event_source.start()

    # Connect the click event to the update function
    fig.canvas.mpl_connect("button_press_event", update_simulation)

    # Show the interactive plot
    plt.show()


if __name__ == "__main__":
    animate_simulation()
