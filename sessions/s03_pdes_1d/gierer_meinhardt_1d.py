from typing import Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent


def gierer_meinhardt_pde(
    t: float,
    uv: np.ndarray,
    gamma: float = 1,
    a: float = 0.40,
    b: float = 1.00,
    d: float = 30,
    dx: float = 1,
) -> np.ndarray:
    """PDEs for the Gierer-Meinhardt model in 1D

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


def gierer_meinhardt_fixed_point(a: float = 0.40, b: float = 1.00) -> np.ndarray:
    """Compute the fixed point of the Gierer-Meinhardt model."""
    # The fixed point is given by f(u*, v*) = 0 and g(u*, v*) = 0
    # We can solve this system of equations analytically
    u = (1 + a) / b
    v = u**2
    return np.array([u, v])


def giere_meinhardt_derivative(
    u: float, v: float, b: float = 1.00
) -> Tuple[float, float, float, float]:
    """Compute the derivatives of the Gierer-Meinhardt model."""
    fu = -b + 2 * u / v
    fv = -(u**2) / v**2
    gu = 2 * u
    gv = -1.0
    return fu, fv, gu, gv


def is_turing_instability(a: float = 0.40, b: float = 1.00, d: float = 30) -> bool:
    """Check if the conditions for Turing instability are met."""
    # Compute the fixed point
    u_star, v_star = gierer_meinhardt_fixed_point(a, b)
    # Evaluate the derivatives at the fixed point
    fu, fv, gu, gv = giere_meinhardt_derivative(u_star, v_star, b)
    # Compute the determinant of the Jacobian
    nabla = fu * gv - fv * gu
    # Check the conditions
    cond1 = (fu + gv) < 0  # Trace of the Jacobian
    cond2 = nabla > 0  # Determinant of the Jacobian
    cond3 = (gv + d * fu) > (2 * np.sqrt(d) * np.sqrt(nabla))
    return cond1 & cond2 & cond3


def find_unstable_spatial_modes(
    a: float = 0.40,
    b: float = 1.00,
    d: float = 30.0,
    gamma: float = 1.0,
    length: float = 40.0,
    num_modes: int = 10,
    boundary_conditions: str = "neumann",
) -> list[int]:
    """
    Find the leading spatial modes (wavenumbers) from linear stability analysis.

    Parameters
    ----------
    a : float
        Reaction parameter a.
    b : float
        Reaction parameter b.
    d : float
        Diffusion coefficient for v (D2).
    length : float
        1D domain length.
    num_modes : int
        Number of wavenumbers (modes) to sample in [0, num_modes].
    boundary_conditions : str
        Boundary conditions to apply. Either 'neumann' or 'periodic'.

    Returns
    -------
    np.ndarray
        Array with the indices of the unstable modes, from largest to smallest.
    """
    # Compute the fixed point
    u_star, v_star = gierer_meinhardt_fixed_point(a, b)
    # Evaluate the derivatives at the fixed point
    fu, fv, gu, gv = giere_meinhardt_derivative(u_star, v_star, b)

    # For Neumann BC on [0,L], modes k_n = (n*pi)/L
    # We will check n=0,...,num_modes-1
    n_values = np.arange(num_modes)
    max_eigs = np.zeros(num_modes)

    for n in n_values:
        if boundary_conditions == "neumann":
            # For a 1D domain of length L with Neumann boundaries,
            # possible modes are k = n*pi/L, n = 0,1,2,...
            lambda_n = -((n * np.pi / length) ** 2)
        elif boundary_conditions == "periodic":
            lambda_n = -(((n + 1) * np.pi / length) ** 2)
        else:
            raise ValueError(
                "Invalid boundary_conditions value. Use 'neumann' or 'periodic'."
            )
        # Compute the eigenvalues of the Jacobian matrix
        a_n = np.array(
            [
                [gamma * fu + lambda_n, gamma * fv],
                [gamma * gu, gamma * gv + d * lambda_n],
            ]
        )
        sigma1, sigma2 = np.linalg.eigvals(a_n)
        # Discard complex part
        sigma1, sigma2 = sigma1.real, sigma2.real
        max_eigs[n] = max(sigma1, sigma2)

    # Sort indices from largest to smallest eigenvalue
    sorted_indices = np.argsort(max_eigs)[::-1]
    # Filter the modes that lead to Turing instability (positive eigenvalues)
    unstable_modes = sorted_indices[max_eigs[sorted_indices] > 0]
    return unstable_modes.tolist()


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

    ax_ad.grid(True)
    ax_ad.set_xlabel("a")
    ax_ad.set_ylabel("d")
    ax_ad.set_title("Turing Space")

    # ------------------------------------------------------------------------ #
    # U-V PLANE
    # ------------------------------------------------------------------------ #

    # Dynamic elements: line and text
    (plot_vline,) = ax_uv.plot(x, uv[1])
    # Initialize text objects to display the leading spatial modes
    plot_text = ax_uv.text(
        0.02 * length,
        4.95,
        "No Turing's instability",
        fontsize=12,
        verticalalignment="top",
    )
    plot_text2 = ax_uv.text(
        0.02 * length,
        4.7,
        "Click to change initial conditions",
        fontsize=12,
        verticalalignment="top",
    )

    # Static elements: Plot limits, title and labels
    ax_uv.set_xlim(0, length)
    ax_uv.set_ylim(0, 5)
    ax_uv.set_xlabel("x")
    ax_uv.set_ylabel("v(x)")
    ax_uv.set_title("Gierer-Meinhardt Model (1D)")

    # ------------------------------------------------------------------------ #
    # ANIMATION
    # ------------------------------------------------------------------------ #

    # This function will be called at each frame of the animation, updating the line objects

    def animate(frame: int, unstable_modes: list[int]):
        # Access the variables from the outer scope
        nonlocal a, d, uv

        # Iterate the simulation as many times as the animation speed
        # We use the Euler's method to integrate the ODEs
        # We cannot use solve_ivp because we must impose the boundary conditions
        # at each iteration
        for _ in range(anim_speed):
            dudt = gierer_meinhardt_pde(0, uv, gamma=gamma, a=a, b=b, d=d, dx=dx)
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
        if len(unstable_modes) == 0:
            plot_text.set_text("No Turing's instability")
            plot_text2.set_text("")
        else:
            plot_text.set_text(f"Leading spatial mode: {unstable_modes[0]}")
            ls_modes = ", ".join(map(str, unstable_modes[1:8]))
            plot_text2.set_text(f"Unstable modes: {ls_modes}")

        # The function must return an iterable with all the artists that have changed
        return [plot_vline, plot_adpoint, plot_text, plot_text2]

    ani = animation.FuncAnimation(fig, animate, fargs=([],), interval=1, blit=True)

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

        # Reset to a constant line
        uv = np.ones((2, lenx)) * uv.mean()
        # Add 1% amplitude additive noise, to break the symmetry
        uv += uv * np.random.randn(2, lenx) / 100

        unstable_modes = find_unstable_spatial_modes(
            a=a,
            b=b,
            d=d,
            gamma=gamma,
            length=length,
            boundary_conditions=boundary_conditions,
        )

        # Stop the current animation, reset the frame sequence, and start a new animation
        ani.event_source.stop()
        ani.frame_seq = ani.new_frame_seq()
        ani._args = (unstable_modes,)
        ani.event_source.start()

    # Connect the click event to the update function
    fig.canvas.mpl_connect("button_press_event", update_simulation)

    # Show the interactive plot
    plt.show()


if __name__ == "__main__":
    animate_simulation()
