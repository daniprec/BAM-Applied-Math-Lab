import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def gierer_meinhardt_pde(
    t: float,
    uv: np.ndarray,
    gamma: float = 1,
    a: float = 0.40,
    b: float = 1.00,
    d: float = 30,
) -> np.ndarray:
    """PDEs for the Gierer-Meinhardt model in 2D

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

    Returns
    -------
    np.ndarray
        Array containing the time derivatives of u and v
    """
    # Compute the Laplacian via finite differences
    # --- FILL HERE ---

    # Compute the ODEs
    # --- FILL HERE ---

    return np.array([du_dt, dv_dt])


def run_simulation(
    gamma: float = 1,
    b: float = 1.00,
    dt: float = 0.001,
    length_x: int = 20,
    length_y: int = 50,
    seed: int = 0,
):
    """Animate the simulation of the Gierer-Meinhardt model in 1D

    Parameters
    ----------
    dt : float, optional
        Time step size, by default 0.001
    anim_speed : int, optional
        Number of iterations per frame, by default 10
    length_x : float
        L1 domain length, by default 20
    length_y : float
        L2 domain length, by default 50
    seed : int, optional
        Random seed for reproducibility, by default 0
    """
    # Fix the random seed for reproducibility
    np.random.seed(seed)

    # Initialize the u and v fields
    # Add 1% amplitude additive noise, to break the symmetry
    # --- FILL HERE ---

    # ------------------------------------------------------------------------ #
    # INITIALIZE PLOT
    # ------------------------------------------------------------------------ #

    # Create a canvas
    fig, axs = plt.subplots(figsize=(10, 5), nrows=1, ncols=1)
    ax_uv: Axes = axs[0]  # U vs V

    # ------------------------------------------------------------------------ #
    # U-V PLANE
    # ------------------------------------------------------------------------ #

    # Dynamic elements: image and text
    im = ax_uv.imshow(
        uv[1],
        interpolation="bilinear",
        origin="lower",
        extent=[0, length_y, 0, length_x],
    )

    # Static elements: Plot limits, title and labels
    ax_uv.set_xlabel("y")
    ax_uv.set_ylabel("x")
    ax_uv.set_title("Gierer-Meinhardt Model (2D)")

    # ------------------------------------------------------------------------ #
    # ANIMATION
    # ------------------------------------------------------------------------ #

    # This function will be called at each frame of the animation, updating the line objects

    def update_animation(frame: int):
        # Access the variables from the outer scope
        # This allows the function to modify uv
        nonlocal uv

        # We use the Euler's method to integrate the ODEs
        # --- FILL HERE ---

        # Apply Neumann boundary conditions
        # --- FILL HERE ---

        # Update the displayed image
        im.set_array(uv[1])
        # Redefine the color limits. We make sure that the maximum value is at least
        # 0.1 to avoid noise in the image
        im.set_clim(vmin=uv[1].min(), vmax=uv[1].max() + 0.1)

        # The function must return an iterable with all the artists that have changed
        return [im]

    ani = animation.FuncAnimation(fig, update_animation, interval=1, blit=True)
    # interval: Delay between frames in milliseconds
    # blit: True to re-draw only the parts that have changed

    # Show the interactive plot
    plt.show()


if __name__ == "__main__":
    run_simulation()
