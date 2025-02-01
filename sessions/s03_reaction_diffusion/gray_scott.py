import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def laplacian(uv: np.ndarray) -> np.ndarray:
    """
    Compute the Laplacian of the u and v fields using a 5-point finite
    difference scheme: considering each point and its immediate neighbors in
    the up, down, left, and right directions.

    Reference: https://en.wikipedia.org/wiki/Five-point_stencil

    Parameters
    ----------
    uv : np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the values of
        u and v.
    Returns
    -------
    np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the Laplacian
        of u and v.
    """
    lap = -4 * uv

    # Immediate neighbors (up, down, left, right)
    lap += np.roll(uv, shift=1, axis=0)  # up
    lap += np.roll(uv, shift=-1, axis=0)  # down
    lap += np.roll(uv, shift=1, axis=1)  # left
    lap += np.roll(uv, shift=-1, axis=1)  # right
    return lap


def laplacian_9pt(uv: np.ndarray) -> np.ndarray:
    """
    Compute the Laplacian of the u and v fields using a 9-point finite
    difference scheme (Patra-Karttunen), considering each point and its
    immediate neighbors, including diagonals.

    Reference: https://en.wikipedia.org/wiki/Nine-point_stencil

    Parameters
    ----------
    uv : np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the values of u and v.

    Returns
    -------
    np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the Laplacian of u and v.
    """
    # Weights for the 9-point stencil (Patra-Karttunen)
    center_weight = -20 / 6
    neighbor_weight = 4 / 6
    diagonal_weight = 1 / 6

    lap = center_weight * uv

    # Shifted arrays for immediate neighbors
    up = np.roll(uv, shift=1, axis=0)
    down = np.roll(uv, shift=-1, axis=0)

    # Immediate neighbors (up, down, left, right)
    lap += neighbor_weight * up  # up
    lap += neighbor_weight * down  # down
    lap += neighbor_weight * np.roll(uv, shift=1, axis=1)  # left
    lap += neighbor_weight * np.roll(uv, shift=-1, axis=1)  # right

    # Diagonal neighbors
    lap += diagonal_weight * np.roll(up, shift=1, axis=1)  # up-left
    lap += diagonal_weight * np.roll(up, shift=-1, axis=1)  # up-right
    lap += diagonal_weight * np.roll(down, shift=1, axis=1)  # down-left
    lap += diagonal_weight * np.roll(down, shift=-1, axis=1)  # down-right

    return lap


def gray_scott_ode(
    t: float,
    uv: np.ndarray,
    d1: float = 0.1,
    d2: float = 0.05,
    f: float = 0.040,
    k: float = 0.060,
    stencil: int = 5,
) -> np.ndarray:
    """
    Update the u and v fields using the Gray-Scott model with explicit Euler
    time integration, where the fields are updated based on their current values
    and the calculated derivatives.

    Reference: https://itp.uni-frankfurt.de/~gros/StudentProjects/Projects_2020/projekt_schulz_kaefer/

    Parameters
    ----------
    uv : np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the values of
        u and v.
    d1 : float, optional
        Diffusion rate of u, default is 0.1.
    d2 : float, optional
        Diffusion rate of v, default is 0.05.
    f : float, optional
        Feed rate (at which u is fed into the system), default is 0.040.
    k : float, optional
        Kill rate (at which v is removed from the system), default is 0.060.
    stencil : int, optional
        Stencil to use for the Laplacian computation. Use 5 or 9, default is 5.
    """

    # Extract the matrices for substances u and v
    u = uv[:, :, 0]
    v = uv[:, :, 1]

    # Compute the Laplacian of u and v
    if stencil == 5:
        lap = laplacian(uv)
    elif stencil == 9:
        lap = laplacian_9pt(uv)
    else:
        raise ValueError("Invalid stencil value. Use 5 or 9.")

    # Extract the Laplacian matrices for u and v
    lu = lap[:, :, 0]
    lv = lap[:, :, 1]

    uv2 = u * v * v

    # Gray-Scott equations
    du_dt = d1 * lu - uv2 + f * (1 - u)
    dv_dt = d2 * lv + uv2 - (f + k) * v

    return np.stack([du_dt, dv_dt], axis=-1)


def animate_simulation(
    grid_size: int = 250,
    dt: int = 1,
    boundary_conditions: str = "neumann",
    anim_speed: int = 100,
    cmap: str = "jet",
):
    """
    Animate the Gray-Scott model simulation.

    Parameters
    ----------
    grid_size : int
        Size of the grid.
    dt : int
        Time step.
    boundary_conditions : str
        Boundary conditions to apply. Use 'neumann' or 'periodic'.
    anim_speed : int
        Animation speed. Number of iterations per frame.
    cmap : str
        Colormap to use for the plot, by default 'jet'.
    """
    # Initialize the u and v fields
    uv = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
    uv[:, :, 0] = 1.0  # Initialize u to 1.0, v to 0.0

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(uv[:, :, 1], cmap=cmap, interpolation="bilinear", vmin=0, vmax=0.4)
    plt.axis("off")

    # Pause parameter - Will be toggled by pressing the space bar
    pause = True

    def update_frame(_):
        """This function is called by the animation module at each frame.
        It updates the u and v fields based on the Gray-Scott model equations.
        Returns the updated elements of the plot. This in combination with the
        blit=True parameter in FuncAnimation will only update the changed elements
        of the plot, making the animation faster.
        """
        # Access the pause variable from the outer scope
        nonlocal pause
        if pause:
            return [im]

        # Access the uv variable from the outer scope
        nonlocal uv
        for _ in range(anim_speed):
            # Solve an initial value problem for a system of ODEs via Euler's method
            uv = uv + gray_scott_ode(_, uv) * dt
            # Apply boundary conditions
            if boundary_conditions == "neumann":
                # Neumann - zero flux boundary conditions
                uv[0, :] = uv[1, :]
                uv[-1, :] = uv[-2, :]
                uv[:, 0] = uv[:, 1]
                uv[:, -1] = uv[:, -2]
            elif boundary_conditions == "periodic":
                # Periodic conditions are already implemented in the laplacian function
                pass
            else:
                raise ValueError(
                    "Invalid boundary_conditions value. Use 'neumann' or 'periodic'."
                )

        im.set_array(uv[:, :, 1])
        return [im]

    def on_click(event, r: int = 20):
        """This function is called when the user clicks on the plot.
        It either adds a source of v or removes it, depending on the mouse button clicked.

        Parameters
        ----------
        event
            The event object.
        r : int, optional
            Radius of the source, by default 20.
        """
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x = int(event.xdata)
        y = int(event.ydata)

        # Left click?
        if event.button == 1:
            u_new = 0.50
            v_new = 0.50
        # Right click?
        elif event.button == 3:
            u_new = 1.0
            v_new = 0.0
        else:
            return

        uv[y - r : y + r, x - r : x + r, 0] = u_new
        uv[y - r : y + r, x - r : x + r, 1] = v_new

        # Update the displayed image
        im.set_array(uv[:, :, 1])

    def on_keypress(event):
        """This function is called when the user presses a key.
        It pauses or resumes the simulation when the space bar is pressed.

        Parameters
        ----------
        event
            The event object.
        """
        # Pressing the space bar pauses or resumes the simulation
        if event.key == " ":
            nonlocal pause
            pause ^= True

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_keypress)

    ani = animation.FuncAnimation(fig, update_frame, interval=1, blit=True)
    plt.show()


if __name__ == "__main__":
    animate_simulation()
