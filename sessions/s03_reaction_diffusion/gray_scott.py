from typing import Tuple

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
    du: float = 0.16,
    dv: float = 0.08,
    f: float = 0.060,
    k: float = 0.062,
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
    du : float, optional
        Diffusion rate of u, default is 0.16.
    dv : float, optional
        Diffusion rate of v, default is 0.08.
    f : float, optional
        Feed rate (at which u is fed into the system), default is 0.060.
    k : float, optional
        Kill rate (at which v is removed from the system), default is 0.062.
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

    # Gray-Scott equations with explicit Euler time integration
    uvv = u * v * v
    du_dt = -uvv + f * (1 - u) + du * lu
    dv_dt = uvv - (f + k) * v + dv * lv

    # Stack the derivatives into a single array (N x N x 2)
    duv_dt = np.stack((du_dt, dv_dt), axis=-1)

    return duv_dt


def add_perturbation(
    uv: np.ndarray,
    center: Tuple[float],
    r: int = 1,
    u: float = 0.5,
    v: float = 0.25,
):
    """
    Add a perturbation to the center of the grid.

    Parameters
    ----------
    uv : np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the values of
        u and v.
    r : int, optional
        Radius of the perturbation.
    u : float, optional
        Value of u in the perturbation, default is 0.5.
    v : float, optional
        Value of v in the perturbation, default is 0.25.

    Returns
    -------
    np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the values of
        u and v with the perturbation added.
    """
    x, y = center
    uv[y - r : y + r, x - r : x + r, 0] = u
    uv[y - r : y + r, x - r : x + r, 1] = v

    # No return value, uv is modified in place


def initialize_grid(grid_size: int, perturb: bool = True) -> np.ndarray:
    """
    Initialize the grid with a perturbation in the center.

    Parameters
    ----------
    grid_size : int
        Size of the grid.
    perturb : bool, optional
        If True, perturb the center of the grid.

    Returns
    -------
    ndarray
        3D array with shape (grid_size, grid_size, 2) containing the initial
        values of u and v.
    """
    uv = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
    uv[:, :, 0] = 1.0  # Initialize u to 1.0, v to 0.0

    if perturb:
        # Add a perturbation in the center
        grid_size = uv.shape[0]
        center = grid_size // 2
        add_perturbation(uv, (center, center), r=1)

    return uv


def animate_simulation(
    grid_size: int = 128,
    dt: int = 1,
    boundary_conditions: str = "neumann",
    anim_speed: int = 10,
):
    """
    Animate the Gray-Scott model simulation.

    Parameters
    ----------
    grid_size : int
        Size of the grid.
    speed : int, optional
        Speed of the simulation, default is 1.
    """
    uv = initialize_grid(grid_size, perturb=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(uv[:, :, 1], cmap="inferno", interpolation="bilinear")
    plt.axis("off")

    def update_frame(_):
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

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x = int(event.xdata)
        y = int(event.ydata)

        # Left click?
        if event.button == 1:
            u_new = 0.50
            v_new = 0.25
        # Right click?
        elif event.button == 3:
            u_new = 1.0
            v_new = 0.0

        add_perturbation(uv, (x, y), r=3, u=u_new, v=v_new)

    fig.canvas.mpl_connect("button_press_event", on_click)

    ani = animation.FuncAnimation(fig, update_frame, interval=1, blit=True)
    plt.show()


def main():
    """
    Main function to run the interactive Gray-Scott model simulation.

    Parameters
    ----------
    config : str, optional
        Path to the configuration file.
    key : str, optional
        Key in the configuration file.
    """
    animate_simulation()


if __name__ == "__main__":
    main()
