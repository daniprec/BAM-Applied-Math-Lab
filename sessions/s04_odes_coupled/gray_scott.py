import sys
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import toml
from matplotlib import animation

# Add the path to the sys module
# (allowing the import of the utils module)
sys.path.append("./sessions")

from s02_odes_2d.solver import laplacian, laplacian_9pt, solve_ode


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

    # Enforce boundary conditions (Neumann)
    duv_dt[0, :] = 0.0
    duv_dt[-1, :] = 0.0
    duv_dt[:, 0] = 0.0
    duv_dt[:, -1] = 0.0

    return duv_dt


def animate_simulation(grid_size: int, **kwargs: Any):
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
        uv = solve_ode(gray_scott_ode, uv, **kwargs)[..., -1]
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

    ani = animation.FuncAnimation(fig, update_frame, interval=0, blit=True)
    plt.show()


def main(config: str = "config.toml", key: str = "gray-scott"):
    """
    Main function to run the interactive Gray-Scott model simulation.

    Parameters
    ----------
    config : str, optional
        Path to the configuration file.
    key : str, optional
        Key in the configuration file.
    """
    dict_config: dict = toml.load(config)[key]
    animate_simulation(**dict_config)


if __name__ == "__main__":
    main()
