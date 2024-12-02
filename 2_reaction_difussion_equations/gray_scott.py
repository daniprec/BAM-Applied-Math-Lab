from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import toml
from matplotlib import animation


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
        grid_size = uv.shape[0]
        center = grid_size // 2
        add_perturbation(uv, (center, center), r=1)

    return uv


def laplacian(uv: np.ndarray) -> np.ndarray:
    """
    Compute the Laplacian of the u and v fields using a 5-point finite
    difference scheme: considering each point and its immediate neighbors in
    the up, down, left, and right directions.

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
    lap += np.roll(uv, shift=1, axis=0)
    lap += np.roll(uv, shift=-1, axis=0)
    lap += np.roll(uv, shift=1, axis=1)
    lap += np.roll(uv, shift=-1, axis=1)
    return lap


def update(
    uv: np.ndarray,
    du: float = 0.16,
    dv: float = 0.08,
    f: float = 0.060,
    k: float = 0.062,
    dt: float = 1.0,
):
    """
    Update the u and v fields using the Gray-Scott model with explicit Euler
    time integration, where the fields are updated based on their current values
    and the calculated derivatives.

    Parameters
    ----------
    uv : np.ndarray
        3D array with shape (grid_size, grid_size, 2) containing the values of
        u and v.
    du : float, optional
        Du parameter, default is 0.16.
    dv : float, optional
        Dv parameter, default is 0.08.
    f : float, optional
        F parameter, default is 0.060.
    k : float, optional
        K parameter, default is 0.062.
    dt : float, optional
        Time step, default is 1.0.
    """
    u = uv[:, :, 0]
    v = uv[:, :, 1]

    lap = laplacian(uv)
    lu = lap[:, :, 0]
    lv = lap[:, :, 1]

    uvv = u * v * v
    du_dt = du * lu - uvv + f * (1 - u)
    dv_dt = dv * lv + uvv - (f + k) * v

    uv[:, :, 0] += du_dt * dt
    uv[:, :, 1] += dv_dt * dt

    # Enforce boundary conditions (Neumann)
    uv[0, :] = uv[1, :]
    uv[-1, :] = uv[-2, :]
    uv[:, 0] = uv[:, 1]
    uv[:, -1] = uv[:, -2]


def animate_simulation(grid_size: int, **kwargs: Any):
    """
    Animate the Gray-Scott model simulation.

    Parameters
    ----------
    grid_size : int
        Size of the grid.
    """
    uv = initialize_grid(grid_size, perturb=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(uv[:, :, 1], cmap="inferno", interpolation="bilinear")
    plt.axis("off")

    def update_frame(_):
        for _ in range(50):
            update(uv, **kwargs)
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

    ani = animation.FuncAnimation(fig, update_frame, interval=100, blit=True)
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
