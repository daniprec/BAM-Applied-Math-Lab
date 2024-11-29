from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import toml
from matplotlib import animation


def initialize_grid(
    grid_size: int, perturb: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initializes the concentration grids for u and v.

    Parameters
    ----------
    grid_size : int
        Length of the grid border (square grid).
    perturb : bool, optional
        Whether to add an initial perturbation.

    Returns
    -------
    u : ndarray
        Concentration grid for u.
    v : ndarray
        Concentration grid for v.
    """
    u = np.ones((grid_size, grid_size), dtype=np.float16)
    v = np.zeros((grid_size, grid_size), dtype=np.float16)

    # Add a small perturbation in the center
    if perturb:
        r = 1  # Radius of perturbation
        center = grid_size // 2
        u[center - r : center + r, center - r : center + r] = 0.50
        v[center - r : center + r, center - r : center + r] = 0.25

    return u, v


def laplacian(z: np.ndarray) -> np.ndarray:
    """
    Computes the Laplacian of a grid using the finite difference method.

    Parameters
    ----------
    z : ndarray
        Input grid.

    Returns
    -------
    lap : ndarray
        Laplacian of the input grid.
    """
    return (
        -4 * z
        + np.roll(z, (0, -1), (0, 1))
        + np.roll(z, (0, 1), (0, 1))
        + np.roll(z, (-1, 0), (0, 1))
        + np.roll(z, (1, 0), (0, 1))
    )


def update(
    u: np.ndarray,
    v: np.ndarray,
    du: float = 0.16,
    dv: float = 0.08,
    f: float = 0.060,
    k: float = 0.062,
    dt: float = 1.0,
) -> None:
    """
    Updates the concentration grids for u and v.

    Parameters
    ----------
    u : ndarray
        Current concentration grid for u.
    v : ndarray
        Current concentration grid for v.
    du : float, optional
        Diffusion rate for u.
    dv : float, optional
        Diffusion rate for v.
    f : float, optional
        Feed rate for u.
    k : float, optional
        Kill rate for v.
    dt : float, optional
        Time step for the simulation.

    Returns
    -------
    None
    """
    lu = laplacian(u)
    lv = laplacian(v)

    uvv = u * v * v
    du_dt = du * lu - uvv + f * (1 - u)
    dv_dt = dv * lv + uvv - (f + k) * v

    u_new = u + du_dt * dt
    v_new = v + dv_dt * dt

    # Enforce boundary conditions (Neumann)
    u_new[0, :] = u_new[1, :]
    u_new[-1, :] = u_new[-2, :]
    u_new[:, 0] = u_new[:, 1]
    u_new[:, -1] = u_new[:, -2]

    v_new[0, :] = v_new[1, :]
    v_new[-1, :] = v_new[-2, :]
    v_new[:, 0] = v_new[:, 1]
    v_new[:, -1] = v_new[:, -2]

    return u_new, v_new


def animate_simulation(grid_size: int, **kwargs) -> None:
    """
    Animates the simulation results in real time with an on-click event.

    Parameters
    ----------
    grid_size : int
        Length of the grid border (square grid).
    **kwargs
        Additional keyword arguments for the simulation parameters.

    Returns
    -------
    None
    """
    u, v = initialize_grid(grid_size, perturb=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(v, cmap="inferno", interpolation="bilinear")
    plt.axis("off")

    def update_frame(_):
        nonlocal u, v
        for _ in range(100):
            u, v = update(u, v, **kwargs)
        im.set_array(v)
        return [im]

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x = int(event.xdata)
        y = int(event.ydata)
        r = 3  # Radius of the perturbation

        # Left click?
        if event.button == 1:
            u_new = 0.50
            v_new = 0.25
        # Right click?
        elif event.button == 3:
            u_new = 0.0
            v_new = 0.0

        # Define the slice ranges, ensuring they are within the grid bounds
        y_min = max(y - r, 0)
        y_max = min(y + r, grid_size)
        x_min = max(x - r, 0)
        x_max = min(x + r, grid_size)

        # Add a perturbation at the clicked location
        nonlocal u, v
        u[y_min:y_max, x_min:x_max] = u_new
        v[y_min:y_max, x_min:x_max] = v_new

    # Connect the on_click event handler to the figure
    fig.canvas.mpl_connect("button_press_event", on_click)

    ani = animation.FuncAnimation(
        fig, update_frame, interval=100, save_count=0, blit=True
    )
    plt.show()


def main(config: str = "config.toml", key: str = "gray-scott"):
    """
    Main function to run the Gray-Scott simulation.

    Parameters
    ----------
    config : str, optional
        Path to the configuration file.
    key : str, optional
        Key in the configuration file to load parameters from.

    Returns
    -------
    None
    """
    # Read parameters from config file
    dict_config: dict = toml.load(config)[key]

    animate_simulation(**dict_config)


if __name__ == "__main__":
    main()
