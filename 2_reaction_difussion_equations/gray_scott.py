from typing import Any, List, Tuple

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
    u = np.ones((grid_size, grid_size))
    v = np.zeros((grid_size, grid_size))

    # Add a small perturbation in the center
    if perturb:
        r = 20  # Radius of perturbation
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
) -> Tuple[np.ndarray, np.ndarray]:
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
    u_new : ndarray
        Updated concentration grid for u.
    v_new : ndarray
        Updated concentration grid for v.
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


def run_simulation(grid_size: int, steps: int, **kwargs: Any) -> List[np.ndarray]:
    """
    Runs the Gray-Scott simulation.

    Parameters
    ----------
    grid_size : int
        Length of the grid border (square grid).
    steps : int
        Number of time steps to simulate.
    **kwargs : Any
        Additional keyword arguments for the simulation.

    Returns
    -------
    frames : list of ndarray
        List of concentration grids for visualization.
    """
    u, v = initialize_grid(grid_size)

    frames = []
    for i in range(steps):
        u, v = update(u, v, **kwargs)
        if i % 10 == 0:
            frames.append(v.copy())
    return frames


def animate_simulation(frames: List[np.ndarray]) -> None:
    """
    Animates the simulation results.

    Parameters
    ----------
    frames : list of ndarray
        List of concentration grids for visualization.

    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(6, 6))
    im = plt.imshow(frames[0], cmap="inferno", interpolation="bilinear")

    def update_frame(i):
        im.set_array(frames[i])
        return [im]

    ani = animation.FuncAnimation(
        fig, update_frame, frames=len(frames), interval=10, blit=True
    )
    plt.axis("off")
    plt.show()


def main(config: str = "config.toml", key: str = "gray-scott"):
    """
    Main function to run the Gray-Scott simulation.
    """
    # Read parameters from config file
    dict_config: dict = toml.load(config)[key]

    frames = run_simulation(**dict_config)
    animate_simulation(frames)


if __name__ == "__main__":
    main()
