from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import toml
from matplotlib import animation


def initialize_grid(n: int, perturb: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initializes the concentration grids for u and v.

    Parameters
    ----------
    n : int
        Size of the grid (n x n).
    perturb : bool, optional
        Whether to add an initial perturbation.

    Returns
    -------
    u : ndarray
        Concentration grid for u.
    v : ndarray
        Concentration grid for v.
    """
    u = np.ones((n, n))
    v = np.zeros((n, n))

    # Add a small perturbation in the center
    if perturb:
        r = 20  # Radius of perturbation
        center = n // 2
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


def update(u: np.ndarray, v: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Updates the concentration grids for u and v.

    Parameters
    ----------
    u : ndarray
        Current concentration grid for u.
    v : ndarray
        Current concentration grid for v.
    params : dict
        Dictionary containing model parameters.

    Returns
    -------
    u_new : ndarray
        Updated concentration grid for u.
    v_new : ndarray
        Updated concentration grid for v.
    """
    du, dv, f, k, dt = (
        params["du"],
        params["dv"],
        params["f"],
        params["k"],
        params["dt"],
    )

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


def run_simulation(n: int, steps: int, params: dict) -> List[np.ndarray]:
    """
    Runs the Gray-Scott simulation.

    Parameters
    ----------
    n : int
        Size of the grid (n x n).
    steps : int
        Number of time steps to simulate.
    params : dict
        Dictionary containing model parameters.

    Returns
    -------
    frames : list of ndarray
        List of concentration grids for visualization.
    """
    u, v = initialize_grid(n)

    frames = []
    for i in range(steps):
        u, v = update(u, v, params)
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
        fig, update_frame, frames=len(frames), interval=50, blit=True
    )
    plt.axis("off")
    plt.show()


def main(config: str = "config.toml"):
    """
    Main function to run the Gray-Scott simulation.
    """
    # Read parameters from config file
    dict_config = toml.load(config)["gray-scott"]

    n = dict_config.get("grid_size", 256)
    steps = dict_config.get("steps", 10000)
    params = dict_config.get(
        "params",
        {
            "du": 0.16,
            "dv": 0.08,
            "f": 0.060,
            "k": 0.062,
            "dt": 1.0,
        },
    )

    frames = run_simulation(n, steps, params)
    animate_simulation(frames)


if __name__ == "__main__":
    main()
