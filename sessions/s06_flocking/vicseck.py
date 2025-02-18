import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial


def initialize_cells(n: int, d: float = 25) -> np.ndarray:
    """
    Initialize the state of the cells.

    Parameters
    ----------
    n : int
        Number of cells.
    d : float, optional
        Dimension of the space (default is 25).

    Returns
    -------
    np.ndarray
        Initial state vector of the cells.
    """
    # Random initial angles
    angles = np.random.uniform(0, 2 * np.pi, n)
    # Random initial x, y xy
    xy = np.random.uniform(0, d, (2, n))
    return xy, angles


def update_rule(
    xy: np.ndarray,
    angles: np.ndarray,
    dt: float = 1,
    eta: float = 0.1,
    d: float = 25,
    r: float = 1,
    v0: float = 2,
) -> np.ndarray:
    """
    Update the state of the cells based on the Vicsek model.

    Parameters
    ----------
    present : np.ndarray
        Current state vector of the cells.
    dt : float, optional
        Step size for the update (default is 1).
    eta : float, optional
        Noise parameter (default is 0.1).
    d : float, optional
        Dimension of the space (default is 25).
    r : float, optional
        Interaction radius (default is 1).
    v0 : float, optional
        Speed of the cells (default is 0.03).

    Returns
    -------
    np.ndarray
        Updated state vector of the cells.
    """
    # Compute distance matrix and neighbor matrix
    d_matrix = scipy.spatial.distance.pdist(xy.T)
    d_matrix = scipy.spatial.distance.squareform(d_matrix)
    neighbors = d_matrix <= r
    neighbor_mean_angles = np.zeros_like(angles)

    for i in range(len(angles)):
        # Compute mean angle of neighbors
        neighbor_mean_angles[i] = np.mean(angles[neighbors[:, i]])
    dtheta = neighbor_mean_angles - angles
    angles = np.mod(angles + dtheta * dt, 2 * np.pi)

    # Update position
    v = v0 * np.array([np.cos(angles), np.sin(angles)])
    xy = xy + dt * v
    xy = np.mod(xy, d)

    return xy, angles


def plot_cells(xy: np.ndarray, angles: np.ndarray, d: float = 25):
    """
    Plot the cells on a 2D plane.

    Parameters
    ----------
    state_vector : np.ndarray
        State vector of the cells.
    d : float, optional
        Dimension of the space (default is 25).
    """
    ax = plt.gca()
    ax.cla()
    ax.set_xlim(0, d)
    ax.set_ylim(0, d)
    ar_scale = d / 250
    for i in range(len(angles)):
        ax.arrow(
            xy[0, i],
            xy[1, i],
            5 * ar_scale * np.cos(angles[i]),
            5 * ar_scale * np.sin(angles[i]),
            head_width=2 * ar_scale,
            head_length=ar_scale,
            fc="k",
            ec="k",
        )
    plt.draw()


def run_animation(
    n: int = 30,
    dt: float = 0.01,
    r: float = 1,
    d: float = 25,
    eta: float = 0.1,
):
    """
    Run the animation of the Vicsek model.
    Reference: Topaz, Chad M., Lori Ziegelmeier, and Tom Halverson. 2015.
    "Topological Data Analysis of Biological Aggregation Models."
    PLOS ONE 10 (5): e0126383. https://doi.org/10/f7mp7k

    Parameters
    ----------
    n : int, optional
        Number of cells (default is 30).
    t : int, optional
        Number of time steps (default is 1000).
    r : float, optional
        Interaction radius (default is 1).
    d : float, optional
        Dimension of the space (default is 25).
    eta : float, optional
        Noise parameter (default is 0.1).
    dt : float, optional
        Step size for the update (default is 1).
    """
    xy, positions = initialize_cells(n, d=d)

    def update_animation(frame: int):
        nonlocal xy, positions
        for _ in range(10):
            xy, positions = update_rule(xy, positions, dt=dt, r=r, d=d, eta=eta)
        plot_cells(xy, positions, d=d)

    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, update_animation, interval=1)
    plt.show()


if __name__ == "__main__":
    run_animation()
