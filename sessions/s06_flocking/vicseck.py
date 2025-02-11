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
    state0 = np.zeros(shape=(n * 3,))
    state0[0::3] = np.random.rand(n) * 2 * np.pi  # Random initial angles
    state0[1::3] = np.random.rand(n) * d  # Random initial x positions
    state0[2::3] = np.random.rand(n) * d  # Random initial y positions
    return state0


def update_rule(
    present: np.ndarray,
    stepsize: float = 1,
    eta: float = 0.1,
    d: float = 25,
    r: float = 1,
    v0: float = 0.03,
) -> np.ndarray:
    """
    Update the state of the cells based on the Vicsek model.

    Parameters
    ----------
    present : np.ndarray
        Current state vector of the cells.
    stepsize : float, optional
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
    future = present.copy()
    n = np.size(present) // 3
    neighbor_mean_angles = np.zeros(n)

    positions = np.zeros((n, 2))
    positions[:, 0] = present[1::3]
    positions[:, 1] = present[2::3]
    angles = present[0::3]

    # Compute distance matrix and neighbor matrix
    d_matrix = scipy.spatial.distance.pdist(positions)
    d_matrix = scipy.spatial.distance.squareform(d_matrix)
    neighbors = d_matrix <= r

    for i in range(n):
        my_theta = angles[i]
        my_pos = positions[i, :]

        # Compute mean angle of neighbors
        neighbor_mean_angles[i] = np.sum(angles[neighbors[:, i]]) / np.sum(
            neighbors[:, i]
        )

        # Update position
        v = v0 * np.array([np.cos(my_theta), np.sin(my_theta)])
        future[np.array([1, 2]) + 3 * i] = (
            present[np.array([1, 2]) + 3 * i] + stepsize * v
        )
        future[np.array([1, 2]) + 3 * i] = np.mod(future[np.array([1, 2]) + 3 * i], d)

    # Add noise to the angles
    noise = -eta / 2 + np.random.rand(n) * eta / 2
    future[0::3] = np.mod(present[0::3] + neighbor_mean_angles + noise, 2 * np.pi)

    return future


def plot_cells(state_vector: np.ndarray, d: float = 25):
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
    n = np.size(state_vector) // 3
    for i in range(n):
        my_theta, my_x, my_y = state_vector[np.array([0, 1, 2]) + 3 * i]
        ax.arrow(
            my_x,
            my_y,
            5 * ar_scale * np.cos(my_theta),
            5 * ar_scale * np.sin(my_theta),
            head_width=2 * ar_scale,
            head_length=ar_scale,
            fc="k",
            ec="k",
        )
    plt.draw()


def update_animation(i: int, trajectory: np.ndarray, d: float):
    """
    Animation function for updating the plot.

    Parameters
    ----------
    i : int
        Frame index.
    trajectory : np.ndarray
        Trajectory of the cells.
    d : float
        Dimension of the space.
    """
    plot_cells(trajectory[:, i], d=d)


def run_animation(
    n: int = 30,
    t: int = 1000,
    r: float = 1,
    d: float = 25,
    eta: float = 0.1,
    stepsize: float = 1,
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
    stepsize : float, optional
        Step size for the update (default is 1).
    """
    state0 = initialize_cells(n, d=d)
    trajectory = np.zeros(shape=(n * 3, t))
    trajectory[:, 0] = state0

    for i in range(t - 1):
        trajectory[:, i + 1] = update_rule(
            trajectory[:, i], stepsize=stepsize, eta=eta, r=r, d=d
        )

    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(
        fig, update_animation, frames=t, fargs=(trajectory, d), interval=50
    )
    plt.show()


if __name__ == "__main__":
    run_animation()
