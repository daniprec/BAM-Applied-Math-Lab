import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial


def initialize_particles(n: int, d: float = 25) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize the state of the particles.

    Parameters
    ----------
    n : int
        Number of particles.
    d : float, optional
        Dimension of the space (default is 25).

    Returns
    -------
    np.ndarray
        Initial positions of the particles.
    np.ndarray
        Initial angle of the particles, in radians.
    """
    # Random initial theta
    theta = np.random.uniform(0, 2 * np.pi, n)
    # Random initial x, y xy
    xy = np.random.uniform(0, d, (2, n))
    return xy, theta


def update_rule(
    xy: np.ndarray,
    theta: np.ndarray,
    dt: float = 1,
    eta: float = 0.1,
    d: float = 25,
    r: float = 1,
    v0: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Update the state of the particles based on the Vicsek model.

    Parameters
    ----------
    xy : np.ndarray
        Position of the particles.
    theta : np.ndarray
        Angle of the particles.
    dt : float, optional
        Time step, default is 1 (standard convention).
    eta : float, optional
        Noise parameter, default is 0.1.
    d : float, optional
        Dimension of the space, default is 25.
    r : float, optional
        Interaction radius, default is 1 (standard convention).
    v0 : float, optional
        Speed of the particles, default is 0.03.

    Returns
    -------
    np.ndarray
        Updated position of the particles.
    np.ndarray
        Updated angle of the particles.
    """
    # Compute distance matrix and neighbor matrix
    d_matrix = scipy.spatial.distance.pdist(xy.T)
    d_matrix = scipy.spatial.distance.squareform(d_matrix)
    neighbors = d_matrix <= r
    # Compute mean angle of neighbors
    neighbor_mean_theta = theta @ neighbors / np.sum(neighbors, axis=1)
    # Update angle
    theta = neighbor_mean_theta + np.random.uniform(-eta / 2, eta / 2, len(theta))

    # Update position
    v = v0 * np.array([np.cos(theta), np.sin(theta)])
    xy = xy + dt * v
    xy = np.mod(xy, d)

    return xy, theta


def plot_particles(xy: np.ndarray, theta: np.ndarray, d: float = 25):
    """
    Plot the particles in the 2D space.

    Parameters
    ----------
    xy : np.ndarray
        Position of the particles.
    theta : np.ndarray
        Angle of the particles.
    d : float, optional
        Dimension of the space, default is 25.
    """
    ax = plt.gca()
    ax.cla()
    ax.set_xlim(0, d)
    ax.set_ylim(0, d)

    # Plot using a quiver plot
    ax.quiver(
        xy[0],
        xy[1],
        np.cos(theta),
        np.sin(theta),
        angles="xy",
        scale_units="xy",
        scale=d / 20,
    )
    plt.draw()


def run_animation(
    n: int = 300,
    v0: float = 0.03,
    dt: float = 1,
    r: float = 1,
    d: float = 25,
    eta: float = 0.1,
    niter: int = 3,
):
    """
    Run the animation of the Vicsek model.
    Reference: Topaz, Chad M., Lori Ziegelmeier, and Tom Halverson. 2015.
    "Topological Data Analysis of Biological Aggregation Models."
    PLOS ONE 10 (5): e0126383. https://doi.org/10/f7mp7k

    Parameters
    ----------
    n : int, optional
        Number of particles, default is 300.
    v0 : float, optional
        Speed of the particles, default is 0.03.
    dt : float, optional
        Time step, default is 1 (standard convention).
    r : float, optional
        Interaction radius, default is 1 (standard convention).
    d : float, optional
        Dimension of the space, default is 25.
    eta : float, optional
        Noise parameter, default is 0.1.
    niter : int, optional
        Number of iterations per frame, default is 3.
    """
    xy, positions = initialize_particles(n, d=d)

    def update_animation(frame: int):
        nonlocal xy, positions
        for _ in range(niter):
            xy, positions = update_rule(xy, positions, v0=v0, dt=dt, r=r, d=d, eta=eta)
        plot_particles(xy, positions, d=d)

    fig, ax = plt.subplots()
    ani = animation.FuncAnimation(fig, update_animation, interval=0)
    plt.show()


if __name__ == "__main__":
    run_animation()
