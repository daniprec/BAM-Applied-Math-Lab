import numpy as np
import scipy.spatial


def initialize_particles(
    num_boids: int, box_size: float = 25
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize the state of the particles.

    Parameters
    ----------
    num_boids : int
        Number of particles.
    box_size : float, optional
        Dimension of the space (default is 25).

    Returns
    -------
    np.ndarray
        Initial positions of the particles.
    np.ndarray
        Initial angle of the particles, in radians.
    """
    # Random initial theta
    theta = np.random.uniform(0, 2 * np.pi, num_boids)
    # Random initial x, y
    xy = np.random.uniform(0, box_size, (2, num_boids))
    return xy, theta


def vicsek_equations(
    xy: np.ndarray,
    theta: np.ndarray,
    dt: float = 1,
    eta: float = 0.1,
    box_size: float = 25,
    radius_interaction: float = 1,
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
    box_size : float, optional
        Dimension of the space, default is 25.
    radius_interaction : float, optional
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
    neighbors = d_matrix <= radius_interaction
    # Compute mean angle of neighbors
    term_theta_avg = theta @ neighbors / np.sum(neighbors, axis=1)
    # Add noise
    term_noise = eta * np.pi * np.random.uniform(-1, 1, len(theta))
    # Update angle
    theta = term_theta_avg + term_noise
    theta = np.mod(theta, 2 * np.pi)

    # Update position
    v = v0 * np.array([np.cos(theta), np.sin(theta)])
    xy = xy + dt * v
    # Boundary conditions: periodic
    xy = np.mod(xy, box_size)

    return xy, theta


def vicsek_order_parameter(theta: np.ndarray) -> float:
    """
    Compute the order parameter of the Vicsek model.

    Parameters
    ----------
    theta : np.ndarray
        Angle of the particles.

    Returns
    -------
    float
        Order parameter of the Vicsek model.
    """
    return np.abs(np.mean(np.exp(1j * theta)))
