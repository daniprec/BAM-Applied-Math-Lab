import os

import matplotlib.pyplot as plt
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


def compute_order_parameter(
    xy: np.ndarray, theta: np.ndarray, v0: float = 0.03
) -> float:
    """
    Compute the normalized order parameter (mean velocity divided by v0), as in Vicsek et al. (1995).
    """
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    avg_vx = np.mean(vx)
    avg_vy = np.mean(vy)
    return float(np.sqrt(avg_vx**2 + avg_vy**2) / v0)


def simulate_vicsek(
    num_boids: int,
    eta: float,
    density: float = 2.0,
    radius_interaction: float = 1.0,
    v0: float = 0.03,
    steps: int = 20000,
    dt: float = 1.0,
    avg_steps: int = 2000,
) -> float:
    """
    Simulate the Vicsek model and return the time-averaged normalized order parameter over the last avg_steps.
    """
    box_size = np.sqrt(num_boids / density)
    xy, theta = initialize_particles(num_boids, box_size=box_size)
    order_params = []
    for i in range(steps):
        xy, theta = vicsek_equations(
            xy,
            theta,
            dt=dt,
            eta=eta,
            box_size=box_size,
            radius_interaction=radius_interaction,
            v0=v0,
        )
        if i >= steps - avg_steps:
            order_params.append(compute_order_parameter(xy, theta, v0=v0))
    avg_order = float(np.mean(order_params))
    return avg_order


def plot_avg_velocity_vs_noise(
    num_boids_list: list[int],
    density: float = 4.0,
    eta_range: tuple[float, float] = (0.0, 5.0),
    eta_steps: int = 20,
    radius_interaction: float = 1.0,
    v0: float = 0.03,
    steps: int = 20000,
    dt: float = 1.0,
    avg_steps: int = 2000,
    n_realizations: int = 1,
    img_dir: str = "img",
) -> None:
    """
    Plot the time-averaged normalized order parameter vs noise for different system sizes (N), keeping density fixed.
    Optionally averages over multiple realizations for each parameter set.
    """
    plt.figure(figsize=(8, 6))
    etas = np.linspace(eta_range[0], eta_range[1], eta_steps)
    for nb in num_boids_list:
        box_size = np.sqrt(nb / density)
        avg_orders = []
        print(f"Simulating for N={nb}, L={box_size:.2f}, density={density}")
        for eta in etas:
            vals = []
            for rep in range(n_realizations):
                val = simulate_vicsek(
                    nb, eta, density, radius_interaction, v0, steps, dt, avg_steps
                )
                vals.append(val)
            mean_val = float(np.mean(vals))
            avg_orders.append(mean_val)
            print(f"  eta={eta:.3f}, " r"\varphi" f"={mean_val:.4f}")
        plt.plot(etas, avg_orders, label=f"N={nb}, L={box_size:.1f}")
    plt.xlabel(r"Noise (\eta)")
    plt.ylabel(r"Order parameter \varphi")
    plt.title(f"Order parameter vs Noise (density={density})")
    plt.legend()
    plt.grid(True)
    os.makedirs(img_dir, exist_ok=True)
    out_path = os.path.join(img_dir, "order_parameter_vs_noise.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_avg_velocity_vs_density(
    fixed_eta: float = 0.1,
    densities: np.ndarray = np.linspace(0.1, 2.0, 20),
    box_size: float = 20,
    radius_interaction: float = 1,
    v0: float = 0.03,
    steps: int = 1000,
    dt: float = 1,
    img_dir: str = "img",
) -> None:
    avg_velocities = []
    for density in densities:
        num_boids = int(density * box_size**2)
        avg_v = simulate_vicsek(
            num_boids, box_size, fixed_eta, radius_interaction, v0, steps, dt
        )
        avg_velocities.append(avg_v)
        print(
            f"Box size: {box_size}, density: {density:.2f}, avg velocity: {avg_v:.4f}"
        )
    plt.figure(figsize=(8, 6))
    plt.plot(densities, avg_velocities, marker="o")
    plt.xlabel("Density (N/L^2)")
    plt.ylabel("|<v>| (average velocity)")
    plt.title(f"Average velocity vs Density (eta={fixed_eta})")
    plt.grid(True)
    os.makedirs(img_dir, exist_ok=True)
    out_path = os.path.join(img_dir, "avg_velocity_vs_density.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    # Example system sizes (N) as in the original paper, e.g. N = 40, 100
    num_boids_list = [40, 100]
    # For publication-quality, set n_realizations > 1 (e.g. 5-10)
    plot_avg_velocity_vs_noise(num_boids_list=num_boids_list, n_realizations=1)
    plot_avg_velocity_vs_density()


if __name__ == "__main__":
    main()
