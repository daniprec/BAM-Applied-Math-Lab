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
        Number of particles, N.
    box_size : float, optional
        Dimension of the space, L (default is 25).

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
    noise: float = 0.1,
    box_size: float = 25,
    dt: float = 1,
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
    noise : float, optional
        Noise parameter, eta, default is 0.1.
    box_size : float, optional
        Dimension of the space, L, default is 25.
    dt : float, optional
        Time step, default is 1 [Vicsek1995]
    radius_interaction : float, optional
        Interaction radius, default is 1 [Vicsek1995]
    v0 : float, optional
        Speed of the particles, default is 0.03 [Vicsek1995]

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
    # Vectorized computation of average direction of neighbors (including itself)
    num_boids = xy.shape[1]
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    # neighbors is (N, N), sin_theta/cos_theta is (N,)
    # Broadcasting: (N, N) * (N,) -> (N, N)
    sum_sin = neighbors @ sin_theta  # (N,)
    sum_cos = neighbors @ cos_theta  # (N,)
    count = neighbors.sum(axis=1)  # (N,)
    mean_sin = sum_sin / count
    mean_cos = sum_cos / count
    theta_avg = np.arctan2(mean_sin, mean_cos)
    # Add noise: uniform in [-noise/2, noise/2]
    noise_arr = noise * (np.random.uniform(size=num_boids) - 0.5)
    theta_new = theta_avg + noise_arr
    theta_new = np.mod(theta_new, 2 * np.pi)

    # Update position
    v = v0 * np.array([np.cos(theta_new), np.sin(theta_new)])
    xy_new = xy + dt * v
    # Periodic boundary conditions
    xy_new = np.mod(xy_new, box_size)

    return xy_new, theta_new


def vicsek_order_parameter(theta: np.ndarray, v0: float = 0.03) -> float:
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
    noise: float,
    density: float = 2.0,
    radius_interaction: float = 1.0,
    v0: float = 0.03,
    steps: int = 20000,
    dt: float = 1.0,
    avg_steps: int = 2000,
) -> float:
    """
    Simulate the Vicsek model and return the time-averaged normalized
    order parameter over the last avg_steps.

    Parameters
    ----------
    num_boids : int
        Number of particles, N.
    noise : float
        Noise parameter, eta.
    density : float, optional
        Density of the system (N/L^2), default is 2.0.
    radius_interaction : float, optional
        Interaction radius, default is 1.0.
    v0 : float, optional
        Speed of the particles, default is 0.03.
    steps : int, optional
        Total number of simulation steps, default is 20000.
    dt : float, optional
        Time step, default is 1.0.
    avg_steps : int, optional
        Number of steps to average over for the order parameter, default is 2000.

    Returns
    -------
    float
        Time-averaged normalized order parameter over the last avg_steps.
    """
    box_size = np.sqrt(num_boids / density)
    xy, theta = initialize_particles(num_boids, box_size=box_size)
    order_params = []
    for i in range(steps):
        xy, theta = vicsek_equations(
            xy,
            theta,
            dt=dt,
            noise=noise,
            box_size=box_size,
            radius_interaction=radius_interaction,
            v0=v0,
        )
        if i >= steps - avg_steps:
            order_params.append(vicsek_order_parameter(theta, v0=v0))
    avg_order = float(np.mean(order_params))
    return avg_order


def plot_avg_velocity_vs_noise(
    num_boids_list: list[int],
    density: float = 4.0,
    noise_range: tuple[float, float] = (0.0, 5.0),
    noise_steps: int = 10,
    n_realizations: int = 1,
) -> None:
    """
    Plot the time-averaged normalized order parameter vs noise for different system sizes (N),
    keeping density fixed.
    Optionally averages over multiple realizations for each parameter set.

    Parameters
    ----------
    num_boids_list : list[int]
        List of system sizes (number of particles, N) to simulate.
    density : float, optional
        Density of the system (N/L^2), default is 4.0.
    noise_range : tuple[float, float], optional
        Range of noise values (eta) to simulate, default is (0.0, 5.0).
    noise_steps : int, optional
        Number of noise values to simulate within the range, default is 20.
    n_realizations : int, optional
        Number of independent realizations to average over for each parameter set, default is 1.
    """
    plt.figure(figsize=(8, 6))
    noises = np.linspace(noise_range[0], noise_range[1], noise_steps)
    # Following the marker style from the original paper
    # Plus sign in scatter is
    markers = ["s", "P", "X", "^", "D"]
    for idx, nb in enumerate(num_boids_list):
        box_size = np.sqrt(nb / density)
        avg_orders = []
        print(f"Simulating for N={nb}, L={box_size:.2f}, density={density}")
        for noise in noises:
            vals = []
            for rep in range(n_realizations):
                val = simulate_vicsek(nb, noise, density=density)
                vals.append(val)
            mean_val = float(np.mean(vals))
            avg_orders.append(mean_val)
            print(f"    noise={noise:.3f}, phi={mean_val:.4f}")
        marker = markers[idx % len(markers)]
        plt.scatter(
            noises, avg_orders, label=f"N={nb}, L={box_size:.1f}", marker=marker, s=60
        )
    plt.xlabel(r"Noise ($\eta$)")
    plt.ylabel(r"Order parameter $\varphi$")
    plt.title(f"Order parameter vs Noise (density={density})")
    plt.legend()
    plt.grid(True)


def plot_avg_velocity_vs_density(
    density_range: tuple[float, float] = (0.1, 10.0),
    density_steps: int = 20,
    noise: float = 2.0,
    box_size: float = 5,
) -> None:
    """
    Plot the time-averaged normalized order parameter vs density for a fixed noise.

    Parameters
    ----------
    density_range : tuple[float, float], optional
        Range of density values (N/L^2) to simulate, default is (0.1, 10.0).
    density_steps : int, optional
        Number of density values to simulate within the range, default is 20.
    noise : float, optional
        Noise parameter (eta) to use for the simulations, default is 2.0.
    box_size : float, optional
        Dimension of the space, L, default is 5.
        Vicsek used 20 but we use smaller L to speed up the simulations at high densities.
    """
    avg_velocities = []
    # Because computation takes way longer in high densities,
    # we use a logarithmic spacing for density values
    densities = np.logspace(
        np.log10(density_range[0]), np.log10(density_range[1]), density_steps
    )

    plt.figure(figsize=(8, 6))
    # Simulate for each density and compute average velocity
    for density in densities:
        num_boids = int(density * box_size**2)
        # If the number of boids is too small, skip to avoid noisy results
        if num_boids <= 5:
            print(
                f"Box size: {box_size}, density: {density:.2f}, num_boids: {num_boids} (skipped)"
            )
            continue
        avg_v = simulate_vicsek(num_boids=num_boids, noise=noise, density=density)
        avg_velocities.append(avg_v)
        print(
            f"Box size: {box_size}, density: {density:.2f}, avg velocity: {avg_v:.4f}"
        )
    # Crop densities to match the length of avg_velocities (in case some were skipped)
    densities = densities[-len(avg_velocities) :]
    plt.scatter(densities, avg_velocities, marker="o")
    plt.xlabel("Density (N/L^2)")
    plt.ylabel(r"Order parameter, avg. velocity ($\varphi$)")
    plt.title("Average velocity vs Density (" r"$\eta$" f"={noise}, L={box_size})")
    plt.grid(True)


def main(img_dir: str = "img") -> None:
    # Example system sizes (N) as in the original paper
    num_boids_list = [40, 100]
    # For publication-quality, set n_realizations > 1 (e.g. 5-10)
    plot_avg_velocity_vs_noise(num_boids_list=num_boids_list, n_realizations=1)
    os.makedirs(img_dir, exist_ok=True)
    out_path = os.path.join(img_dir, "s04_order_parameter_vs_noise.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    plot_avg_velocity_vs_density()
    os.makedirs(img_dir, exist_ok=True)
    out_path = os.path.join(img_dir, "s04_avg_velocity_vs_density.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
