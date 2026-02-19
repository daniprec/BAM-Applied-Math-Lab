import numpy as np
from scipy.spatial import cKDTree


def initialize_particles_couzin(
    num_boids: int, box_size: float = 25.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize the state of the particles for the (2D) Couzin model.

    Returns
    -------
    xy : (2, N) array
        Positions in [0, box_size) x [0, box_size).
    theta : (N,) array
        Headings in [0, 2*pi).
    """
    theta = np.random.uniform(0.0, 2.0 * np.pi, num_boids)
    xy = np.random.uniform(0.0, box_size, (2, num_boids))
    return xy, theta


def couzin_equations(
    xy: np.ndarray,
    theta: np.ndarray,
    box_size: float = 25.0,
    dt: float = 0.1,
    v0: float = 3.0,  # speed_s in the paper, units / s (you can rescale)
    radius_repulsion: float = 1.0,  # rr
    radius_alignment: float = 6.0,  # ro
    radius_attraction: float = 14.0,  # ra
    noise_std: float = 0.05,  # noise_std in radians (Gaussian)
    perception_angle_deg: float = 270.0,  # alpha (field of view) from the paper
    turning_rate_deg: float = 40.0,  # max_turn_rate (max turning rate) from the paper
) -> tuple[np.ndarray, np.ndarray]:
    """
    One time step of a 2D version of Couzin et al. (2002) model.

    This follows the rules in the paper:
    - Priority of zones: repulsion (zor) > orientation (zoo) + attraction (zoa)
    - Repulsion uses the sum of unit vectors away from all neighbours in zor.
    - Orientation uses the sum of neighbours' heading unit vectors.
    - Attraction uses the sum of unit vectors towards neighbours in zoa.
    - A finite field of perception (blind rear sector).
    - Finite maximum turning rate.
    - Angular noise is Gaussian (wrapped on the circle).

    Positions are updated in a periodic square of side `box_size`.
    """

    num_boids = xy.shape[1]

    # Build KD-tree for efficient neighbour search up to the largest radius (ra)
    tree = cKDTree(xy.T)

    theta_new = theta.copy()

    half_fov = np.deg2rad(perception_angle_deg) / 2.0
    max_turn = np.deg2rad(turning_rate_deg) * dt

    for i in range(num_boids):
        pos_i = xy[:, i]
        heading_i = theta[i]

        # Query neighbours within radius_attraction (zoa outer boundary)
        neighbour_indices = tree.query_ball_point(pos_i, radius_attraction)

        # Remove self if present
        neighbour_indices = [j for j in neighbour_indices if j != i]
        if not neighbour_indices:
            # No neighbours at all: desired direction is current heading
            desired_heading = heading_i
        else:
            rel = (
                xy[:, neighbour_indices] - pos_i[:, None]
            )  # vectors from i -> neighbours
            dists = np.linalg.norm(rel, axis=0)

            # Bearing of each neighbour relative to focal
            angles_to_neigh = np.arctan2(rel[1], rel[0])
            # Smallest signed angle between current heading and neighbour direction
            angle_diff = (angles_to_neigh - heading_i + np.pi) % (2.0 * np.pi) - np.pi

            # Apply field of view: ignore neighbours outside the perception cone
            visible = np.abs(angle_diff) <= half_fov
            if not np.any(visible):
                desired_heading = heading_i
            else:
                rel = rel[:, visible]
                dists = dists[visible]
                neigh_visible = np.array(neighbour_indices, dtype=int)[visible]

                # --- Zone of repulsion (highest priority) ---
                rep_mask = dists < radius_repulsion
                if np.any(rep_mask):
                    # dr(t + Δt) = sum_j rij / |rij| with rij = (cj - ci)
                    # Desired direction is away from neighbours in zor: ci - cj
                    vec_away = -rel[:, rep_mask]  # (ci - cj) = -(cj - ci)
                    norms = np.linalg.norm(vec_away, axis=0)
                    # Avoid division by zero
                    norms[norms == 0.0] = 1.0
                    unit_away = vec_away / norms
                    direction_vec = unit_away.sum(axis=1)
                    if np.allclose(direction_vec, 0.0):
                        direction_vec = np.array([np.cos(heading_i), np.sin(heading_i)])
                else:
                    # --- No repulsion: orientation + attraction ---
                    orient_mask = (dists >= radius_repulsion) & (
                        dists < radius_alignment
                    )
                    attract_mask = (dists >= radius_alignment) & (
                        dists < radius_attraction
                    )

                    direction_vec = np.zeros(2)
                    have_o = np.any(orient_mask)
                    have_a = np.any(attract_mask)

                    if have_o:
                        # do(t + Δt) = sum_j v_j / |v_j|
                        heading_o = theta[neigh_visible[orient_mask]]
                        vj = np.vstack((np.cos(heading_o), np.sin(heading_o)))
                        norms = np.linalg.norm(vj, axis=0)
                        norms[norms == 0.0] = 1.0
                        unit_vj = vj / norms
                        do = unit_vj.sum(axis=1)
                    else:
                        do = np.zeros(2)

                    if have_a:
                        # da(t + Δt) = sum_j rij / |rij| with rij = (cj - ci)
                        rel_a = rel[:, attract_mask]
                        norms = np.linalg.norm(rel_a, axis=0)
                        norms[norms == 0.0] = 1.0
                        unit_to = rel_a / norms
                        da = unit_to.sum(axis=1)
                    else:
                        da = np.zeros(2)

                    if have_o and have_a:
                        # di = 1/2 (do + da); the scale is irrelevant for direction
                        direction_vec = do + da
                    elif have_o:
                        direction_vec = do
                    elif have_a:
                        direction_vec = da
                    else:
                        direction_vec = np.array([np.cos(heading_i), np.sin(heading_i)])

                    if np.allclose(direction_vec, 0.0):
                        direction_vec = np.array([np.cos(heading_i), np.sin(heading_i)])

                desired_heading = np.arctan2(direction_vec[1], direction_vec[0])

        # Add angular noise: Gaussian, then wrap
        desired_heading += np.random.normal(loc=0.0, scale=noise_std)
        desired_heading = desired_heading % (2.0 * np.pi)

        # Enforce maximum turning rate
        dheading = (desired_heading - heading_i + np.pi) % (2.0 * np.pi) - np.pi
        if np.abs(dheading) > max_turn:
            dheading = np.sign(dheading) * max_turn
        theta_new[i] = (heading_i + dheading) % (2.0 * np.pi)

    # Move at constant speed v0
    v = v0 * np.vstack((np.cos(theta_new), np.sin(theta_new)))
    xy_new = xy + dt * v

    # Periodic boundary conditions
    xy_new = np.mod(xy_new, box_size)

    return xy_new, theta_new
