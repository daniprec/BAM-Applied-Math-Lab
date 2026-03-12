import matplotlib.pyplot as plt
import numpy as np


def initialize_road(
    length: int = 100,
    density: float = 0.2,
    vmax: int = 5,
    seed: int | None = 1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize a 1D road with cars and their speeds.

    Parameters
    ----------
    length : int
            Number of cells in the road.
    density : float
            Fraction of cells occupied by cars.
    vmax : int
            Maximum speed (cells per step).
    seed : int | None
            Random seed for reproducibility.
    rng : np.random.Generator | None
            Optional random number generator. If provided, it is used instead of
            creating a new generator from ``seed``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
            Occupancy array (bool) and speed array (int) per cell.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    occupied = rng.random(length) < density
    speeds = np.zeros(length, dtype=int)
    speeds[occupied] = rng.integers(0, vmax + 1, size=occupied.sum())
    return occupied, speeds


def step(
    occupied: np.ndarray,
    speeds: np.ndarray,
    vmax: int = 5,
    p_slow: float = 0.3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance the traffic model by one step (Nagel-Schreckenberg).

    Parameters
    ----------
    occupied : np.ndarray
        Boolean occupancy array for the current road state.
    speeds : np.ndarray
        Integer speed array aligned with ``occupied``.
    vmax : int
        Maximum speed (cells per step).
    p_slow : float
        Probability of a random slowdown event.
    rng : np.random.Generator | None
        Optional random number generator used for the stochastic slowdown.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated occupancy array and updated speed array.
    """
    length = len(occupied)
    positions = np.where(occupied)[0]
    if len(positions) == 0:
        return occupied, speeds

    if rng is None:
        rng = np.random.default_rng()

    # 1) Accelerate
    speeds_new = speeds.copy()
    speeds_new[positions] = np.minimum(speeds_new[positions] + 1, vmax)

    # 2) Brake to avoid collisions (periodic road)
    gaps = np.zeros_like(positions)
    for idx, pos in enumerate(positions):
        next_pos = positions[(idx + 1) % len(positions)]
        gaps[idx] = (next_pos - pos - 1) % length
    speeds_new[positions] = np.minimum(speeds_new[positions], gaps)

    # 3) Random slowdown
    slow_mask = rng.random(len(positions)) < p_slow
    speeds_new[positions[slow_mask]] = np.maximum(
        speeds_new[positions[slow_mask]] - 1, 0
    )

    # 4) Move cars
    new_occupied = np.zeros_like(occupied)
    new_speeds = np.zeros_like(speeds)
    new_positions = (positions + speeds_new[positions]) % length
    new_occupied[new_positions] = True
    new_speeds[new_positions] = speeds_new[positions]

    return new_occupied, new_speeds


def simulate(
    steps: int = 200,
    length: int = 100,
    density: float = 0.2,
    vmax: int = 5,
    p_slow: float = 0.3,
    seed: int | None = 1,
) -> np.ndarray:
    """Run the traffic model and return a space-time occupancy grid.

    Parameters
    ----------
    steps : int
        Number of time steps to simulate.
    length : int
        Number of cells in the periodic road.
    density : float
        Initial fraction of occupied cells.
    vmax : int
        Maximum speed (cells per step).
    p_slow : float
        Probability of a random slowdown event.
    seed : int | None
        Random seed used to initialize the shared generator.

    Returns
    -------
    np.ndarray
        Space-time occupancy grid with shape ``(steps, length)``.
    """
    rng = np.random.default_rng(seed)
    occupied, speeds = initialize_road(length, density, vmax, rng=rng)
    grid = np.zeros((steps, length), dtype=int)
    grid[0] = occupied.astype(int)

    for t in range(1, steps):
        occupied, speeds = step(
            occupied,
            speeds,
            vmax=vmax,
            p_slow=p_slow,
            rng=rng,
        )
        grid[t] = occupied.astype(int)

    return grid


def main():
    """Plot a space-time diagram for the traffic model."""
    grid = simulate(steps=200, length=100, density=0.2, vmax=5, p_slow=0.3, seed=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(grid, cmap="binary", interpolation="nearest", aspect="auto")
    ax.set_xlabel("Road position")
    ax.set_ylabel("Time")
    ax.set_title("Traffic CA (Nagel-Schreckenberg)")
    plt.show()


if __name__ == "__main__":
    main()
