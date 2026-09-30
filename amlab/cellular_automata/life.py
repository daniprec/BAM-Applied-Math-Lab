"""Conway's Game of Life on a finite grid.

Rules (B3/S23): a dead cell with exactly three live neighbours becomes alive;
a live cell with two or three live neighbours survives; every other cell is
dead in the next generation. Neighbours are the eight surrounding cells
(Moore neighbourhood) [Gardner 1970].
"""

from __future__ import annotations

import numpy as np

PATTERNS: dict[str, np.ndarray] = {
    # still lifes
    "block": np.array([[1, 1], [1, 1]]),
    "beehive": np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0]]),
    "loaf": np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]]),
    "boat": np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0]]),
    # oscillators
    "blinker": np.array([[1, 1, 1]]),
    "toad": np.array([[0, 1, 1, 1], [1, 1, 1, 0]]),
    "beacon": np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]),
    # spaceship (moves down and to the right)
    "glider": np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]]),
}


def count_neighbours(grid: np.ndarray, boundary: str = "periodic") -> np.ndarray:
    """Number of live cells among the eight neighbours of each cell.

    Parameters
    ----------
    grid : np.ndarray
        Two-dimensional array of 0s and 1s.
    boundary : {"periodic", "fixed"}
        ``"periodic"`` wraps the grid into a torus. ``"fixed"`` treats every
        cell outside the grid as permanently dead.
    """
    g = np.asarray(grid, dtype=np.uint8)
    if boundary == "periodic":
        total = np.zeros(g.shape, dtype=np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    total += np.roll(np.roll(g, dy, axis=0), dx, axis=1)
        return total
    if boundary == "fixed":
        p = np.pad(g, 1)
        h, w = g.shape
        total = np.zeros(g.shape, dtype=np.uint8)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                if (dy, dx) != (1, 1):
                    total += p[dy : dy + h, dx : dx + w]
        return total
    raise ValueError("boundary must be 'periodic' or 'fixed'")


def life_step(grid: np.ndarray, boundary: str = "periodic") -> np.ndarray:
    """Advance the Game of Life by one generation (rule B3/S23)."""
    g = np.asarray(grid, dtype=np.uint8)
    n = count_neighbours(g, boundary)
    born = (g == 0) & (n == 3)
    survive = (g == 1) & ((n == 2) | (n == 3))
    return (born | survive).astype(np.uint8)


def simulate_life(grid: np.ndarray, steps: int, boundary: str = "periodic") -> np.ndarray:
    """Return an array of shape ``(steps, *grid.shape)`` with every generation."""
    g = np.asarray(grid, dtype=np.uint8)
    out = np.zeros((steps, *g.shape), dtype=np.uint8)
    out[0] = g
    for t in range(1, steps):
        out[t] = life_step(out[t - 1], boundary)
    return out


def place(shape: tuple[int, int], pattern: np.ndarray | str, top: int, left: int) -> np.ndarray:
    """Return an empty grid of ``shape`` with ``pattern`` placed at ``(top, left)``."""
    if isinstance(pattern, str):
        pattern = PATTERNS[pattern]
    grid = np.zeros(shape, dtype=np.uint8)
    h, w = pattern.shape
    grid[top : top + h, left : left + w] = pattern
    return grid


def period_and_shift(grid: np.ndarray, max_steps: int = 50) -> tuple[int, tuple[int, int]] | None:
    """Smallest period of a pattern on a periodic grid, up to a translation.

    Returns ``(period, (dy, dx))`` where the pattern at time ``period`` equals
    the initial pattern shifted by ``(dy, dx)``, or ``None`` if no repeat is
    found within ``max_steps``. A still life has period 1 and shift (0, 0); an
    oscillator has shift (0, 0); a spaceship has a nonzero shift.
    """
    g0 = np.asarray(grid, dtype=np.uint8)
    ys0, xs0 = np.nonzero(g0)
    if ys0.size == 0:
        return None
    g = g0.copy()
    h, w = g0.shape
    for t in range(1, max_steps + 1):
        g = life_step(g)
        ys, xs = np.nonzero(g)
        if ys.size != ys0.size:
            continue
        dy = (ys.min() - ys0.min()) % h
        dx = (xs.min() - xs0.min()) % w
        if np.array_equal(np.roll(np.roll(g0, dy, axis=0), dx, axis=1), g):
            dy = dy - h if dy > h // 2 else dy
            dx = dx - w if dx > w // 2 else dx
            return t, (int(dy), int(dx))
    return None
