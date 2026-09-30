"""Vectorized Nagel-Schreckenberg traffic models and common extensions.

Cars live on a ring of ``length`` cells. Each car stores a position and an
integer speed. One update applies, in parallel for all cars [Nagel and
Schreckenberg 1992]:

1. acceleration, ``v <- min(v + 1, vmax)``;
2. braking, ``v <- min(v, gap)``, where ``gap`` is the number of empty cells
   to the car ahead;
3. randomization, ``v <- max(v - 1, 0)`` with probability ``p``;
4. motion, ``x <- x + v``.

Extensions in this module:

* velocity-dependent randomization (slow-to-start): cars that were stopped
  brake randomly with probability ``p0`` instead of ``p`` [Barlovic et al. 1998];
* heterogeneous maximum speeds (a fraction of slow vehicles);
* a symmetric two-lane model with a simple lane-changing rule in the spirit of
  Rickert et al. (1996).

The original cell-based code in :mod:`amlab.cellular_automata.traffic` is kept
for the lab pages.
"""

from __future__ import annotations

import numpy as np


def ring_initial(
    length: int,
    n_cars: int,
    mode: str = "random",
    vmax: int = 5,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Initial positions and speeds of ``n_cars`` on a ring.

    Parameters
    ----------
    length : int
        Number of cells.
    n_cars : int
        Number of cars, at most ``length``.
    mode : {"random", "homogeneous", "jam"}
        ``"random"``: random distinct cells, speed 0.
        ``"homogeneous"``: evenly spaced cars moving at the largest speed that
        the spacing allows (at most ``vmax``).
        ``"jam"``: one compact block of stopped cars.
    vmax : int
        Maximum speed, used by ``"homogeneous"``.
    rng : np.random.Generator | None
        Random generator for ``"random"``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Positions (sorted, cyclic order) and speeds.
    """
    if n_cars > length:
        raise ValueError("n_cars cannot exceed length")
    if mode == "random":
        rng = np.random.default_rng() if rng is None else rng
        x = np.sort(rng.choice(length, size=n_cars, replace=False))
        v = np.zeros(n_cars, dtype=int)
    elif mode == "homogeneous":
        x = np.floor(np.arange(n_cars) * length / max(n_cars, 1)).astype(int)
        gaps = (np.roll(x, -1) - x - 1) % length
        v = np.minimum(gaps.min() if n_cars else 0, vmax) * np.ones(n_cars, dtype=int)
    elif mode == "jam":
        x = np.arange(n_cars)
        v = np.zeros(n_cars, dtype=int)
    else:
        raise ValueError("mode must be 'random', 'homogeneous' or 'jam'")
    return x.astype(int), v.astype(int)


def gaps_ahead(x: np.ndarray, length: int) -> np.ndarray:
    """Empty cells between each car and the next one (cars in cyclic order)."""
    if x.size == 1:
        return np.array([length - 1])
    return (np.roll(x, -1) - x - 1) % length


def nasch_step(
    x: np.ndarray,
    v: np.ndarray,
    length: int,
    vmax: int | np.ndarray = 5,
    p: float = 0.3,
    p0: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One parallel update of the (extended) Nagel-Schreckenberg model.

    Parameters
    ----------
    x, v : np.ndarray
        Positions (in cyclic order) and speeds. Order is preserved because
        cars cannot overtake on a single lane.
    length : int
        Ring length.
    vmax : int or np.ndarray
        Maximum speed, one value or one per car.
    p : float
        Randomization probability for moving cars.
    p0 : float | None
        Randomization probability for cars with ``v == 0`` before the update
        (slow-to-start). ``None`` means ``p0 = p`` (standard model).
    rng : np.random.Generator | None
        Random generator.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        New positions and speeds.
    """
    rng = np.random.default_rng() if rng is None else rng
    if x.size == 0:
        return x, v
    prob = np.full(x.size, p, dtype=float)
    if p0 is not None:
        prob[v == 0] = p0
    v = np.minimum(v + 1, vmax)
    v = np.minimum(v, gaps_ahead(x, length))
    slow = rng.random(x.size) < prob
    v = np.where(slow, np.maximum(v - 1, 0), v)
    x = (x + v) % length
    return x, v


def space_time(
    length: int = 200,
    density: float = 0.2,
    steps: int = 200,
    vmax: int | np.ndarray = 5,
    p: float = 0.3,
    p0: float | None = None,
    mode: str = "random",
    seed: int | None = 0,
) -> np.ndarray:
    """Occupancy space-time diagram of shape ``(steps, length)``."""
    rng = np.random.default_rng(seed)
    n = int(round(density * length))
    vm = vmax if np.isscalar(vmax) else int(np.max(vmax))
    x, v = ring_initial(length, n, mode, vm, rng)
    grid = np.zeros((steps, length), dtype=np.uint8)
    for t in range(steps):
        grid[t, x] = 1
        x, v = nasch_step(x, v, length, vmax, p, p0, rng)
    return grid


def mean_flow(
    length: int,
    n_cars: int,
    steps: int = 600,
    burn_in: int = 200,
    vmax: int | np.ndarray = 5,
    p: float = 0.3,
    p0: float | None = None,
    mode: str = "random",
    seed: int | None = 0,
) -> float:
    """Time-averaged flow ``q = sum(v) / length`` after a burn-in period."""
    rng = np.random.default_rng(seed)
    vm = vmax if np.isscalar(vmax) else int(np.max(vmax))
    x, v = ring_initial(length, n_cars, mode, vm, rng)
    total = 0.0
    for t in range(steps):
        x, v = nasch_step(x, v, length, vmax, p, p0, rng)
        if t >= burn_in:
            total += v.sum()
    return total / (length * (steps - burn_in))


def fundamental_diagram(
    densities: np.ndarray,
    length: int = 400,
    steps: int = 600,
    burn_in: int = 200,
    vmax: int = 5,
    p: float = 0.3,
    p0: float | None = None,
    mode: str = "random",
    slow_fraction: float = 0.0,
    vmax_slow: int = 3,
    seed: int | None = 0,
) -> np.ndarray:
    """Mean flow for each density on a single-lane ring.

    With ``slow_fraction > 0`` a random subset of
    ``max(1, round(slow_fraction * n_cars))`` cars gets maximum speed
    ``vmax_slow`` instead of ``vmax``.
    """
    rng = np.random.default_rng(seed)
    flows = []
    for rho in densities:
        n = max(1, int(round(rho * length)))
        vm: int | np.ndarray = vmax
        if slow_fraction > 0:
            vm = _max_speeds(n, slow_fraction, vmax, vmax_slow, rng)
        flows.append(
            mean_flow(length, n, steps, burn_in, vm, p, p0, mode, int(rng.integers(1 << 31)))
        )
    return np.array(flows)


def _max_speeds(n: int, slow_fraction: float, vmax: int, vmax_slow: int, rng: np.random.Generator) -> np.ndarray:
    """Per-car maximum speeds with exactly ``max(1, round(slow_fraction * n))`` slow cars."""
    vm = np.full(n, vmax, dtype=int)
    if slow_fraction > 0:
        k = min(n, max(1, int(round(slow_fraction * n))))
        vm[rng.choice(n, size=k, replace=False)] = vmax_slow
    return vm


def deterministic_flow(density: np.ndarray, vmax: int = 5) -> np.ndarray:
    """Stationary flow of the deterministic model (``p = 0``), ``min(rho vmax, 1 - rho)``."""
    density = np.asarray(density, dtype=float)
    return np.minimum(density * vmax, 1 - density)


def _gap_in_lane(pos_lane: np.ndarray, x: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Empty cells ahead of and behind positions ``x`` in a lane with cars at ``pos_lane``.

    Positions in ``x`` are assumed not occupied in that lane.
    """
    if pos_lane.size == 0:
        big = np.full(x.size, length, dtype=int)
        return big, big
    s = np.sort(pos_lane)
    i = np.searchsorted(s, x, side="right")
    ahead = s[i % s.size]
    behind = s[(i - 1) % s.size]
    return (ahead - x - 1) % length, (x - behind - 1) % length


def two_lane_step(
    x: np.ndarray,
    lane: np.ndarray,
    v: np.ndarray,
    length: int,
    vmax: int | np.ndarray = 5,
    p: float = 0.3,
    p_change: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One update of a symmetric two-lane model.

    First, all cars decide in parallel whether to change lane. A car changes
    lane (with probability ``p_change``) if its gap ahead is smaller than
    ``v + 1``, the gap ahead in the other lane is larger than its own gap, the
    target cell is empty, and the gap behind in the other lane is at least
    ``vmax``. If two cars target the same cell, only one moves. Then each lane
    applies the Nagel-Schreckenberg update.

    Parameters
    ----------
    x, lane, v : np.ndarray
        Position, lane (0 or 1) and speed of every car.
    length : int
        Ring length.
    vmax : int or np.ndarray
        Maximum speed, one value or one per car.
    p : float
        Randomization probability.
    p_change : float
        Probability of executing an allowed lane change.
    rng : np.random.Generator | None
        Random generator.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        New positions, lanes and speeds.
    """
    rng = np.random.default_rng() if rng is None else rng
    vm = np.broadcast_to(np.asarray(vmax), x.shape)
    occ = np.zeros((2, length), dtype=bool)
    occ[lane, x] = True

    own_gap = np.empty(x.size, dtype=int)
    other_ahead = np.empty(x.size, dtype=int)
    other_behind = np.empty(x.size, dtype=int)
    for k in (0, 1):
        in_k = lane == k
        pos_k = x[in_k]
        order = np.argsort(pos_k)
        g = np.empty(pos_k.size, dtype=int)
        g[order] = gaps_ahead(pos_k[order], length) if pos_k.size else g[order]
        own_gap[in_k] = g
        a, b = _gap_in_lane(x[lane == 1 - k], pos_k, length)
        other_ahead[in_k] = a
        other_behind[in_k] = b

    target_free = ~occ[1 - lane, x]
    want = (
        (own_gap < v + 1)
        & (other_ahead > own_gap)
        & target_free
        & (other_behind >= vm)
        & (rng.random(x.size) < p_change)
    )
    new_lane = np.where(want, 1 - lane, lane)
    # resolve conflicts: two cars cannot occupy the same cell
    cell = new_lane * length + x
    _, first = np.unique(cell, return_index=True)
    keep = np.zeros(x.size, dtype=bool)
    keep[first] = True
    lane = np.where(keep, new_lane, lane)

    new_x = x.copy()
    new_v = v.copy()
    for k in (0, 1):
        idx = np.nonzero(lane == k)[0]
        if idx.size == 0:
            continue
        order = idx[np.argsort(x[idx])]
        xk, vk = nasch_step(x[order], v[order], length, vm[order], p, None, rng)
        new_x[order] = xk
        new_v[order] = vk
    return new_x, lane, new_v


def two_lane_flow(
    length: int,
    density: float,
    steps: int = 500,
    burn_in: int = 200,
    vmax: int = 5,
    p: float = 0.3,
    p_change: float = 1.0,
    slow_fraction: float = 0.0,
    vmax_slow: int = 3,
    seed: int | None = 0,
) -> float:
    """Mean flow per lane of the two-lane model, ``sum(v) / (2 length)``."""
    rng = np.random.default_rng(seed)
    n = max(1, int(round(density * 2 * length)))
    cells = rng.choice(2 * length, size=n, replace=False)
    lane, x = cells // length, cells % length
    v = np.zeros(n, dtype=int)
    vm = _max_speeds(n, slow_fraction, vmax, vmax_slow, rng)
    total = 0.0
    for t in range(steps):
        x, lane, v = two_lane_step(x, lane, v, length, vm, p, p_change, rng)
        if t >= burn_in:
            total += v.sum()
    return total / (2 * length * (steps - burn_in))
