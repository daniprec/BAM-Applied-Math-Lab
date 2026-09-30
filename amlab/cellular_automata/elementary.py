"""Vectorized tools for elementary cellular automata (ECA).

An elementary cellular automaton has binary states and a three-cell
neighbourhood. A rule is identified by its Wolfram number (0 to 255): bit
``n`` of the rule number is the new state for the neighbourhood whose binary
value is ``n``, where the neighbourhood ``(left, centre, right)`` is read as
the binary number ``4 * left + 2 * centre + right`` [Wolfram 1983].

These functions complement :mod:`amlab.cellular_automata.cellular`, which keeps
the original loop-based ``apply_rule`` used in the lab pages.
"""

from __future__ import annotations

import numpy as np


def rule_table(rule: int) -> np.ndarray:
    """Return the lookup table of an elementary rule.

    Parameters
    ----------
    rule : int
        Wolfram rule number, between 0 and 255.

    Returns
    -------
    np.ndarray
        Integer array of length 8. Entry ``n`` is the new state for the
        neighbourhood with binary value ``n`` (``000`` is 0, ``111`` is 7).
    """
    if not 0 <= rule <= 255:
        raise ValueError("rule must be between 0 and 255")
    return np.array([(rule >> n) & 1 for n in range(8)], dtype=np.uint8)


def step_eca(
    state: np.ndarray,
    rule: int,
    boundary: str = "periodic",
    fixed_value: int = 0,
) -> np.ndarray:
    """Apply one synchronous update of an elementary rule.

    Parameters
    ----------
    state : np.ndarray
        One-dimensional array of 0s and 1s.
    rule : int
        Wolfram rule number.
    boundary : {"periodic", "fixed"}
        ``"periodic"`` joins the two ends of the row into a ring.
        ``"fixed"`` surrounds the row with two ghost cells that always hold
        ``fixed_value``. Every real cell, including the edge cells, is updated.
    fixed_value : int
        State of the ghost cells when ``boundary="fixed"``.

    Returns
    -------
    np.ndarray
        The new row, same shape as ``state``.
    """
    state = np.asarray(state, dtype=np.uint8)
    table = rule_table(rule)
    if boundary == "periodic":
        left = np.roll(state, 1)
        right = np.roll(state, -1)
    elif boundary == "fixed":
        padded = np.concatenate(([fixed_value], state, [fixed_value])).astype(np.uint8)
        left = padded[:-2]
        right = padded[2:]
    else:
        raise ValueError("boundary must be 'periodic' or 'fixed'")
    index = 4 * left + 2 * state + right
    return table[index]


def simulate_eca(
    initial_state: np.ndarray,
    rule: int,
    steps: int,
    boundary: str = "periodic",
    fixed_value: int = 0,
) -> np.ndarray:
    """Return the space-time diagram of an elementary rule.

    Parameters
    ----------
    initial_state : np.ndarray
        Initial row of 0s and 1s.
    rule : int
        Wolfram rule number.
    steps : int
        Number of rows in the output, including the initial row.
    boundary, fixed_value
        Passed to :func:`step_eca`.

    Returns
    -------
    np.ndarray
        Array of shape ``(steps, len(initial_state))``.
    """
    row = np.asarray(initial_state, dtype=np.uint8)
    grid = np.zeros((steps, row.size), dtype=np.uint8)
    grid[0] = row
    for t in range(1, steps):
        grid[t] = step_eca(grid[t - 1], rule, boundary, fixed_value)
    return grid


def single_seed(width: int) -> np.ndarray:
    """Return a row of zeros with a single 1 in the centre."""
    row = np.zeros(width, dtype=np.uint8)
    row[width // 2] = 1
    return row


def random_row(width: int, density: float = 0.5, seed: int | None = 0) -> np.ndarray:
    """Return a random row where each cell is 1 with probability ``density``."""
    rng = np.random.default_rng(seed)
    return (rng.random(width) < density).astype(np.uint8)


def langton_lambda(rule: int) -> float:
    """Langton's lambda parameter of an elementary rule.

    With the quiescent state 0, lambda is the fraction of the 8 neighbourhoods
    that are mapped to the non-quiescent state 1 [Langton 1990].
    """
    return rule_table(rule).sum() / 8


def reflect_rule(rule: int) -> int:
    """Rule obtained by exchanging left and right neighbours."""
    table = rule_table(rule)
    new = 0
    for n in range(8):
        left, centre, right = (n >> 2) & 1, (n >> 1) & 1, n & 1
        mirrored = 4 * right + 2 * centre + left
        new |= int(table[mirrored]) << n
    return new


def complement_rule(rule: int) -> int:
    """Rule obtained by exchanging the roles of 0 and 1."""
    table = rule_table(rule)
    new = 0
    for n in range(8):
        new |= (1 - int(table[7 - n])) << n
    return new


def equivalence_class(rule: int) -> set[int]:
    """Rules equivalent to ``rule`` under reflection and complement."""
    r, c = reflect_rule(rule), complement_rule(rule)
    return {rule, r, c, complement_rule(r)}


def transient_and_period(initial_state: np.ndarray, rule: int) -> tuple[int, int]:
    """Transient length and period of an orbit on a periodic lattice.

    A periodic lattice of ``N`` cells has only ``2**N`` configurations, so
    every orbit eventually repeats. Use small ``N`` (up to about 20).

    Returns
    -------
    tuple[int, int]
        Number of steps before the orbit enters its cycle, and cycle length.
    """
    seen: dict[bytes, int] = {}
    row = np.asarray(initial_state, dtype=np.uint8)
    t = 0
    while row.tobytes() not in seen:
        seen[row.tobytes()] = t
        row = step_eca(row, rule, "periodic")
        t += 1
    first = seen[row.tobytes()]
    return first, t - first
