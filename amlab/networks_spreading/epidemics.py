"""Fast discrete-time SIS and SIR simulations on networks.

The update rule matches the guided exercises of Module 5: at each step a
susceptible node with ``m`` infected neighbours becomes infected with
probability ``1 - (1 - beta)**m`` and an infected node recovers with
probability ``mu``. All nodes update synchronously.

The functions use a sparse adjacency matrix instead of loops over
``networkx`` nodes, so a few hundred runs on graphs with about a thousand
nodes take seconds.
"""

import networkx as nx
import numpy as np


def _adjacency(G: nx.Graph):
    """Return the sparse adjacency matrix and the node list of ``G``."""
    nodes = list(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr", dtype=float)
    return A, nodes


def run_sis_fast(G, beta, mu, steps=200, initial_infected=5, rng=None):
    """Simulate the discrete-time SIS model and return the infected fraction.

    Parameters
    ----------
    G : networkx.Graph
        Contact network.
    beta : float
        Infection probability per infected neighbour and step.
    mu : float
        Recovery probability per step.
    steps : int
        Number of time steps.
    initial_infected : int
        Number of randomly chosen initially infected nodes.
    rng : numpy.random.Generator, optional

    Returns
    -------
    numpy.ndarray
        Infected fraction at times ``0, 1, ..., steps``.
    """
    rng = np.random.default_rng() if rng is None else rng
    A, nodes = _adjacency(G)
    n = len(nodes)
    infected = np.zeros(n, dtype=bool)
    infected[rng.choice(n, size=initial_infected, replace=False)] = True
    rho = np.empty(steps + 1)
    rho[0] = infected.mean()
    for t in range(1, steps + 1):
        m = A @ infected.astype(float)
        p_inf = 1.0 - (1.0 - beta) ** m
        new_inf = (~infected) & (rng.random(n) < p_inf)
        recover = infected & (rng.random(n) < mu)
        infected = (infected | new_inf) & ~recover
        rho[t] = infected.mean()
    return rho


def run_sir_fast(G, beta, mu, initial_infected=1, immune=None, rng=None, max_steps=10_000):
    """Simulate the discrete-time SIR model until no node is infected.

    Parameters
    ----------
    G : networkx.Graph
        Contact network.
    beta : float
        Infection probability per infected neighbour and step.
    mu : float
        Recovery probability per step (must be positive).
    initial_infected : int
        Number of initially infected nodes, chosen among non-immune nodes.
    immune : iterable of nodes, optional
        Nodes that start in the recovered state (vaccinated).
    rng : numpy.random.Generator, optional
    max_steps : int
        Safety limit.

    Returns
    -------
    float
        Final epidemic size: fraction of all nodes that were ever infected.
    """
    rng = np.random.default_rng() if rng is None else rng
    A, nodes = _adjacency(G)
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    recovered = np.zeros(n, dtype=bool)
    if immune is not None:
        recovered[[index[v] for v in immune]] = True
    candidates = np.flatnonzero(~recovered)
    infected = np.zeros(n, dtype=bool)
    infected[rng.choice(candidates, size=initial_infected, replace=False)] = True
    ever = infected.copy()
    for _ in range(max_steps):
        if not infected.any():
            break
        m = A @ infected.astype(float)
        p_inf = 1.0 - (1.0 - beta) ** m
        susceptible = ~infected & ~recovered
        new_inf = susceptible & (rng.random(n) < p_inf)
        recover = infected & (rng.random(n) < mu)
        infected = (infected | new_inf) & ~recover
        recovered = recovered | recover
        ever = ever | new_inf
    return ever.sum() / n


def random_immunization(G, fraction, rng=None):
    """Return a uniformly random set of ``fraction * N`` nodes."""
    rng = np.random.default_rng() if rng is None else rng
    nodes = list(G.nodes())
    k = int(round(fraction * len(nodes)))
    return [nodes[i] for i in rng.choice(len(nodes), size=k, replace=False)]


def targeted_immunization(G, fraction):
    """Return the ``fraction * N`` nodes with the largest degree."""
    k = int(round(fraction * G.number_of_nodes()))
    ranked = sorted(G.degree(), key=lambda kv: kv[1], reverse=True)
    return [node for node, _ in ranked[:k]]


def acquaintance_immunization(G, fraction, rng=None):
    """Return nodes chosen by acquaintance immunization.

    Repeatedly pick a random node and immunize one of its random
    neighbours (Cohen, Havlin & ben-Avraham 2003) until ``fraction * N``
    distinct nodes are immunized. The method needs only local knowledge.
    """
    rng = np.random.default_rng() if rng is None else rng
    nodes = list(G.nodes())
    k = int(round(fraction * len(nodes)))
    chosen = set()
    while len(chosen) < k:
        v = nodes[rng.integers(len(nodes))]
        nbrs = list(G.neighbors(v))
        if nbrs:
            chosen.add(nbrs[rng.integers(len(nbrs))])
    return list(chosen)
