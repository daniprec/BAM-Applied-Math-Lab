"""Motter-Lai model of cascading overload failures.

The model follows Motter & Lai (2002), Phys. Rev. E 66, 065102(R):

- The load ``L_i`` of node ``i`` is the number of shortest paths between
  pairs of other nodes that pass through ``i`` (its unnormalized
  betweenness centrality).
- The capacity of node ``i`` is ``C_i = (1 + alpha) * L_i(0)``, where
  ``L_i(0)`` is the load in the intact network and ``alpha >= 0`` is the
  tolerance parameter.
- A trigger removes one node (or a few). Loads are recomputed on the
  remaining network. Every node whose new load exceeds its capacity fails
  and is removed. The process repeats until no node is overloaded.
- The damage is measured by ``G = N' / N``, the size of the largest
  connected component after the cascade divided by the original size.

The model is deliberately simple: loads follow shortest paths, not power
flow physics, and it has no time scale. Use it to study how topology and
spare capacity shape cascades, not to reproduce a real blackout.
"""

from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass
class CascadeResult:
    """Outcome of one cascade.

    Attributes
    ----------
    G_ratio : float
        Largest connected component after the cascade divided by the
        number of nodes in the intact network (Motter-Lai ``G``).
    failed_per_step : list of int
        Number of nodes removed at each step. Entry 0 is the trigger.
    remaining : networkx.Graph
        The network that survives the cascade.
    failed_nodes : list of lists
        Nodes removed at each step. Entry 0 holds the trigger(s).
    """

    G_ratio: float
    failed_per_step: list = field(default_factory=list)
    remaining: nx.Graph = None
    failed_nodes: list = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        """Number of overload steps after the trigger."""
        return len(self.failed_per_step) - 1

    @property
    def n_failed(self) -> int:
        """Total number of removed nodes, trigger included."""
        return int(sum(self.failed_per_step))


def node_loads(G: nx.Graph) -> dict:
    """Return the Motter-Lai load of every node.

    The load is the unnormalized betweenness centrality: the number of
    shortest paths between pairs of other nodes that pass through the
    node, with ties split equally among the shortest paths.

    Parameters
    ----------
    G : networkx.Graph
        Undirected, unweighted graph.

    Returns
    -------
    dict
        Mapping node -> load.
    """
    return nx.betweenness_centrality(G, normalized=False)


def node_capacities(loads: dict, alpha: float) -> dict:
    """Return the capacity ``C_i = (1 + alpha) * L_i(0)`` of every node.

    Parameters
    ----------
    loads : dict
        Initial loads ``L_i(0)``, e.g. from :func:`node_loads`.
    alpha : float
        Tolerance parameter, ``alpha >= 0``.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    return {node: (1.0 + alpha) * load for node, load in loads.items()}


def largest_component_size(G: nx.Graph) -> int:
    """Return the number of nodes in the largest connected component."""
    if G.number_of_nodes() == 0:
        return 0
    return len(max(nx.connected_components(G), key=len))


def select_trigger(G: nx.Graph, strategy: str = "random", rng=None, loads=None):
    """Pick the node that fails first.

    Parameters
    ----------
    G : networkx.Graph
        Intact network.
    strategy : {"random", "degree", "load"}
        ``"random"`` picks a uniformly random node. ``"degree"`` picks the
        node with the largest degree. ``"load"`` picks the node with the
        largest initial load (the attack studied by Motter & Lai).
    rng : numpy.random.Generator, optional
        Random generator for the ``"random"`` strategy.
    loads : dict, optional
        Precomputed initial loads, to avoid recomputing them.
    """
    if strategy == "random":
        rng = np.random.default_rng() if rng is None else rng
        nodes = list(G.nodes())
        return nodes[rng.integers(len(nodes))]
    if strategy == "degree":
        return max(G.degree(), key=lambda kv: kv[1])[0]
    if strategy == "load":
        loads = node_loads(G) if loads is None else loads
        return max(loads, key=loads.get)
    raise ValueError("strategy must be 'random', 'degree' or 'load'")


def run_cascade(
    G: nx.Graph,
    trigger,
    alpha: float,
    loads: dict = None,
    max_steps: int = 100,
    tol: float = 1e-9,
) -> CascadeResult:
    """Run one Motter-Lai cascade.

    Parameters
    ----------
    G : networkx.Graph
        Intact network. It is not modified.
    trigger : node or list of nodes
        Node(s) removed at step 0.
    alpha : float
        Tolerance parameter of the capacity rule.
    loads : dict, optional
        Precomputed initial loads of ``G``.
    max_steps : int
        Safety limit on the number of overload steps.
    tol : float
        Numerical tolerance in the overload test ``L_i > C_i``.

    Returns
    -------
    CascadeResult
    """
    n0 = G.number_of_nodes()
    loads0 = node_loads(G) if loads is None else loads
    capacity = node_capacities(loads0, alpha)

    triggers = list(trigger) if isinstance(trigger, (list, tuple, set)) else [trigger]
    H = G.copy()
    H.remove_nodes_from(triggers)
    failed_per_step = [len(triggers)]
    failed_nodes = [list(triggers)]

    for _ in range(max_steps):
        loads_now = node_loads(H)
        overloaded = [n for n, load in loads_now.items() if load > capacity[n] + tol]
        if not overloaded:
            break
        H.remove_nodes_from(overloaded)
        failed_per_step.append(len(overloaded))
        failed_nodes.append(overloaded)

    return CascadeResult(
        G_ratio=largest_component_size(H) / n0,
        failed_per_step=failed_per_step,
        remaining=H,
        failed_nodes=failed_nodes,
    )


def cascade_curve(
    G: nx.Graph,
    alphas,
    strategy: str = "random",
    n_trials: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Return the mean ``G`` for each tolerance value in ``alphas``.

    Parameters
    ----------
    G : networkx.Graph
        Intact network.
    alphas : array-like
        Tolerance values.
    strategy : {"random", "degree", "load"}
        Trigger selection. Deterministic strategies run one trial per alpha.
    n_trials : int
        Number of random triggers per alpha when ``strategy="random"``.
    seed : int
        Seed of the random generator.
    """
    rng = np.random.default_rng(seed)
    loads0 = node_loads(G)
    trials = n_trials if strategy == "random" else 1
    out = np.zeros(len(alphas))
    for i, alpha in enumerate(alphas):
        values = []
        for _ in range(trials):
            trigger = select_trigger(G, strategy, rng=rng, loads=loads0)
            values.append(run_cascade(G, trigger, alpha, loads=loads0).G_ratio)
        out[i] = np.mean(values)
    return out
