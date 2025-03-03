import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def create_networks(n: int = 30, p: float = 0.1, k: int = 4, m: int = 2) -> tuple:
    """Create three types of networks: Random, Small-World, and Scale-Free.

    Parameters
    ----------
    n : int, optional
        Number of nodes, by default 30
    p : float, optional
        Probability for random network, by default 0.1
    k : int, optional
        Each node is connected to k nearest neighbors in small-world, by default 4
    m : int, optional
        Each new node in scale-free attaches to m existing nodes, by default 2

    Returns
    -------
    tuple
        Tuple of three NetworkX graph objects: Random, Small-World, and Scale-Free
    """
    graph_random = nx.erdos_renyi_graph(n, p)
    graph_small_world = nx.watts_strogatz_graph(n, k, p)
    graph_scale_free = nx.barabasi_albert_graph(n, m)

    return graph_random, graph_small_world, graph_scale_free


def compute_metrics(graph: nx.Graph):
    """Compute the average degree, clustering coefficient, and average shortest path length.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph object

    Returns
    -------
    degrees : list
        List of degrees for each node
    centrality : dict
        Dictionary of centrality values for each node
    """
    degrees = [d for _, d in graph.degree()]
    avg_degree = np.mean(degrees)
    clustering_coefficient = nx.average_clustering(graph)

    if nx.is_connected(graph):
        avg_shortest_path = nx.average_shortest_path_length(graph)
    else:
        avg_shortest_path = np.inf

    centrality = nx.degree_centrality(graph)

    dict_output = {
        "degrees": degrees,
        "centrality": centrality,
        "average_degree": avg_degree,
        "clustering_coefficient": clustering_coefficient,
        "average_shortest_path": avg_shortest_path,
    }

    return dict_output


def plot_networks(ls_networks: list[nx.Graph]):
    """Plot the three networks."""
    fig, axes = plt.subplots(1, len(ls_networks), figsize=(15, 5))

    for ax, graph in zip(axes, ls_networks):
        # Compute metrics
        dict_metrics = compute_metrics(graph)
        # Draw the network
        nx.draw(graph, ax=ax, with_labels=True, node_size=100, font_size=8)
        # Add some metrics to the plot
        ax.set_title(
            f"Average Degree: {dict_metrics['average_degree']:.2f}\n"
            f"Clustering Coefficient: {dict_metrics['clustering_coefficient']:.2f}\n"
            f"Average Shortest Path: {dict_metrics['average_shortest_path']:.2f}"
        )

    plt.show()


def main():
    ls_graphs = create_networks()
    plot_networks(ls_graphs)


if __name__ == "__main__":
    main()
