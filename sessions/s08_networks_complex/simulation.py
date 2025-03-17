import random
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes


def initial_state(G: nx.Graph) -> dict:
    """Assigns the initial state of the nodes in the graph,
    in the form of a dictionary.

    Parameters
    ----------
    G : nx.Graph
        The graph on which the simulation is run.

    Returns
    -------
    dict
        A dictionary containing the initial state of the nodes.
        The keys are the nodes and the values are the states.
        The states are represented as strings, with "S" for
        susceptible, "I" for infected, and "R" for recovered.
    """
    # Initialize all nodes as susceptible
    state = {}
    for node in G.nodes:
        state[node] = "S"

    # Infect a random node
    patient_zero = random.choice(list(G.nodes))
    state[patient_zero] = "I"
    return state


def state_transition(
    G: nx.Graph, current_state: dict, mu: float = 0.1, beta: float = 0.1
) -> dict:
    """Determines the next state of the nodes in the graph,"
    based on the current state."

    Parameters
    ----------
    G : nx.Graph
        The graph on which the simulation is run.
    current_state : dict
        A dictionary containing the current state of the nodes.
        The keys are the nodes and the values are the states.
        The states are represented as strings, with "S" for
        susceptible, "I" for infected, and "R" for recovered.
    mu : float
        The probability of recovery, by default 0.1.
    beta : float
        The probability of infection, by default 0.1.

    Returns
    -------
    dict
        A dictionary containing the next state of the nodes.
        The keys are the nodes and the values are the states.
        The states are represented as strings, with "S" for
        susceptible, "I" for infected, and "R" for recovered.
    """
    # Initialize the next state as an empty dictionary
    next_state = {}
    # Loop over all nodes in the graph
    for node in G.nodes:
        # If the node is infected, it may recover
        if current_state[node] == "I":
            if random.random() < mu:
                next_state[node] = "S"
        # If the node is susceptible, it may become infected
        else:
            # There is a chance of infection from each infected neighbor
            for neighbor in G.neighbors(node):
                if current_state[neighbor] == "I":
                    if random.random() < beta:
                        next_state[node] = "I"

    return next_state


class StopCondition(StopIteration):
    pass


class Simulation:
    """Simulate state transitions on a network"""

    def __init__(
        self,
        G: nx.Graph,
        initial_state: callable,
        state_transition: callable,
        stop_condition: callable | None = None,
        name: str = "",
    ):
        """
        Create a Simulation instance.

        Parameters
        ----------
        G : nx.Graph
            The graph on which the simulation is run.
        initial_state : callable
            Function with signature `initial_state(G)`, that
            accepts a single argument, the Graph, and returns a dictionary
            of all node states. The keys in this dict should be node names
            and the values the corresponding initial node state.
        state_transition : callable
            Function with signature
            `state_transition(G, current_state)` that accepts two
            arguments, the Graph and a dictionary of current node states,
            and returns a dictionary of updated node states. The keys in
            this dict should be node names and the values the corresponding
            updated node state.
        stop_condition : callable, optional
            Function with signature
            `stop_condition(G, current_state)` that accepts two arguments,
            the Graph and a dictionary of current node states, and returns
            True if the simulation should be stopped at its current state.
        name : str, optional
            A string used in titles of plots and drawings.

        Raises
        ------
        ValueError
            If not all graph nodes have an initial state.
        TypeError
            If `stop_condition` is not a function.
        """
        self.G = G.copy()
        self._initial_state = initial_state
        self._state_transition = state_transition
        self._stop_condition = stop_condition
        # It is okay to specify stop_condition=False
        if stop_condition and not callable(stop_condition):
            raise TypeError("'stop_condition' should be a function")
        self.name = name or "Simulation"

        self._states = []
        self._value_index = {}
        self._cmap = plt.cm.get_cmap("tab10")

        self._initialize()

        self._pos = nx.layout.spring_layout(G)

    def _append_state(self, state: dict):
        """Append a state to the list of states and update the value index."""
        self._states.append(state)
        # Update self._value_index
        for value in set(state.values()):
            if value not in self._value_index:
                self._value_index[value] = len(self._value_index)

    def _initialize(self):
        """Initialize the simulation by setting the initial state."""
        if self._initial_state:
            if callable(self._initial_state):
                state = self._initial_state(self.G)
            else:
                state = self._initial_state
            nx.set_node_attributes(self.G, state, "state")

        if any(self.G.nodes[n].get("state") is None for n in self.G.nodes):
            raise ValueError("All nodes must have an initial state")

        self._append_state(state)

    def _step(self):
        """Advance the simulation by one step."""
        # We're choosing to use the node attributes as the source of truth.
        # This allows the user to manually perturb the network in between steps.
        state = nx.get_node_attributes(self.G, "state")
        if self._stop_condition and self._stop_condition(self.G, state):
            raise StopCondition
        state = nx.get_node_attributes(self.G, "state")
        new_state = self._state_transition(self.G, state)
        state.update(new_state)
        nx.set_node_attributes(self.G, state, "state")
        self._append_state(state)

    def _categorical_color(self, value):
        index = self._value_index[value]
        node_color = self._cmap(index)
        return node_color

    @property
    def steps(self):
        """Returns the number of steps the sumulation has run"""
        return len(self._states) - 1

    def state(self, step: int = -1):
        """
        Get a state of the simulation; by default returns the current state.

        Parameters
        ----------
        step : int, optional
            The step of the simulation to return. Default is -1, the
            current state.

        Returns
        -------
        dict
            Dictionary of node states.

        Raises
        ------
        IndexError
            If `step` argument is greater than the number of steps.
        """
        try:
            return self._states[step]
        except IndexError:
            raise IndexError("Simulation step %i out of range" % step)

    def draw(self, step: int = -1, labels: list | None = None, **kwargs):
        """
        Use networkx.draw to draw a simulation state with nodes colored by
        their state value. By default, draws the current state.

        Parameters
        ----------
        step : int, optional
            The step of the simulation to draw. Default is -1, the
            current state.
        labels : list, optional
            Ordered sequence of state values to plot. Default is all
            observed state values, approximately ordered by appearance.
        kwargs : dict
            Keyword arguments are passed to networkx.draw()

        Raises
        ------
        IndexError
            If `step` argument is greater than the number of steps.
        """
        state = self.state(step)
        node_colors = [self._categorical_color(state[n]) for n in self.G.nodes]
        nx.draw(self.G, pos=self._pos, node_color=node_colors, **kwargs)

        if labels is None:
            labels = sorted(set(state.values()), key=self._value_index.get)
        patches = [
            mpl.patches.Patch(color=self._categorical_color(label), label=label)
            for label in labels
        ]
        plt.legend(handles=patches)

        if step == -1:
            step = self.steps
        if step == 0:
            title = "initial state"
        else:
            title = "step %i" % (step)
        if self.name:
            title = "{}: {}".format(self.name, title)
        plt.title(title)

    def plot(
        self,
        min_step: int = None,
        max_step: int = None,
        labels: list | None = None,
        **kwargs,
    ) -> Axes:
        """
        Use pyplot to plot the relative number of nodes with each state at each
        simulation step. By default, plots all simulation steps.

        Parameters
        ----------
        min_step : int, optional
            The first step of the simulation to draw. Default is None, which
            plots starting from the initial state.
        max_step : int, optional
            The last step, not inclusive, of the simulation to draw. Default is
            None, which plots up to the current step.
        labels : list, optional
            Ordered sequence of state values to plot. Default is all
            observed state values, approximately ordered by appearance.
        kwargs : dict
            Keyword arguments are passed along to plt.plot()

        Returns
        -------
        matplotlib.axes.Axes
            Axes object for the current plot
        """
        x_range = range(min_step or 0, max_step or len(self._states))
        counts = [Counter(s.values()) for s in self._states[min_step:max_step]]
        if labels is None:
            labels = {k for count in counts for k in count}
            labels = sorted(labels, key=self._value_index.get)

        for label in labels:
            series = [count.get(label, 0) / sum(count.values()) for count in counts]
            plt.plot(x_range, series, label=label, **kwargs)

        title = "node state proportions"
        if self.name:
            title = "{}: {}".format(self.name, title)
        plt.title(title)
        plt.xlabel("Simulation step")
        plt.ylabel("Proportion of nodes")
        plt.legend()
        plt.xlim(x_range.start)

        return plt.gca()

    def run(self, steps: int = 1):
        """
        Run the simulation one or more steps, as specified by the `steps`
        argument. Default is to run a single step.

        Parameters
        ----------
        steps : int, optional
            Number of steps to advance the simulation. Default is 1.
        """
        for _ in range(steps):
            try:
                self._step()
            except StopCondition:
                print("Stop condition met at step %i." % self.steps)
                break
