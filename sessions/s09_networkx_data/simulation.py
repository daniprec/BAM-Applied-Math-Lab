import random
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes


def initial_state(G: nx.Graph, first_node: int = None) -> dict:
    """Assigns the initial state of the nodes in the graph,
    in the form of a dictionary.

    Parameters
    ----------
    G : nx.Graph
        The graph on which the simulation is run.
    first_node : int
        The node that will be the first spreader. If None,
        a random node is chosen.

    Returns
    -------
    dict
        A dictionary containing the initial state of the nodes.
        The keys are the nodes and the values are the states.
        The states are represented as strings, with "I" for
        ignorant, "S" for spreader, and "R" for stifler.
    """
    # Initialize all nodes as ignorant
    state = {}
    for node in G.nodes:
        state[node] = "I"

    # Choose a random node to be the first spreader
    if first_node is None:
        first_node = random.choice(list(G.nodes))
    state[first_node] = "S"
    return state


def state_transition(
    G: nx.Graph, current_state: dict, gamma: float = 0.1, beta: float = 0.2
) -> dict:
    """Daley-Kendall rumor spreading: Ignorant-Spreader-Stifler.

    States:
    - 'I': Ignorant
    - 'S': Spreader
    - 'R': Stifler
    """
    next_state = current_state.copy()

    for node in G.nodes:
        state = current_state[node]

        if state == "I":
            # Ignorant: may become a spreader if contacted by a spreader
            for neighbor in G.neighbors(node):
                if current_state[neighbor] == "S":
                    if random.random() < beta:
                        next_state[node] = "S"
                        break  # only needs one spreader contact

        elif state == "S":
            # Spreader: may become stifler if contacts another spreader or stifler
            for neighbor in G.neighbors(node):
                if current_state[neighbor] in ("S", "R"):
                    if random.random() < gamma:
                        next_state[node] = "R"
                        break  # only needs one stifler contact

        # Stiflers ('R') remain unchanged

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
        stop_condition: callable = None,
        name: str = "",
        **kwargs,
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
        **kwargs
            Additional keyword arguments are passed to the state transition
            function.

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
        self._kwargs = kwargs
        self._stop_condition = stop_condition
        # It is okay to specify stop_condition=False
        if stop_condition and not callable(stop_condition):
            raise TypeError("'stop_condition' should be a function")
        self.name = name or "Simulation"

        # Initialize the list of past states - we will use it later to
        # plot the evolution of the simulation
        self._ls_states_history = []
        # Initialize the value index - we will use it to assign colors
        # to the nodes in the plot
        self._colors = {"I": "blue", "S": "red", "R": "green"}
        # Initialize the color map
        self._cmap = plt.cm.get_cmap("tab10")

        # Initialize the simulation by calling the initial state function
        self._initialize()

        # Define the layout of the graph
        self._pos = nx.spring_layout(G, iterations=15, seed=1721)

    def _initialize(self):
        """Initialize the simulation by setting the initial state."""
        # Use the initial state function to set the initial state
        state = self._initial_state(self.G)
        nx.set_node_attributes(self.G, state, "state")

        if any(self.G.nodes[n].get("state") is None for n in self.G.nodes):
            raise ValueError("All nodes must have an initial state")

        self._ls_states_history.append(state)

    def _step(self):
        """Advance the simulation by one step."""
        # We're choosing to use the node attributes as the source of truth.
        # This allows the user to manually perturb the network in between steps.
        state = nx.get_node_attributes(self.G, "state")
        if self._stop_condition and self._stop_condition(self.G, state):
            raise StopCondition
        state = nx.get_node_attributes(self.G, "state")
        new_state = self._state_transition(self.G, state, **self._kwargs)
        state.update(new_state)
        nx.set_node_attributes(self.G, state, "state")
        self._ls_states_history.append(state)

    def _categorical_color(self, value: str) -> str:
        """Return a color for a categorical value"""
        node_color = self._colors.get(value, "grey")
        return node_color

    @property
    def steps(self):
        """Returns the number of steps the sumulation has run"""
        return len(self._ls_states_history) - 1

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
            return self._ls_states_history[step]
        except IndexError:
            raise IndexError("Simulation step %i out of range" % step)

    def draw(self, step: int = -1, axis: Axes = None, **kwargs):
        """
        Use networkx.draw to draw a simulation state with nodes colored by
        their state value. By default, draws the current state.

        Parameters
        ----------
        step : int, optional
            The step of the simulation to draw. Default is -1, the
            current state.
        kwargs : dict
            Keyword arguments are passed to networkx.draw()

        Raises
        ------
        IndexError
            If `step` argument is greater than the number of steps.
        """

        axis = axis if axis else plt.gca()

        # Get the state of the simulation at the specified step
        state = self.state(step)
        # Get the color of each node based on its state
        node_colors = [self._categorical_color(state[n]) for n in self.G.nodes]
        # Draw the graph
        nx.draw(self.G, pos=self._pos, node_color=node_colors, ax=axis, **kwargs)

        # Add a legend to the plot, with colors corresponding to the states
        labels = self._colors.keys()
        patches = [
            mpl.patches.Patch(color=self._categorical_color(label), label=label)
            for label in labels
        ]
        axis.legend(handles=patches)

        # Set the title of the plot
        if step == -1:
            step = self.steps
        if step == 0:
            title = "initial state"
        else:
            title = "step %i" % (step)
        if self.name:
            title = "{}: {}".format(self.name, title)
        axis.set_title(title)
        return axis

    def plot(
        self,
        min_step: int = None,
        max_step: int = None,
        axis: Axes = None,
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
        kwargs : dict
            Keyword arguments are passed along to plt.plot()

        Returns
        -------
        matplotlib.axes.Axes
            Axes object for the current plot
        """
        axis = axis if axis else plt.gca()

        # Range of X: first and last step to plot
        x_range = range(min_step or 0, max_step or len(self._ls_states_history))

        # Get the state of the simulation at each step in the range
        # This will be a list of dictionaries, one for each step
        # Each dictionary will contain the counts of each state
        # Example: [{'S': 99, 'I': 1}, {'S': 98, 'I': 2}]
        counts = []
        for state in self._ls_states_history[min_step:max_step]:
            counts.append(Counter(state.values()))
        labels = self._colors.keys()

        # Plot the proportion of nodes in each state at each step
        for label in labels:
            series = [count.get(label, 0) / sum(count.values()) for count in counts]
            axis.plot(x_range, series, label=label, color=self._colors[label], **kwargs)

        title = "node state proportions"
        if self.name:
            title = "{}: {}".format(self.name, title)
        axis.set_title(title)
        axis.set_xlabel("Simulation step")
        axis.set_ylabel("Proportion of nodes")
        axis.legend()
        axis.set_xlim(x_range.start)
        axis.set_ylim(0, 1)

        return axis

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


def run_animation(gamma: float = 0.05, beta: float = 0.1):
    """
    Run the simulation, updating a plot at each step.
    """
    # Pause parameter - Will be toggled by pressing the space bar (see on_keypress)
    pause = True

    # --------------------------------
    # MAIN BODY
    # --------------------------------

    facebook = pd.read_csv(
        "./data/facebook_combined.txt.gz",
        compression="gzip",
        sep=" ",
        names=["start_node", "end_node"],
    )
    G = nx.from_pandas_edgelist(facebook, "start_node", "end_node")

    # Reinitialize the simulation
    sim = Simulation(
        G, initial_state, state_transition, gamma=gamma, beta=beta, name="Fake News"
    )

    # --------------------------------
    # ANIMATION
    # --------------------------------

    fig, [ax_graph, ax_history] = plt.subplots(1, 2, figsize=(12, 6))

    # Draw the graph and the history for the first frame
    ax_graph = sim.draw(axis=ax_graph, node_size=20)
    ax_history = sim.plot(axis=ax_history, min_step=0)

    def update_frame(step: int):
        """
        Update the plot with the state of the simulation at a given step.

        Parameters
        ----------
        step : int
            The step of the simulation to plot.
        """
        nonlocal sim, G, ax_graph, ax_history
        if pause:
            return ax_graph, ax_history
        # Clear the axes - We will draw them again
        ax_graph.clear()
        ax_history.clear()
        # Run the simulation for one step
        sim._step()
        # Draw the graph and the history
        ax_graph = sim.draw(axis=ax_graph, node_size=20)
        ax_history = sim.plot(axis=ax_history, min_step=max(0, step - 100))
        return ax_graph, ax_history

    anim = FuncAnimation(fig, update_frame, interval=100, blit=True)

    # --------------------------------
    # SPACE BAR PRESS EVENT
    # --------------------------------

    # When the user presses the space bar, the simulation is paused / resumed

    def on_space(event):
        nonlocal pause
        if event.key == " ":
            pause = not pause

    fig.canvas.mpl_connect("key_press_event", on_space)

    # --------------------------------
    # ENTER PRESS EVENT
    # --------------------------------

    # When the user presses the enter key, the simulation is reset

    def on_enter(event):
        nonlocal sim, ax_graph, ax_history
        if event.key == "enter":
            sim = Simulation(
                G,
                initial_state,
                state_transition,
                gamma=gamma,
                beta=beta,
                name="Fake News",
            )
            ax_graph.clear()
            ax_history.clear()

    fig.canvas.mpl_connect("key_press_event", on_enter)

    # --------------------------------
    # SHOW
    # --------------------------------

    plt.show()


if __name__ == "__main__":
    run_animation()
