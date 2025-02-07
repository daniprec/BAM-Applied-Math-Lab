import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent


def apply_rule(state: np.ndarray, rule_bin: np.ndarray) -> np.ndarray:
    """Apply the given rule to the current state of the automaton.

    Parameters
    ----------
    state : np.ndarray
        The current state of the automaton (a 1D numpy array of 0s and 1s).
    rule_bin : np.ndarray
        The transition rule as a binary array (a 1D numpy array of 0s and 1s).

    Returns
    -------
    np.ndarray
        The new state of the automaton after applying the rule.
    """
    # Initialize the new state as an array of zeros with the same size as the current state
    new_state = np.zeros_like(state)

    # Iterate over each cell, excluding the edges
    for i in range(1, len(state) - 1):
        # Extract the neighborhood (current cell and its two neighbors)
        neighborhood = state[i - 1 : i + 2]
        # Convert the neighborhood to an index (as a 3-bit binary number)
        index = 7 - int("".join(neighborhood.astype(str)), 2)
        # Set the new state of the cell based on the rule
        new_state[i] = rule_bin[index]

    return new_state


# Function to update the grid after a change in the first row
def update_grid(grid: np.ndarray, rule_bin: np.ndarray, ax: Axes, iterations: int = 50):
    """Recompute the grid from the initial condition after user changes.

    Parameters
    ----------
    grid : np.ndarray
        The 2D grid representing the state of the automaton over time.
    rule_bin : np.ndarray
        The transition rule as a binary array (a 1D numpy array of 0s and 1s).
    ax : Axes
        The matplotlib axes object to update with the new grid.
    iterations : int, optional
        The number of iterations to run the automaton, by default 50.
    """
    # Recompute the grid for each time step
    for t in range(1, iterations):
        grid[t] = apply_rule(state=grid[t - 1], rule_bin=rule_bin)
    # Update the plot with the new grid
    ax.imshow(grid, cmap="binary", interpolation="nearest")
    plt.draw()


def main(rule_number: int = 30, grid_size: int = 101, iterations: int = 50):
    """Run the cellular automaton simulation with the given parameters.

    Parameters
    ----------
    rule_number : int, optional
        The rule number to use for the cellular automaton, by default 30.
    grid_size : int, optional
        The size of the grid (odd number for symmetry), by default 101.
    iterations : int, optional
        The number of iterations to run the automaton, by default 50.
    """
    # Rule transition table (as a binary array)
    rule_bin = np.array([int(x) for x in f"{rule_number:08b}"])

    # Initialize the grid
    grid = np.zeros((iterations, grid_size), dtype=int)

    # Set the initial condition (all 0s except for a single 1 in the middle)
    grid[0, grid_size // 2] = 1

    # Plot the initial grid
    ax: Axes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(grid, cmap="binary", interpolation="nearest")
    ax.set_title("Cellular Automaton - Click to Change Initial Row")
    ax.set_xticks([])
    ax.set_yticks([])

    # Mouse click event handler
    def on_click(event: MouseEvent):
        """Handle clicks on the first row to toggle cell values between 0 and 1."""
        # Get the x and y coordinates of the click
        ix, iy = int(event.xdata), int(event.ydata)

        # Check if the click was on the first row (time = 0)
        if iy == 0:
            # Toggle the value at the clicked cell (0 <-> 1)
            grid[0, ix] = 1 - grid[0, ix]

            # Recompute and update the grid
            update_grid(grid=grid, ax=ax, rule_bin=rule_bin, iterations=iterations)

    # Connect the mouse click event to the on_click function
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
