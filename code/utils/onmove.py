import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent


def main():
    """Example of how to handle mouse movements in a Matplotlib plot"""

    # Function to be called when the mouse moves
    def on_move(event: MouseEvent):
        if event.inaxes:  # Check if the mouse is within the plot area
            print(f"Mouse coordinates: x={event.xdata:.2f}, y={event.ydata:.2f}")

    # Create a figure and connect the event handler
    ax: Axes
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])  # Simple plot for reference
    ax.set_title("Move the mouse over the plot (see terminal for coordinates)")

    # Connect the motion_notify_event to the on_move function
    fig.canvas.mpl_connect("motion_notify_event", on_move)

    plt.show()


if __name__ == "__main__":
    main()
