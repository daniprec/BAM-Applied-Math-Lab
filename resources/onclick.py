import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent


def main():
    """Example of how to handle mouse clicks in a Matplotlib plot"""

    # Function to handle mouse clicks
    def onclick(event: MouseEvent):
        # Get the x and y coordinates of the click
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            # Place a point at the clicked location
            plt.scatter(x, y, color="red")
            plt.draw()

    # Create a figure and axis
    ax: Axes
    fig, ax = plt.subplots()

    # Connect the click event to the handler function
    fig.canvas.mpl_connect("button_press_event", onclick)

    # Show an empty plot
    ax.set_title("Click to add points")
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()


if __name__ == "__main__":
    main()
