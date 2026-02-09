import matplotlib.pyplot as plt
import numpy as np


def lorenz(xyz, *, s=10, r=28, b=2.667):
    """Compute the partial derivatives of the Lorenz attractor at a given point.

    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])


def main(dt: float = 0.01, num_steps: int = 10000):
    """Plot the Lorenz attractor using a simple numerical integration scheme.

    Parameters
    ----------
    dt : float, optional
       Time step for the numerical integration, by default 0.01.
    num_steps : int, optional
         Number of time steps to integrate, by default 10000.
    """
    # Set up the array of initial values
    xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
    xyzs[0] = (0.0, 1.0, 1.05)  # Set initial values

    # Euler's method
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point

    for i in range(num_steps):
        xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt

    # Plot
    ax = plt.figure().add_subplot(projection="3d")

    ax.plot(*xyzs.T, lw=0.6)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    plt.show()


if __name__ == "__main__":
    main()
