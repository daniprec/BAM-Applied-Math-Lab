import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def gierer_meinhardt_ode(
    t: float,
    uv: np.ndarray,
    a: float = 0.40,
    b: float = 1.00,
    gamma: float = 1,
    d: float = 30,
    dx: float = 1,
) -> np.ndarray:
    """ODEs for the Gierer-Meinhardt model in 1D

    Parameters
    ----------
    t : float
        Time variable (not used in the function)
    uv : np.ndarray
        Array containing the concentrations of u and v
    a : float
        Reaction rate parameter, by default 0.40
    b : float
        Reaction rate parameter, by default 1.00
    gamma : float
        Reaction rate parameter, by default 1
    d : float
        Diffusion rate parameter of v, by default 30
    dx : float
        Spatial step size, by default 1

    Returns
    -------
    np.ndarray
        Array containing the time derivatives of u and v
    """
    # Compute the Laplacian via finite differences
    lap = -2 * uv
    lap += np.roll(uv, shift=1, axis=1)  # left
    lap += np.roll(uv, shift=-1, axis=1)  # right
    lap /= dx**2

    # Extract the matrices for u and v
    u, v = uv
    lu, lv = lap

    # Compute the ODEs
    f = a - b * u + u**2 / (v)
    g = u**2 - v
    du_dt = lu + gamma * f  # d1 = 1
    dv_dt = d * lv + gamma * g  # d2 = d
    return np.array([du_dt, dv_dt])


def animate_simulation(
    dt: float = 0.01,
    dx: float = 1,
    anim_speed: int = 10,
    length: int = 40,
    seed: int = 0,
    boundary_conditions: str = "neumann",
):
    """Animate the simulation of the Gierer-Meinhardt model in 1D

    Parameters
    ----------
    dt : float, optional
        Time step size, by default 0.01
    dx : float, optional
        Spatial step size, by default 1
    anim_speed : int, optional
        Number of iterations per frame, by default 10
    length : int, optional
        Length of the 1D domain, by default 40
    seed : int, optional
        Random seed for reproducibility, by default 0
    boundary_conditions : str, optional
        Boundary conditions to apply, by default "neumann"
    """
    # Fix the random seed for reproducibility
    np.random.seed(seed)

    # Initialize the u and v fields
    uv = 2 - np.random.uniform(0, 0.2, (2, int(length / dx))).astype(np.float64)
    # Initialize the x-axis
    x = np.linspace(0, length, int(length / dx))

    # Initialize the figure
    fig, ax = plt.subplots()
    ax.set_xlim(0, length)
    ax.set_ylim(1.4, 3)
    (line_v,) = ax.plot(x, uv[1])
    ax.set_xlabel("x")
    ax.set_ylabel("v(x)")

    def update(frame: int):
        # Access the uv variable from the outer scope
        nonlocal uv

        # Iterate the simulation as many times as the animation speed
        for _ in range(anim_speed):
            uv = uv + gierer_meinhardt_ode(0, uv, dx=dx) * dt
            # Apply boundary conditions
            if boundary_conditions == "neumann":
                # Neumann - zero flux boundary conditions
                uv[:, 0] = uv[:, 1]
                uv[:, -1] = uv[:, -2]
            elif boundary_conditions == "periodic":
                # Periodic conditions are already implemented in the laplacian function
                pass
            else:
                raise ValueError(
                    "Invalid boundary_conditions value. Use 'neumann' or 'periodic'."
                )
        # Update the plot
        line_v.set_ydata(uv[1])
        return [line_v]

    ani = animation.FuncAnimation(fig, update, interval=1, blit=True)
    plt.show()


if __name__ == "__main__":
    animate_simulation()
2
