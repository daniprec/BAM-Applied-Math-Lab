import matplotlib.pyplot as plt
import numpy as np


def coupled_oscillators(n: int = 10, k: float = 1.0, t: float = 10.0, dt: float = 0.01):
    """
    Reference: Biological rhythms and the behavior of populations of coupled oscillators
    https://www.sciencedirect.com/science/article/pii/0022519367900513

    Parameters
    ----------
    n : int, optional
        The number of oscillators, by default 10.
    k : float, optional
        The coupling strength, by default 1.0.
    t : float, optional
        The total simulation time, by default 10.0.
    dt : float, optional
        The time step size, by default 0.01.
    """
    # Parameters
    steps = int(t / dt)

    # Initial conditions
    theta = np.random.rand(n) * 2 * np.pi
    omega = np.random.randn(n)

    # Initialize array to store the phases over time
    theta_over_time = np.zeros((n, steps))

    # Simulation
    for s in range(steps):
        dtheta = omega + (k / n) * np.sum(np.sin(theta[:, None] - theta), axis=1)
        theta += dtheta * dt
        theta_over_time[:, s] = theta

    # Plotting
    plt.figure(figsize=(10, 6))
    for i in range(n):
        plt.plot(
            np.linspace(0, t, steps), theta_over_time[i, :], label=f"Oscillator {i + 1}"
        )
    plt.xlabel("Time")
    plt.ylabel("Phase")
    plt.title("Coupled Oscillators")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    coupled_oscillators()
