import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve


def compute_fixed_points(equations: callable) -> np.ndarray:
    """
    Computes the fixed points of the FitzHugh-Nagumo model for a given external current.

    Parameters
    ----------
    i_ext : float
        External stimulus current.

    Returns
    -------
    fixed_points : ndarray
        Array of fixed points [v*, w*].
    """

    # Initial guesses for fixed points
    guesses = [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0)]
    fixed_points = []
    for guess in guesses:
        fixed_point, info, ier, mesg = fsolve(equations, guess, full_output=True)
        if ier == 1:
            # Check for duplicates
            if not any(np.allclose(fixed_point, fp, atol=1e-5) for fp in fixed_points):
                fixed_points.append(fixed_point)
    return np.array(fixed_points)


def plot_phase_plane(equations: callable, i_ext: float) -> None:
    """
    Plots the phase plane of any excitable-oscillatory model.

    Parameters
    ----------
    equations : callable
        Function that defines the model equations.
    i_ext : float
        External stimulus current.

    Returns
    -------
    None
    """
    # Create a grid of points
    v = np.linspace(-3, 3, 20)
    w = np.linspace(-3, 3, 20)
    V, W = np.meshgrid(v, w)

    # Compute derivatives
    dv, dw = equations(0, [V, W], i_ext)

    # Compute nullclines
    v_nullcline, _ = equations(0, [V, 0], i_ext)

    # Plot vector field
    plt.quiver(V, W, dv, dw, color="gray", alpha=0.5)

    # Plot nullclines
    plt.contour(
        V,
        W,
        v_nullcline - W,
        levels=[0],
        colors="r",
        linestyles="--",
        linewidths=2,
        label="v nullcline",
    )
    plt.contour(
        V,
        W,
        dw,
        levels=[0],
        colors="b",
        linestyles="-.",
        linewidths=2,
        label="w nullcline",
    )

    # This is a nonlinear equation; we'll use numerical methods to find fixed points.
    # Compute and plot fixed points
    fixed_points = compute_fixed_points(lambda y: equations(0, y, i_ext))
    for fp in fixed_points:
        plt.plot(fp[0], fp[1], "ko", markersize=8)
        plt.text(fp[0] + 0.1, fp[1] + 0.1, f"({fp[0]:.2f}, {fp[1]:.2f})")


def main(i_ext: float = 0.5) -> None:
    """
    Main function to perform phase plane analysis.
    """

    plt.figure(figsize=(8, 6))

    # External stimulus current
    # TODO: Define the FitzHugh-Nagumo model equations
    plot_phase_plane(equations, i_ext)

    plt.xlabel("Membrane Potential (v)")
    plt.ylabel("Recovery Variable (w)")
    plt.title("Phase Plane Analysis of FitzHugh-Nagumo Model")
    plt.legend(["v nullcline", "w nullcline", "Fixed Points"])
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
