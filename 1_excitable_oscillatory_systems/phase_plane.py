import matplotlib.pyplot as plt
import numpy as np
from fitzhugh_nagumo import fitzhugh_nagumo
from scipy.optimize import fsolve


def compute_fixed_points(i_ext: float) -> np.ndarray:
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
    # This is a nonlinear equation; we'll use numerical methods to find fixed points.

    def equations(y):
        return fitzhugh_nagumo(0, y, i_ext)

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


def plot_phase_plane(i_ext: float) -> None:
    """
    Plots the phase plane of the FitzHugh-Nagumo model.

    Parameters
    ----------
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
    dV = V - (V**3) / 3 - W + i_ext
    dW = 0.08 * (V + 0.7 - 0.8 * W)

    # Compute nullclines
    v_nullcline = V - (V**3) / 3 + i_ext
    w_nullcline = (V + 0.7) / 0.8

    # Plot vector field
    plt.figure(figsize=(8, 6))
    plt.quiver(V, W, dV, dW, color="gray", alpha=0.5)

    # Plot nullclines
    plt.contour(
        V,
        W,
        v_nullcline - W,
        levels=[0],
        colors="r",
        linestyles="--",
        linewidths=2,
        label="dv/dt = 0",
    )
    plt.contour(
        V,
        W,
        dW,
        levels=[0],
        colors="b",
        linestyles="-.",
        linewidths=2,
        label="dw/dt = 0",
    )

    # Compute and plot fixed points
    fixed_points = compute_fixed_points(i_ext)
    for fp in fixed_points:
        plt.plot(fp[0], fp[1], "ko", markersize=8)
        plt.text(fp[0] + 0.1, fp[1] + 0.1, f"({fp[0]:.2f}, {fp[1]:.2f})")

    plt.xlabel("Membrane Potential (v)")
    plt.ylabel("Recovery Variable (w)")
    plt.title("Phase Plane Analysis of FitzHugh-Nagumo Model")
    plt.legend(["dv/dt = 0 Nullcline", "dw/dt = 0 Nullcline", "Fixed Points"])
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid(True)
    plt.show()


def main() -> None:
    """
    Main function to perform phase plane analysis.
    """
    i_ext = 0.5  # External stimulus current
    plot_phase_plane(i_ext)


if __name__ == "__main__":
    main()
