"""Linear stability and Hopf bifurcation of the CDIMA (Lengyel-Epstein) model.

Model (Strogatz 2024, Section 8.3; Lengyel and Epstein 1991):

    dx/dt = a - x - 4 x y / (1 + x^2)
    dy/dt = b x (1 - y / (1 + x^2))

The unique fixed point is x* = a/5, y* = 1 + (a/5)^2. The Jacobian there is

    J = 1/(1 + x*^2) [[3 x*^2 - 5, -4 x*], [2 b x*^2, -b x*]],

with det J = 5 b x* / (1 + x*^2) > 0 and tr J = (3 x*^2 - 5 - b x*)/(1 + x*^2).
The fixed point is unstable when tr J > 0, that is b < b_c = 3a/5 - 25/a.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from amlab.odes_2d.cdima import cdima


def cdima_fixed_point(a: float) -> np.ndarray:
    """Return the fixed point (a/5, 1 + (a/5)^2)."""
    x = a / 5
    return np.array([x, 1 + x**2])


def cdima_jacobian(a: float, b: float) -> np.ndarray:
    """Analytical Jacobian of the CDIMA model at its fixed point."""
    x = a / 5
    return np.array([[3 * x**2 - 5, -4 * x], [2 * b * x**2, -b * x]]) / (1 + x**2)


def hopf_b(a: float) -> float:
    """Critical value b_c = 3a/5 - 25/a where tr J = 0."""
    return 3 * a / 5 - 25 / a


def numerical_jacobian(a: float, b: float, h: float = 1e-6) -> np.ndarray:
    """Central-difference Jacobian of the CDIMA vector field at the fixed point."""
    p = cdima_fixed_point(a)
    jac = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        jac[:, j] = (cdima(0, p + e, a, b) - cdima(0, p - e, a, b)) / (2 * h)
    return jac


def locate_hopf_numerically(a: float, b_lo: float = 0.01, b_hi: float = 20.0) -> float:
    """Find b where the largest real part of the eigenvalues crosses zero.

    Uses only the numerical Jacobian, so it is independent of the formula
    in :func:`hopf_b`.
    """

    def growth(b):
        return np.max(np.linalg.eigvals(numerical_jacobian(a, b)).real)

    return brentq(growth, b_lo, b_hi)


def limit_cycle_amplitude(a: float, b: float, t_end: float = 600.0) -> float:
    """Half the peak-to-peak range of x(t) after transients have decayed."""
    p = cdima_fixed_point(a)
    sol = solve_ivp(
        cdima, (0, t_end), p + [0.01, 0.0], args=(a, b), rtol=1e-9, atol=1e-11,
        dense_output=True,
    )
    t = np.linspace(0.6 * t_end, t_end, 20000)
    x = sol.sol(t)[0]
    return 0.5 * (x.max() - x.min())


def run_parameter_map(a_range=(2.0, 20.0), b_range=(0.0, 10.0), t_end: float = 60.0):
    """Interactive map: click a point (a, b) to see its phase portrait.

    Left panel: the (a, b) plane with the Hopf curve b = 3a/5 - 25/a.
    Right panel: trajectories from a few initial conditions, nullclines and
    the fixed point for the chosen parameters.
    """
    fig, (ax_map, ax_phase) = plt.subplots(1, 2, figsize=(11, 5))
    a_vals = np.linspace(max(a_range[0], 0.1), a_range[1], 400)
    ax_map.plot(a_vals, hopf_b(a_vals), "k--", label="Hopf curve")
    ax_map.fill_between(a_vals, 0, np.clip(hopf_b(a_vals), 0, None), alpha=0.2,
                        label="limit cycle")
    ax_map.set_xlim(a_range)
    ax_map.set_ylim(b_range)
    ax_map.set_xlabel("a")
    ax_map.set_ylabel("b")
    ax_map.set_title("Click to choose (a, b)")
    ax_map.legend(loc="upper left")
    (marker,) = ax_map.plot([], [], "ro")

    def draw(a, b):
        ax_phase.clear()
        p = cdima_fixed_point(a)
        xmax, ymax = 1.6 * p[0] + 1, 1.6 * p[1] + 1
        xg = np.linspace(0.01, xmax, 300)
        ax_phase.plot(xg, (a - xg) * (1 + xg**2) / (4 * xg), "b", lw=1, label="dx/dt = 0")
        ax_phase.plot(xg, 1 + xg**2, "r", lw=1, label="dy/dt = 0")
        for x0, y0 in [(0.1, 0.1), (xmax * 0.9, 0.1), (p[0] + 0.05, p[1])]:
            sol = solve_ivp(cdima, (0, t_end), [x0, y0], args=(a, b),
                            t_eval=np.linspace(0, t_end, 3000))
            ax_phase.plot(sol.y[0], sol.y[1], lw=1)
        stable = b > hopf_b(a)
        # Filled marker: stable fixed point. Open marker: unstable.
        ax_phase.plot(*p, "o", color="k", mfc="k" if stable else "w")
        ax_phase.set_xlim(0, xmax)
        ax_phase.set_ylim(0, ymax)
        ax_phase.set_xlabel("x")
        ax_phase.set_ylabel("y")
        ax_phase.set_title(f"a = {a:.2f}, b = {b:.2f}")
        ax_phase.legend(loc="upper right")
        marker.set_data([a], [b])
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes == ax_map and event.xdata > 0 and event.ydata > 0:
            draw(event.xdata, event.ydata)

    fig.canvas.mpl_connect("button_press_event", on_click)
    draw(10.0, 3.0)
    plt.show()


if __name__ == "__main__":
    run_parameter_map()
