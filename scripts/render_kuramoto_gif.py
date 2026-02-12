"""Render a Kuramoto unit-circle animation as a GIF.

Usage (from repo root):
  python scripts/render_kuramoto_gif.py

This writes:
  img/kuramoto-animation.gif

This GIF is used in the Kuramoto oscillators module,
and shows the oscillators moving around the unit circle as they synchronize.
Each blue dot represents an oscillator with phase $\theta_i$.
The red line and point represent the order parameter $r e^{i\phi}$,
which indicates the degree of synchronization among the oscillators.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amlab.odes_coupled.kuramoto import (
    initialize_oscillators,
    kuramoto_ode_meanfield,
    kuramoto_order_parameter,
)

matplotlib.use("Agg")


def render_gif(
    out_path: Path,
    *,
    seed: int = 1,
    num_oscillators: int = 100,
    coupling_strength: float = 2.0,
    scale_omega: float = 0.5,
    scale_phase: float = 1.0,
    dt: float = 0.05,
    frames: int = 200,
    fps: int = 20,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    theta, omega = initialize_oscillators(
        num_oscillators,
        distribution="normal",
        scale_omega=scale_omega,
        scale_phase=scale_phase,
        seed=seed,
    )

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_title(f"Kuramoto oscillators (K={coupling_strength:.1f})")
    ax.set_xlabel("cos(θ)")
    ax.set_ylabel("sin(θ)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.grid(True)

    circle = plt.Circle((0, 0), 1, color="lightgray", fill=False)
    ax.add_artist(circle)

    scatter = ax.scatter([], [], s=30, color="tab:blue", alpha=0.35)
    (centroid_line,) = ax.plot([], [], color="tab:red", linewidth=2)
    (centroid_point,) = ax.plot([], [], "o", color="tab:red", markersize=5)

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        centroid_line.set_data([], [])
        centroid_point.set_data([], [])
        return scatter, centroid_line, centroid_point

    def update(_frame: int):
        nonlocal theta

        sol = solve_ivp(
            kuramoto_ode_meanfield,
            (0, dt),
            theta,
            args=(omega, coupling_strength),
            rtol=1e-6,
            atol=1e-9,
        )
        theta = np.mod(sol.y[:, -1], 2 * np.pi)

        x = np.cos(theta)
        y = np.sin(theta)
        scatter.set_offsets(np.column_stack([x, y]))

        r, _phi, rcosphi, rsinphi = kuramoto_order_parameter(theta)
        centroid_line.set_data([0, rcosphi], [0, rsinphi])
        centroid_point.set_data([rcosphi], [rsinphi])

        return scatter, centroid_line, centroid_point

    ani = FuncAnimation(
        fig, update, init_func=init, frames=frames, blit=True, interval=1000 / fps
    )

    writer = PillowWriter(fps=fps)
    ani.save(out_path, writer=writer)

    plt.close(fig)
    return out_path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "img" / "kuramoto-animation.gif"
    render_gif(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
