import sys
from typing import Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# ensure repo root on path so we can import the provided model
sys.path.append(".")
from spruce_budworm import spruce_budworm


def k_func(t, k0: float = 6.0, amp: float = 4.0, freq: float = 0.01):
    """Periodic variation of carrying capacity k(t). Accepts scalar or array-like t."""
    t_arr = np.asarray(t)
    return k0 + amp * (1 + np.sin(2 * np.pi * freq * t_arr))


def integrate_time_varying_k(
    r: float = 0.5,
    k0: float = 6.0,
    amp: float = 4.0,
    freq: float = 0.01,
    t0: float = 0.0,
    tf: float = 1200.0,
    dt: float = 1.0,
    x0: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the spruce budworm ODE while k varies in time.

    Returns t, x(t), k(t)
    """
    t_eval = np.arange(t0, tf + dt, dt)

    def f(t, x):
        return spruce_budworm(t, x, r=r, k=float(k_func(t, k0=k0, amp=amp, freq=freq)))

    sol = solve_ivp(f, (t0, tf), [x0], t_eval=t_eval, method="RK45", rtol=1e-6)
    x = sol.y[0]
    kvals = k_func(sol.t, k0=k0, amp=amp, freq=freq)
    x = np.clip(x, a_min=0.0, a_max=None)
    return sol.t, x, kvals


def find_zero_crossings(xgrid: np.ndarray, dxdt: np.ndarray) -> np.ndarray:
    """Return approximate roots of dxdt(x)=0 by linear interpolation across sign changes."""
    sign = np.sign(dxdt)
    idx = np.where(np.diff(sign) != 0)[0]
    roots = []
    for i in idx:
        xL, xR = xgrid[i], xgrid[i + 1]
        yL, yR = dxdt[i], dxdt[i + 1]
        if (yR - yL) != 0:
            xr = xL - yL * (xR - xL) / (yR - yL)
            roots.append(xr)
    return np.array(roots)


def make_animation():
    # parameters (tweak if you want faster/slower cycles)
    r = 0.5
    k0 = 0.1
    amp = 4.95
    freq = 0.01
    t0, tf, dt = 0.0, 500.0, 0.1
    x0 = (k0 + amp) / 2.0  # start near middle of carrying capacity

    t, x, kvals = integrate_time_varying_k(
        r=r, k0=k0, amp=amp, freq=freq, t0=t0, tf=tf, dt=dt, x0=x0
    )

    # x grid sized by maximum possible k
    xgrid = np.linspace(0, (k0 + amp) * 1.4 + 1.0, 1500)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    plt.tight_layout()
    ax_hyst = axs[0, 0]  # top-left: x vs k (hysteresis)
    ax_k = axs[1, 0]  # bottom-left: k(t)
    ax_x = axs[0, 1]  # top-right: x(t)
    axs[1, 1].axis("off")  # bottom-right unused

    # Top-left: hysteresis curve k -> x
    (hyst_line,) = ax_hyst.plot([], [], "-", color="tab:blue", lw=1.5)
    (hyst_point,) = ax_hyst.plot([], [], "o", color="tab:green", ms=8)
    ax_hyst.set_xlabel("k")
    ax_hyst.set_ylabel("x")
    ax_hyst.set_title("Hysteresis: x vs k")
    # set limits so x vs k plot matches the ranges of k(t) and x(t)
    kmin, kmax = kvals.min(), kvals.max()
    xmin, xmax = x.min(), x.max()
    kpad = 0.05 * (kmax - kmin) if (kmax - kmin) > 0 else 0.5
    xpad = 0.05 * (xmax - xmin) if (xmax - xmin) > 0 else 0.5
    ax_hyst.set_xlim(kmin - kpad, kmax + kpad)
    ax_hyst.set_ylim(max(0.0, xmin - xpad), xmax + xpad)
    ax_hyst.grid()

    # Bottom-left: k(t)
    ax_k.plot(kvals, t, color="gray", lw=1)
    (k_marker,) = ax_k.plot([], [], "rD")
    ax_k.set_title("k(t)")
    ax_k.set_ylim(t[0], t[-1])
    # Reuse the k limits from hysteresis plot
    ax_k.set_xlim(kmin - kpad, kmax + kpad)
    ax_k.set_ylabel("Time")
    ax_k.grid()

    # Top-right: x(t)
    ax_x.plot(t, x, color="gray", lw=1)
    (x_marker,) = ax_x.plot([], [], "gD")
    ax_x.set_title("x(t)")
    ax_x.set_xlabel("Time")
    ax_x.set_xlim(t[0], t[-1])
    ax_x.set_ylim(max(0.0, xmin - xpad), xmax + xpad)
    ax_x.grid()

    def update(frame: int):
        ti = t[frame]
        ki = kvals[frame]
        xi = x[frame]

        # update hysteresis: plot k(tau) vs x(tau) up to current frame
        hyst_line.set_data(kvals[: frame + 1], x[: frame + 1])
        hyst_point.set_data([ki], [xi])

        # update time-series markers
        k_marker.set_data([ki], [ti])
        x_marker.set_data([ti], [xi])

        return hyst_line, hyst_point, k_marker, x_marker

    ani = animation.FuncAnimation(
        fig, update, frames=len(t), interval=1, blit=True, repeat=True
    )

    plt.show()


if __name__ == "__main__":
    make_animation()
