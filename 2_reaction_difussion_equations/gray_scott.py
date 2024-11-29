from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import toml
from matplotlib import animation


def initialize_grid(grid_size: int, perturb: bool = True) -> np.ndarray:
    uv = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
    uv[:, :, 0] = 1.0  # Initialize u to 1.0, v to 0.0

    if perturb:
        r = 1  # Radius of perturbation
        center = grid_size // 2
        uv[center - r : center + r, center - r : center + r, 0] = 0.50  # u
        uv[center - r : center + r, center - r : center + r, 1] = 0.25  # v

    return uv


def laplacian(uv: np.ndarray) -> np.ndarray:
    lap = -4 * uv
    lap += np.roll(uv, shift=1, axis=0)
    lap += np.roll(uv, shift=-1, axis=0)
    lap += np.roll(uv, shift=1, axis=1)
    lap += np.roll(uv, shift=-1, axis=1)
    return lap


def update(
    uv: np.ndarray,
    du: float = 0.16,
    dv: float = 0.08,
    f: float = 0.060,
    k: float = 0.062,
    dt: float = 1.0,
) -> None:
    u = uv[:, :, 0]
    v = uv[:, :, 1]

    lap = laplacian(uv)
    lu = lap[:, :, 0]
    lv = lap[:, :, 1]

    uvv = u * v * v
    du_dt = du * lu - uvv + f * (1 - u)
    dv_dt = dv * lv + uvv - (f + k) * v

    uv[:, :, 0] += du_dt * dt
    uv[:, :, 1] += dv_dt * dt

    # Enforce boundary conditions (Neumann)
    uv[0, :] = uv[1, :]
    uv[-1, :] = uv[-2, :]
    uv[:, 0] = uv[:, 1]
    uv[:, -1] = uv[:, -2]


def animate_simulation(grid_size: int, **kwargs: Any) -> None:
    uv = initialize_grid(grid_size, perturb=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(uv[:, :, 1], cmap="inferno", interpolation="bilinear")
    plt.axis("off")

    def update_frame(_):
        for _ in range(50):
            update(uv, **kwargs)
        im.set_array(uv[:, :, 1])
        return [im]

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        x = int(event.xdata)
        y = int(event.ydata)
        r = 3  # Radius of the perturbation

        # Left click?
        if event.button == 1:
            u_new = 0.50
            v_new = 0.25
        # Right click?
        elif event.button == 3:
            u_new = 1.0
            v_new = 0.0

        y_min = max(y - r, 0)
        y_max = min(y + r, grid_size)
        x_min = max(x - r, 0)
        x_max = min(x + r, grid_size)

        uv[y_min:y_max, x_min:x_max, 0] = u_new
        uv[y_min:y_max, x_min:x_max, 1] = v_new

    fig.canvas.mpl_connect("button_press_event", on_click)

    ani = animation.FuncAnimation(fig, update_frame, interval=100, blit=True)
    plt.show()


def main(config: str = "config.toml", key: str = "gray-scott"):
    dict_config: dict = toml.load(config)[key]
    animate_simulation(**dict_config)


if __name__ == "__main__":
    main()
