"""Interactive animation: amoebae relay a cAMP signal and walk toward it.

Run from the repository root:

    python amlab/collective_motion/dicty_animation.py
    python amlab/collective_motion/dicty_animation.py --theme light --density 0.2
    python amlab/collective_motion/dicty_animation.py --save dicty.gif --frames 400
    python amlab/collective_motion/dicty_animation.py --save dicty.png --frames 900     # one picture

What you see
------------
* The colour map is the cAMP field u. Dark = rest, bright = a wave of cAMP. After a
  wave has passed, the medium is refractory for a while, so waves do not run back.
* Cells are dots. A **small round dot** is moving at random. A **larger triangle with a
  white edge** is a cell that has just sensed a rising front and is walking toward it.
  Triangles and dots differ in shape and size, not only in colour.
* The panel on the right counts the share of cells in crowded blocks, the number of
  crowded clusters and the excited area, and plots the first two over time.

Controls
--------
Sliders: cell density (restarts the run), excitation threshold theta (live; higher =
harder to excite) and chemotactic speed (live; 0 = the cells ignore the signal).
Buttons: New run, Pause. Keys: space pauses, N starts a new run.

The animation redraws only the moving parts (blitting). Use ``--speed 2`` on a slow computer.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

sys.path.append(".")  # run from the repository root, like the other lab scripts

import matplotlib


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theme", choices=["dark", "light"], default="dark")
    ap.add_argument("--save", metavar="FILE", help="write a .gif, .mp4 or a single .png instead of opening a window")
    ap.add_argument("--frames", type=int, default=400, help="frames to render with --save")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--speed", type=int, default=3, help="simulation steps per frame (1 step = 0.2 time units)")
    ap.add_argument("--density", type=float, default=0.30, help="cells per grid site")
    ap.add_argument("--theta", type=float, default=0.15, help="excitation threshold")
    ap.add_argument("--chem", type=float, default=0.30, help="chemotactic speed (0 = control)")
    ap.add_argument("--no-chemotaxis", action="store_true", help="control run: cells ignore the signal")
    return ap.parse_args(argv)


if __name__ == "__main__":
    _args = _parse_args()
    if _args.save:
        matplotlib.use("Agg")
else:
    _args = None

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, Slider

from amlab.collective_motion.dicty import DictyParams, DictySimulation
from amlab.collective_motion.ie_style import apply_theme, style_slider

WINDOW = 600.0  # the time series show the last WINDOW time units


def field_cmap(mode: str) -> LinearSegmentedColormap:
    """Rest to excited. Dark: Tech black -> electric blue -> sea blue -> white."""
    if mode == "dark":
        colors = ["#08083F", "#0000DB", "#47BFFF", "#FFFFFF"]
    else:
        colors = ["#FFFFFF", "#B8DFFA", "#47BFFF", "#000066"]
    return LinearSegmentedColormap.from_list("ie_camp", colors)


class DictyView:
    """Matplotlib scene for a :class:`DictySimulation`."""

    def __init__(self, params: DictyParams, seed: int = 3, theme: str = "dark", steps_per_frame: int = 3):
        self.theme = theme
        self.P = apply_theme(theme)
        self.params = params
        self.seed = seed
        self.steps_per_frame = steps_per_frame
        self.paused = False
        self._build_figure()
        self.rebuild()

    # ------------------------------------------------------------------ layout
    def _build_figure(self) -> None:
        P = self.P
        n = self.params.n
        self.fig = plt.figure(figsize=(12, 6.6), facecolor=P["bg"])
        self.ax = self.fig.add_axes([0.02, 0.08, 0.57, 0.88])
        ax = self.ax
        ax.set(xlim=(0, n), ylim=(0, n))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_color(P["grid"])
        edges = np.arange(n + 1)
        self.field = ax.pcolormesh(
            edges, edges, np.zeros((n, n)), cmap=field_cmap(self.theme), vmin=0, vmax=1,
            shading="flat", antialiased=False, linewidth=0, zorder=2, animated=True,
        )  # fmt: skip
        light = self.theme == "light"
        self.c_rest = ax.scatter([], [], s=5, marker="o", color=P["bad"] if light else "#FF8A91", linewidths=0, zorder=4, animated=True)
        self.c_move = ax.scatter(
            [], [], s=26, marker="^", facecolors=P["good"], edgecolors=P["predator_edge"], linewidths=0.7, zorder=5, animated=True
        )
        self.banner = ax.text(
            0.02, 0.97, "", transform=ax.transAxes, ha="left", va="top", color=P["text"] if not light else P["accent"],
            fontsize=11, fontweight="bold", zorder=9, animated=True,
        )  # fmt: skip
        self.fig.text(
            0.02, 0.03, "dot: cell at rest      triangle: cell walking toward a wave front      blue band: cAMP wave",
            color=P["text"], fontsize=9,
        )

        self.ax_hud = self.fig.add_axes([0.63, 0.70, 0.35, 0.25], facecolor=P["panel"])
        self.ax_hud.set_xticks([])
        self.ax_hud.set_yticks([])
        self.ax_hud.grid(False)
        for s in self.ax_hud.spines.values():
            s.set_color(P["grid"])
        rows = [
            ("time", "time"),
            ("cells", "cells"),
            ("share", "share of cells in crowded blocks"),
            ("clusters", "crowded clusters"),
            ("wave", "area excited (u > 0.5)"),
        ]
        self.hud_vals = {}
        y = 0.86
        for key, label in rows:
            self.ax_hud.text(0.04, y, label, color=P["muted"], fontsize=10, va="center")
            self.hud_vals[key] = self.ax_hud.text(
                0.96, y, "", color=P["text"], fontsize=11, fontweight="bold", va="center", ha="right", animated=True
            )
            y -= 0.18

        self.ax_ts = self.fig.add_axes([0.68, 0.43, 0.30, 0.18])
        self.ax_ts.set(ylim=(0, 1.02), xlim=(-WINDOW, 0))
        self.ax_ts.set_ylabel("share", fontsize=9)
        self.ax_ts.set_xlabel("time before now", fontsize=9)
        self.ax_ts.tick_params(labelsize=8)
        (self.line_share,) = self.ax_ts.plot([], [], color=P["accent"], lw=2.2, animated=True, label="share aggregated")
        (self.line_wave,) = self.ax_ts.plot([], [], color=P["predator"], lw=1.6, ls="--", animated=True, label="area excited (x4)")
        self.ax_ts.set_title("Aggregation and waves", fontsize=10, loc="left")
        self.ax_ts.legend(loc="upper left", fontsize=7)

        def mk(rect, label, lo, hi, init, step, fmt):
            axs = self.fig.add_axes(rect, facecolor=P["panel"])
            sl = Slider(axs, label, lo, hi, valinit=init, valstep=step, valfmt=fmt)
            style_slider(sl, P)
            return sl

        p = self.params
        x0, w = 0.80, 0.17
        self.s_dens = mk([x0, 0.25, w, 0.022], "Cell density  ", 0.05, 0.60, p.density, 0.05, "%.2f")
        self.s_theta = mk([x0, 0.21, w, 0.022], "Threshold theta  ", 0.05, 0.40, p.theta, 0.01, "%.2f")
        self.s_chem = mk([x0, 0.17, w, 0.022], "Chemotaxis  ", 0.0, 0.6, p.v_chem if p.chemotaxis else 0.0, 0.05, "%.2f")

        def btn(rect, text):
            b = Button(self.fig.add_axes(rect), text, color=P["slider_bg"], hovercolor=P["accent"])
            b.label.set_color(P["text"])
            b.label.set_fontsize(10)
            return b

        self.b_new = btn([0.67, 0.04, 0.12, 0.05], "New run")
        self.b_pause = btn([0.81, 0.04, 0.12, 0.05], "Pause")
        self.s_theta.on_changed(self._on_live)
        self.s_chem.on_changed(self._on_live)
        self.s_dens.on_changed(lambda _v: self._restart())
        self.b_new.on_clicked(lambda _e: self._restart(bump=True))
        self.b_pause.on_clicked(lambda _e: self._toggle_pause())
        self.fig.canvas.mpl_connect(
            "key_press_event",
            lambda e: self._toggle_pause() if e.key == " " else self._restart(bump=True) if e.key == "n" else None,
        )

    def animated_artists(self):
        return [self.field, self.c_rest, self.c_move, self.banner, self.line_share, self.line_wave, *self.hud_vals.values()]

    # ----------------------------------------------------------------- control
    def _on_live(self, _v) -> None:
        chem = float(self.s_chem.val)
        self.params = replace(self.params, theta=float(self.s_theta.val), v_chem=max(chem, 1e-9), chemotaxis=chem > 0)
        self.sim.p = self.params

    def _restart(self, bump: bool = False) -> None:
        if bump:
            self.seed += 1
        chem = float(self.s_chem.val)
        self.params = replace(
            self.params, density=float(self.s_dens.val), theta=float(self.s_theta.val),
            v_chem=max(chem, 1e-9), chemotaxis=chem > 0,
        )  # fmt: skip
        self.rebuild()

    def _toggle_pause(self) -> None:
        self.paused = not self.paused

    def rebuild(self) -> None:
        self.sim = DictySimulation(self.params, seed=self.seed)
        self.t_hist: list[float] = []
        self.share_hist: list[float] = []
        self.wave_hist: list[float] = []
        self.metrics = self.sim.aggregation()
        self.draw()

    # ---------------------------------------------------------------- drawing
    def draw(self) -> None:
        sim = self.sim
        self.field.set_array(sim.u.T.ravel())
        moving = sim.active if sim.p.chemotaxis else np.zeros(len(sim.x), bool)
        xy = np.column_stack([sim.x, sim.y])
        self.c_rest.set_offsets(xy[~moving])
        self.c_move.set_offsets(xy[moving])
        if sim.steps % 10 == 0 or not self.t_hist:
            self.metrics = sim.aggregation()
        m = self.metrics
        wave = sim.wave_fraction()
        if not self.paused and (not self.t_hist or sim.time - self.t_hist[-1] >= 2.0):
            self.t_hist.append(sim.time)
            self.share_hist.append(m["share"])
            self.wave_hist.append(wave)
        if self.t_hist:
            t = np.array(self.t_hist) - sim.time
            keep = t > -WINDOW
            self.line_share.set_data(t[keep], np.array(self.share_hist)[keep])
            self.line_wave.set_data(t[keep], np.minimum(np.array(self.wave_hist)[keep] * 4.0, 1.0))
        h = self.hud_vals
        h["time"].set_text(f"{sim.time:7.1f}")
        h["cells"].set_text(f"{len(sim.x)}")
        h["share"].set_text(f"{100 * m['share']:.0f} %")
        h["clusters"].set_text(f"{m['clusters']:.0f}")
        h["wave"].set_text(f"{100 * wave:.0f} %")
        self.banner.set_text("PAUSED" if self.paused else ("control: cells ignore cAMP" if not sim.p.chemotaxis else ""))

    # --------------------------------------------------------------- animation
    def update(self, _frame):
        if not self.paused:
            for _ in range(self.steps_per_frame):
                self.sim.step()
        self.draw()
        return self.animated_artists()


def main(args=None) -> None:
    args = args if args is not None else _parse_args()
    params = DictyParams(
        density=args.density, theta=args.theta, v_chem=max(args.chem, 1e-9),
        chemotaxis=not args.no_chemotaxis and args.chem > 0,
    )  # fmt: skip
    view = DictyView(params, seed=args.seed, theme=args.theme, steps_per_frame=args.speed)
    if args.save and args.save.endswith(".png"):
        for f in range(args.frames):
            view.update(f)
        view.draw()
        view.fig.savefig(args.save, dpi=100)
        print(f"saved {args.save}")
        return
    ani = animation.FuncAnimation(
        view.fig, view.update, frames=args.frames if args.save else None,
        interval=33, blit=not args.save, cache_frame_data=False,
    )  # fmt: skip
    if args.save:
        writer = animation.PillowWriter(fps=24) if args.save.endswith(".gif") else animation.FFMpegWriter(fps=30, bitrate=4000)
        ani.save(args.save, writer=writer, dpi=90)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main(_args)
