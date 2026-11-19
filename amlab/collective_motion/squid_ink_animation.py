"""Interactive animation: a school of fish hunts a squid that releases ink.

Run from the repository root:

    python amlab/collective_motion/squid_ink_animation.py
    python amlab/collective_motion/squid_ink_animation.py --theme light
    python amlab/collective_motion/squid_ink_animation.py --save squid-ink.gif --frames 400

What you see
------------
* Camera: follows the squid (the animal with eight arms and two long tentacles) by
  default. "Camera" (or C) switches to a fixed frame on the whole 40 x 40 body-length sea.
* The squid turns from green to grey as its ink sac empties. "Recharge ink" (or R)
  refills it to 100 %. The "ink supply" row shows how much is left.
* The ink cloud (blue-violet) spreads, thins out and vanishes.
* The school: red arrows. A **filled** arrow can see the squid, a **hollow** arrow
  has lost it behind the ink (transmission below 0.5).
* A white ring around the squid means that it has just inked and must wait.
* Hunts restart automatically; captures and escapes are tallied on the right.

Controls: sliders for the ink strength, the share of ink used per
release and the school size; "Ink on/off", "Camera", "New hunt", "Pause". Keys: space
pauses, S switches the fish search (go to the last place the squid was seen) on or off,
D switches the decoy tactic (chase the densest ink blob) on or off.
Command-line options ``--jet 2`` and ``--escape-angle 90`` give the squid a decision after each release.

The animation uses blitting: only the moving parts are redrawn, which keeps it
smooth. Use ``--speed 1`` on a slow computer.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

sys.path.append(".")  # run from the repository root, like the other lab scripts

import matplotlib


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--theme", choices=["dark", "light"], default="dark")
    ap.add_argument(
        "--save", metavar="FILE", help="write a .gif or .mp4 instead of opening a window"
    )
    ap.add_argument("--frames", type=int, default=400, help="frames to render with --save")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--speed", type=int, default=2, help="simulation steps per frame")
    ap.add_argument("--fish", type=int, default=24, help="number of fish")
    ap.add_argument("--kappa", type=float, default=2.8, help="ink strength (opacity of the ink)")
    ap.add_argument("--reload", type=float, default=5.0, help="minimum time between releases")
    ap.add_argument("--ink-use", type=float, default=0.6, help="share of the remaining ink used per release (0 = unlimited)")
    ap.add_argument("--fixed", action="store_true", help="start with a fixed camera on the whole sea")
    ap.add_argument("--blind", action="store_true", help="fish do not search the last-seen place (keys: S)")
    ap.add_argument("--decoy", action="store_true", help="fish chase the densest ink blob (keys: D)")
    ap.add_argument("--no-ink", action="store_true", help="squid without ink (control)")
    ap.add_argument("--jet", type=float, default=1.0, help="speed factor of the escape jet after a release (1 = no jet, 2 = twice as fast)")
    ap.add_argument("--escape-angle", type=float, default=0.0, help="degrees the squid leaves the flight line after inking (0 = straight away)")
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
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.widgets import Button, Slider

from amlab.collective_motion.ie_style import apply_theme, style_slider
from amlab.collective_motion.squid_ink import (
    InkParams,
    SquidInkSimulation,
    wrap_delta,
)

T_MAX = 250.0  # a hunt ends after this many time units: the squid has escaped
SEE = 0.5  # transmission above which a fish is drawn filled
PAUSE_AFTER_END = 36  # frames to show the outcome before the next hunt
SPENT = "#8A8FA8"  # colour of the squid when its ink sac is empty (green when full)
VIEW = 15.0  # half-width of the camera window in body lengths
WINDOW = 60.0  # the time series show the last WINDOW time units
SQUID_SIZE = 3.0  # mantle length in body lengths


def ink_cmap(mode: str) -> LinearSegmentedColormap:
    """Transparent water to dense ink."""
    if mode == "dark":
        colors = [(0.16, 0.16, 1.0, 0.0), "#3A3AFF", "#8E7BFF", "#F0EBFF"]
    else:
        colors = [(1, 1, 1, 0.0), "#9C9CF0", "#3B2BC4", "#12096E"]
    return LinearSegmentedColormap.from_list("ie_ink", colors)


# --- squid outline in local coordinates: x forward (mantle tip), y to the left ---
MANTLE = np.array(
    [(1.0, 0.0), (0.82, 0.10), (0.38, 0.17), (0.0, 0.17), (-0.14, 0.10),
     (-0.14, -0.10), (0.0, -0.17), (0.38, -0.17), (0.82, -0.10)]
)  # fmt: skip
FIN_L = np.array([(0.98, 0.0), (0.74, 0.30), (0.50, 0.32), (0.38, 0.17)])
FIN_R = FIN_L * np.array([1.0, -1.0])
_ARM_Y = np.linspace(-0.09, 0.09, 8)
ARMS = [np.array([(-0.14, y), (-0.45, 1.6 * y), (-0.78, 2.4 * y)]) for y in _ARM_Y]
TENTACLES = [
    np.array([(-0.14, s * 0.04), (-0.65, s * 0.10), (-1.2, s * 0.20)]) for s in (1, -1)
]
CLUBS = [np.array([(-1.2, s * 0.20), (-1.32, s * 0.24)]) for s in (1, -1)]


def _rot(poly: np.ndarray, theta: float, size: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return size * poly @ np.array([[c, s], [-s, c]])


def squid_shapes(theta: float, size: float):
    """Vertices of the squid body, fins and the arm lines, rotated to ``theta``."""
    body = [_rot(MANTLE, theta, size), _rot(FIN_L, theta, size), _rot(FIN_R, theta, size)]
    arms = [_rot(a, theta, size) for a in ARMS]
    tent = [_rot(t, theta, size) for t in TENTACLES]
    clubs = [_rot(c, theta, size) for c in CLUBS]
    return body, arms, tent, clubs


class InkView:
    """Matplotlib scene for a :class:`SquidInkSimulation`."""

    def __init__(self, params: InkParams, seed: int = 4, theme: str = "dark", steps_per_frame: int = 2):
        self.theme = theme
        self.P = apply_theme(theme)
        self.params = params
        self.seed = seed
        self.steps_per_frame = steps_per_frame
        self.paused = False
        self.follow = True
        self.tally = {"captured": 0, "escaped": 0}
        self._build_figure()
        self.rebuild()

    # ------------------------------------------------------------------ layout
    def _build_figure(self) -> None:
        P = self.P
        self.fig = plt.figure(figsize=(12, 6.6), facecolor=P["bg"])
        self.ax = self.fig.add_axes([0.02, 0.05, 0.57, 0.90])
        ax = self.ax
        d, g = self.params.domain, self.params.grid
        ax.set_facecolor(P["floor"])
        ax.set(xlim=(-VIEW, VIEW), ylim=(-VIEW, VIEW))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_color(P["grid"])

        # The ink is a mesh of coloured cells, one per grid cell inside the camera window.
        # (A mesh is about ten times faster to redraw than a resampled image.)
        self.half = int(np.ceil(VIEW / (d / g))) + 1  # cells on each side of the squid
        edges = np.linspace(-self.half, self.half, 2 * self.half + 1) * (d / g)
        self.field_fixed = ax.pcolormesh(
            np.linspace(0, d, g + 1), np.linspace(0, d, g + 1), np.zeros((g, g)), cmap=ink_cmap(self.theme),
            vmin=0, vmax=1.2, shading="flat", antialiased=False, linewidth=0, zorder=2, animated=True,
        )  # fmt: skip
        self.field_fixed.set_visible(False)
        self.field = ax.pcolormesh(
            edges, edges, np.zeros((2 * self.half, 2 * self.half)), cmap=ink_cmap(self.theme),
            vmin=0, vmax=1.2, shading="flat", antialiased=False, linewidth=0, zorder=2, animated=True,
        )  # fmt: skip
        # world-fixed dots give a sense of motion while the camera follows the squid
        grid = np.arange(0.0, d, 4.0)
        self.dots_world = np.array([(x, y) for x in grid for y in grid])
        self.dots = ax.scatter([], [], s=5, color=P["grid"], zorder=1, linewidths=0, animated=True)
        self.fish = PolyCollection([], zorder=5, linewidths=1.1, animated=True)
        ax.add_collection(self.fish)
        # the squid: green body and fins, eight arms and two tentacles, white outline
        self.s_body = PolyCollection(
            [], zorder=7, facecolors=P["odd"], edgecolors=P["predator_edge"], linewidths=1.2, animated=True
        )
        self.s_arms = LineCollection([], zorder=6, colors=P["odd"], linewidths=1.8, animated=True)
        self.s_tent = LineCollection([], zorder=6, colors=P["odd"], linewidths=2.4, animated=True)
        self.s_club = LineCollection([], zorder=6, colors=P["predator_edge"], linewidths=3.2, animated=True)
        for c in (self.s_body, self.s_arms, self.s_tent, self.s_club):
            ax.add_collection(c)
        self.ring = ax.scatter(
            [], [], s=900, facecolors="none", edgecolors=P["text"], linewidths=1.6, zorder=6, animated=True
        )
        self.banner = ax.text(
            0.5, 0.96, "", transform=ax.transAxes, ha="center", va="top", color=P["text"],
            fontsize=13, fontweight="bold", zorder=9, animated=True,
        )  # fmt: skip

        # right-hand side: scoreboard and two time series
        self.ax_hud = self.fig.add_axes([0.63, 0.66, 0.35, 0.29], facecolor=P["panel"])
        self.ax_hud.set_xticks([])
        self.ax_hud.set_yticks([])
        self.ax_hud.grid(False)
        for s in self.ax_hud.spines.values():
            s.set_color(P["grid"])
        rows = [
            ("time", "time"),
            ("closest", "closest fish (BL)"),
            ("vis", "mean visibility of the squid"),
            ("rel", "ink releases"),
            ("gland", "ink supply"),
            ("tally", "captured / escaped"),
        ]
        self.hud_vals = {}
        y = 0.88
        for key, label in rows:
            self.ax_hud.text(0.04, y, label, color=P["muted"], fontsize=10, va="center")
            self.hud_vals[key] = self.ax_hud.text(
                0.96, y, "", color=P["text"], fontsize=11, fontweight="bold",
                va="center", ha="right", animated=True,
            )  # fmt: skip
            y -= 0.15

        # time axes are relative to "now", so their limits never change (needed for blitting)
        self.ax_vis = self.fig.add_axes([0.68, 0.47, 0.30, 0.12])
        self.ax_vis.set(ylim=(0, 1.02), xlim=(-WINDOW, 0))
        self.ax_vis.set_ylabel("visibility", fontsize=9)
        self.ax_vis.tick_params(labelsize=8)
        (self.line_vis,) = self.ax_vis.plot([], [], color=P["accent"], lw=2.2, animated=True)
        self.ax_vis.set_title("Mean visibility of the squid", fontsize=10, loc="left")

        self.ax_dist = self.fig.add_axes([0.68, 0.315, 0.30, 0.10])
        self.ax_dist.set(ylim=(0, 30), xlim=(-WINDOW, 0))
        self.ax_dist.set_ylabel("BL", fontsize=9)
        self.ax_dist.set_xlabel("time before now", fontsize=9)
        self.ax_dist.tick_params(labelsize=8)
        (self.line_dist,) = self.ax_dist.plot([], [], color=P["predator"], lw=2.2, animated=True)
        self.ax_dist.axhline(self.params.capture_radius, color=P["muted"], lw=1, ls="--")
        self.ax_dist.set_title("Distance of the closest fish", fontsize=10, loc="left")

        def mk(rect, label, lo, hi, init, step, fmt):
            axs = self.fig.add_axes(rect, facecolor=P["panel"])
            sl = Slider(axs, label, lo, hi, valinit=init, valstep=step, valfmt=fmt)
            style_slider(sl, P)
            return sl

        x0, w = 0.80, 0.17
        p = self.params
        self.s_kappa = mk([x0, 0.205, w, 0.022], "Ink strength  ", 0.0, 8.0, p.kappa, 0.2, "%.1f")
        self.s_use = mk([x0, 0.170, w, 0.022], "Ink used/release  ", 0.0, 0.9, p.use_fraction, 0.1, "%.1f")
        self.s_n = mk([x0, 0.120, w, 0.022], "School size  ", 8, 60, p.n_fish, 4, "%d")

        def btn(rect, text):
            b = Button(self.fig.add_axes(rect), text, color=P["slider_bg"], hovercolor=P["accent"])
            b.label.set_color(P["text"])
            b.label.set_fontsize(10)
            return b

        self.b_ink = btn([0.63, 0.02, 0.085, 0.045], "Ink on/off")
        self.b_cam = btn([0.725, 0.02, 0.085, 0.045], "Camera")
        self.b_new = btn([0.82, 0.02, 0.075, 0.045], "New hunt")
        self.b_pause = btn([0.905, 0.02, 0.075, 0.045], "Pause")
        self.b_cam.on_clicked(lambda _e: self._toggle_camera())
        self.b_fill = btn([0.63, 0.075, 0.18, 0.04], "Recharge ink to 100 %  (R)")
        self.b_fill.on_clicked(lambda _e: self._recharge())
        for s in (self.s_kappa, self.s_use):
            s.on_changed(self._on_live)
        self.s_n.on_changed(lambda _v: self._restart())
        self.b_ink.on_clicked(lambda _e: self._toggle_ink())
        self.b_new.on_clicked(lambda _e: self._restart(bump=True))
        self.b_pause.on_clicked(lambda _e: self._toggle_pause())
        self.fig.canvas.mpl_connect(
            "key_press_event",
            lambda e: self._toggle_pause() if e.key == " " else self._toggle_camera() if e.key == "c" else self._recharge() if e.key == "r" else self._flip("search") if e.key == "s" else self._flip("decoy") if e.key == "d" else None,
        )

    def animated_artists(self):
        return [
            self.field, self.field_fixed, self.dots, self.fish, self.s_body, self.s_arms, self.s_tent,
            self.s_club, self.ring, self.banner, self.line_vis, self.line_dist,
            *self.hud_vals.values(),
        ]  # fmt: skip

    # ----------------------------------------------------------------- control
    def _on_live(self, _v) -> None:
        self.params = replace(
            self.params,
            kappa=float(self.s_kappa.val),
            use_fraction=float(self.s_use.val),
        )
        self.sim.p = self.params

    def _flip(self, name: str) -> None:
        """Switch a fish tactic on or off while the hunt runs ('search' or 'decoy')."""
        self.params = replace(self.params, **{name: not getattr(self.params, name)})
        self.sim.p = self.params
        print(f"{name}: {'on' if getattr(self.params, name) else 'off'}")

    def _toggle_ink(self) -> None:
        self.params = replace(self.params, ink=not self.params.ink)
        self.sim.p = self.params

    def _restart(self, bump: bool = False) -> None:
        if bump:
            self.seed += 1
        self.params = replace(
            self.params,
            n_fish=int(self.s_n.val),
            kappa=float(self.s_kappa.val),
            use_fraction=float(self.s_use.val),
        )
        self.rebuild()

    def _toggle_pause(self) -> None:
        self.paused = not self.paused

    def _recharge(self) -> None:
        """Refill the ink sac to 100 %."""
        self.sim.stock = 1.0

    def _toggle_camera(self) -> None:
        """Switch between following the squid and a fixed frame on the whole sea."""
        self.follow = not self.follow
        lim = (-VIEW, VIEW) if self.follow else (0.0, self.params.domain)
        self.ax.set(xlim=lim, ylim=lim)
        self.field.set_visible(self.follow)
        self.field_fixed.set_visible(not self.follow)
        self.draw()
        self.fig.canvas.draw()  # refresh the blitting background for the new frame

    def rebuild(self) -> None:
        self.sim = SquidInkSimulation(self.params, seed=self.seed)
        self.t_hist: list[float] = []
        self.vis_hist: list[float] = []
        self.dist_hist: list[float] = []
        self.end_frames = 0
        self.ended = ""
        self.draw()

    # ---------------------------------------------------------------- drawing
    @staticmethod
    def _triangles(xy: np.ndarray, theta: np.ndarray, size: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        local = np.array([[1.0, 0.0], [-0.65, 0.45], [-0.65, -0.45]]) * size
        x = xy[:, 0][:, None] + local[:, 0][None, :] * c[:, None] - local[:, 1][None, :] * s[:, None]
        y = xy[:, 1][:, None] + local[:, 0][None, :] * s[:, None] + local[:, 1][None, :] * c[:, None]
        return np.stack([x, y], axis=2)

    def draw(self) -> None:
        P, sim = self.P, self.sim
        d, g = sim.p.domain, sim.p.grid
        if self.follow:
            ix, iy = int(round(sim.squid[0] / sim.dx)), int(round(sim.squid[1] / sim.dx))
            centred = np.roll(sim.ink, (g // 2 - iy, g // 2 - ix), axis=(0, 1))
            h = self.half
            self.field.set_array(centred[g // 2 - h : g // 2 + h, g // 2 - h : g // 2 + h].ravel())
            origin = np.zeros(2)
            fish_xy = wrap_delta(sim.fish - sim.squid, d)
            dots_xy = wrap_delta(self.dots_world - sim.squid, d)
        else:
            self.field_fixed.set_array(sim.ink.ravel())
            origin = np.asarray(sim.squid)
            fish_xy = sim.fish
            dots_xy = self.dots_world
        self.fish.set_verts(self._triangles(fish_xy, sim.fish_heading, 0.95))
        sees = sim.transmission > SEE
        self.fish.set_facecolor([P["predator"] if s else (0, 0, 0, 0) for s in sees])
        self.fish.set_edgecolor(P["predator"])
        self.dots.set_offsets(dots_xy)
        body, arms, tent, clubs = squid_shapes(sim.squid_heading, SQUID_SIZE)
        self.s_body.set_verts([v + origin for v in body])
        self.s_arms.set_segments([v + origin for v in arms])
        self.s_tent.set_segments([v + origin for v in tent])
        self.s_club.set_segments([v + origin for v in clubs])
        fill = 1.0 if sim.p.use_fraction == 0 else float(np.clip(sim.stock, 0.0, 1.0))
        tint = tuple(np.array(to_rgb(SPENT)) * (1 - fill) + np.array(to_rgb(P["odd"])) * fill)
        self.s_body.set_facecolor(tint)
        self.s_arms.set_color(tint)
        self.s_tent.set_color(tint)
        empty = sim.p.ink and sim.p.use_fraction > 0 and sim.stock < sim.p.empty_below
        reloading = sim.p.ink and sim.cooldown > 0 and not empty
        self.ring.set_offsets(origin[None, :] if reloading else np.empty((0, 2)))

        vis = float(sim.transmission.mean())
        if not self.paused and not self.ended:
            self.t_hist.append(sim.time)
            self.vis_hist.append(vis)
            self.dist_hist.append(sim.closest)
        if self.t_hist:
            t = np.array(self.t_hist) - sim.time
            keep = t > -WINDOW
            self.line_vis.set_data(t[keep], np.array(self.vis_hist)[keep])
            self.line_dist.set_data(t[keep], np.minimum(np.array(self.dist_hist)[keep], 30))
        h = self.hud_vals
        h["time"].set_text(f"{sim.time:6.1f}")
        h["closest"].set_text(f"{sim.closest:5.1f}")
        h["vis"].set_text(f"{vis:4.2f}")
        h["rel"].set_text(f"{sim.releases}")
        if not sim.p.ink:
            h["gland"].set_text("no ink (control)")
        elif sim.p.use_fraction == 0:
            h["gland"].set_text("unlimited")
        else:
            h["gland"].set_text("empty" if empty else f"{100 * sim.stock:.0f} % left")
        h["tally"].set_text(f"{self.tally['captured']} / {self.tally['escaped']}")
        self.banner.set_text(
            {"captured": "✕  CAPTURED", "escaped": "✓  ESCAPED"}.get(self.ended, "")
        )

    # --------------------------------------------------------------- animation
    def update(self, _frame):
        if self.paused:
            return self.animated_artists()
        if self.ended:
            self.end_frames += 1
            if self.end_frames > PAUSE_AFTER_END:
                self.seed += 1
                self.rebuild()
            else:
                self.draw()
            return self.animated_artists()
        for _ in range(self.steps_per_frame):
            self.sim.step()
            if self.sim.captured:
                self.ended = "captured"
                self.tally["captured"] += 1
                break
            if self.sim.time >= T_MAX:
                self.ended = "escaped"
                self.tally["escaped"] += 1
                break
        self.draw()
        return self.animated_artists()


def main(args=None) -> None:
    args = args if args is not None else _parse_args()
    params = InkParams(
        n_fish=args.fish, kappa=args.kappa,
        reload_time=args.reload, use_fraction=args.ink_use, ink=not args.no_ink, search=not args.blind, decoy=args.decoy,
        jet_boost=args.jet, escape_angle=args.escape_angle,
    )  # fmt: skip
    view = InkView(params, seed=args.seed, theme=args.theme, steps_per_frame=args.speed)
    if args.fixed:
        view._toggle_camera()
    ani = animation.FuncAnimation(
        view.fig, view.update, frames=args.frames if args.save else None,
        interval=33, blit=not args.save, cache_frame_data=False,
    )  # fmt: skip
    if args.save:
        writer = (
            animation.PillowWriter(fps=24)
            if args.save.endswith(".gif")
            else animation.FFMpegWriter(fps=30, bitrate=4000)
        )
        ani.save(args.save, writer=writer, dpi=90)
        print(f"saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main(_args)
