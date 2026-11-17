"""Interactive animation of the confusion effect.

Run from the repository root:

    python amlab/collective_motion/couzin_predator_animation.py
    python amlab/collective_motion/couzin_predator_animation.py --theme light
    python amlab/collective_motion/couzin_predator_animation.py --save hunt.gif --frames 400

What you see
------------
* The school (Couzin model) as arrows with fading trails. The camera follows it.
* The predator (large red arrow with a white outline) and its field of view.
* While it is locked on a target: a dashed line and a ring around the target.
  Every time another prey crosses the line of sight the lock breaks (white
  ring). Three broken locks and the pass is wasted.
* Odd prey (if any) are drawn in Tech green with a white ring.

Controls: sliders for cohesion, confusion strength, school size and number of
odd prey; "New hunt" restarts; space pauses.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque

sys.path.append(".")  # run from the repository root, like the other lab scripts

import matplotlib


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--theme", choices=["dark", "light"], default="dark")
    ap.add_argument(
        "--save",
        metavar="FILE",
        help="write a .gif or .mp4 instead of opening a window",
    )
    ap.add_argument(
        "--frames", type=int, default=400, help="frames to render with --save"
    )
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--speed", type=int, default=1, help="simulation steps per frame")
    ap.add_argument("--n", type=int, default=60, help="number of prey")
    ap.add_argument("--odd", type=int, default=0, help="number of odd prey")
    ap.add_argument(
        "--cohesion", type=float, default=10.0, help="width of the zone of attraction"
    )
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
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Wedge
from matplotlib.widgets import Button, Slider

from amlab.collective_motion.couzin_predator import (
    LOCK,
    LOST,
    RECOVER,
    SEARCH,
    ConfusionSim,
    HuntParams,
)
from amlab.collective_motion.ie_style import apply_theme, style_slider

TRAIL = 14  # frames of trail
STATE_TEXT = {
    SEARCH: "○  SEARCHING",
    LOCK: "◎  LOCKED ON TARGET",
    LOST: "✕  LOCK BROKEN",
    RECOVER: "↻  RECOVERING",
}
EFFECT_LIFE = 10  # frames a ring effect stays on screen


class HuntView:
    """Matplotlib scene for a :class:`ConfusionSim`."""

    def __init__(
        self,
        params: HuntParams,
        seed: int = 3,
        theme: str = "dark",
        steps_per_frame: int = 1,
    ):
        self.theme = theme
        self.p = apply_theme(theme)
        self.params = params
        self.seed = seed
        self.steps_per_frame = steps_per_frame
        self.paused = False
        self._build_figure()
        self.rebuild()

    # ------------------------------------------------------------------ layout
    def _build_figure(self) -> None:
        P = self.p
        self.fig = plt.figure(figsize=(13, 7.2), facecolor=P["bg"])
        self.ax = self.fig.add_axes([0.02, 0.05, 0.56, 0.90])
        ax = self.ax
        ax.set_facecolor(P["floor"])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_color(P["grid"])

        # world-fixed dots give a sense of motion when the camera follows the school
        self.dots = ax.scatter([], [], s=5, color=P["grid"], zorder=1, linewidths=0)
        # prey: glow underlay + trails + oriented arrows
        self.glow = ax.scatter(
            [], [], s=150, color=P["prey"], alpha=0.10, zorder=2, linewidths=0
        )
        self.trails = LineCollection([], linewidths=1.7, zorder=3)
        ax.add_collection(self.trails)
        self.prey = PolyCollection([], zorder=5, linewidths=0.6)
        ax.add_collection(self.prey)
        # predator
        self.cone = Wedge(
            (0, 0),
            1,
            0,
            1,
            facecolor=P["predator"],
            alpha=0.10,
            edgecolor="none",
            zorder=2,
        )
        ax.add_patch(self.cone)
        self.pred = PolyCollection([], zorder=8, linewidths=1.6)
        ax.add_collection(self.pred)
        (self.lock_line,) = ax.plot(
            [], [], color=P["predator"], lw=1.6, ls=(0, (4, 3)), zorder=6
        )
        self.lock_ring = Circle(
            (0, 0), 1, fill=False, ec=P["text"], lw=2.0, zorder=7, visible=False
        )
        ax.add_patch(self.lock_ring)
        # event rings (pool)
        self.rings = []
        for _ in range(14):
            c = Circle((0, 0), 1, fill=False, lw=2.2, zorder=9, visible=False)
            ax.add_patch(c)
            self.rings.append([c, 0, None])  # patch, age, kind

        self.hud_state = ax.text(
            0.02,
            0.97,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            fontweight="bold",
            color=P["text"],
            zorder=20,
        )
        self.hud_msg = ax.text(
            0.02,
            0.925,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color=P["muted"],
            zorder=20,
        )
        self.scale_txt = ax.text(
            0.98,
            0.02,
            "",
            transform=ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=9,
            color=P["muted"],
            zorder=20,
        )

        # ---- right column
        self.fig.text(
            0.62,
            0.945,
            "The confusion effect",
            fontsize=20,
            fontweight="bold",
            color=P["text"],
            va="center",
        )
        self.fig.text(
            0.62,
            0.905,
            "A predator hunts a Couzin school. Blending in is survival.",
            fontsize=10,
            color=P["muted"],
            va="center",
        )

        self.ax_hud = self.fig.add_axes([0.62, 0.57, 0.36, 0.30])
        self.ax_hud.set_facecolor(P["panel"])
        self.ax_hud.set_xticks([])
        self.ax_hud.set_yticks([])
        self.ax_hud.grid(False)
        for s in self.ax_hud.spines.values():
            s.set_color(P["grid"])
        self.ax_hud.set_xlim(0, 1)
        self.ax_hud.set_ylim(0, 1)
        rows = [
            ("attacks", "Attacks"),
            ("captured", "Captured"),
            ("wasted", "Passes wasted (3 broken locks)"),
            ("escaped", "Strikes that missed"),
            ("p", "Capture probability"),
            ("look", "Look-alikes within 4 bodies"),
            ("nn", "Nearest-neighbour distance"),
        ]
        self.hud_vals = {}
        y = 0.90
        for key, label in rows:
            self.ax_hud.text(0.04, y, label, color=P["muted"], fontsize=10, va="center")
            self.hud_vals[key] = self.ax_hud.text(
                0.96,
                y,
                "",
                color=P["text"],
                fontsize=11,
                fontweight="bold",
                va="center",
                ha="right",
            )
            y -= 0.125

        self.ax_prob = self.fig.add_axes([0.67, 0.355, 0.31, 0.165])
        self.ax_prob.set_ylim(0, 1)
        self.ax_prob.set_xlim(0, 1)
        self.ax_prob.set_ylabel("P(capture)", fontsize=9)
        self.ax_prob.set_xlabel("attack number", fontsize=9)
        self.ax_prob.tick_params(labelsize=8)
        (self.line_prob,) = self.ax_prob.plot([], [], color=P["accent"], lw=2.2)
        self.ax_prob.set_title("Running capture probability", fontsize=10, loc="left")

        # sliders and buttons
        def mk(rect, label, lo, hi, init, step, fmt):
            axs = self.fig.add_axes(rect, facecolor=P["panel"])
            sl = Slider(axs, label, lo, hi, valinit=init, valstep=step, valfmt=fmt)
            style_slider(sl, P)
            return sl

        x0, w = 0.74, 0.20
        self.s_coh = mk(
            [x0, 0.235, w, 0.025],
            "Cohesion  ",
            0.0,
            14.0,
            self.params.d_attract,
            0.5,
            "%.1f",
        )
        self.s_conf = mk(
            [x0, 0.190, w, 0.025],
            "Confusion c  ",
            0.0,
            2.0,
            self.params.confusion,
            0.1,
            "%.1f",
        )
        self.s_n = mk(
            [x0, 0.145, w, 0.025],
            "School size  ",
            10,
            150,
            self.params.n_prey,
            10,
            "%d",
        )
        self.s_odd = mk(
            [x0, 0.100, w, 0.025], "Odd prey  ", 0, 3, self.params.n_odd, 1, "%d"
        )
        axb = self.fig.add_axes([0.62, 0.045, 0.12, 0.04])
        self.btn = Button(axb, "New hunt", color=P["slider_bg"], hovercolor=P["accent"])
        self.btn.label.set_color(P["text"])
        self.btn.label.set_fontsize(10)
        axb2 = self.fig.add_axes([0.76, 0.045, 0.12, 0.04])
        self.btn_pause = Button(
            axb2, "Pause / play", color=P["slider_bg"], hovercolor=P["accent"]
        )
        self.btn_pause.label.set_color(P["text"])
        self.btn_pause.label.set_fontsize(10)

        self.s_coh.on_changed(self._on_live)
        self.s_conf.on_changed(self._on_live)
        self.s_n.on_changed(self._on_restart)
        self.s_odd.on_changed(self._on_restart)
        self.btn.on_clicked(lambda _e: self._on_restart(None, bump=True))
        self.btn_pause.on_clicked(lambda _e: self._toggle_pause())
        self.fig.canvas.mpl_connect(
            "key_press_event", lambda e: self._toggle_pause() if e.key == " " else None
        )

    # ----------------------------------------------------------------- control
    def _on_live(self, _val) -> None:
        self.sim.p.d_attract = float(self.s_coh.val)
        self.sim.p.confusion = float(self.s_conf.val)
        self.params = self.sim.p

    def _on_restart(self, _val, bump: bool = False) -> None:
        if bump:
            self.seed += 1
        self.params = self.params.with_(
            n_prey=int(self.s_n.val),
            n_odd=int(self.s_odd.val),
            d_attract=float(self.s_coh.val),
            confusion=float(self.s_conf.val),
        )
        self.rebuild()

    def _toggle_pause(self) -> None:
        self.paused = not self.paused

    def rebuild(self) -> None:
        self.sim = ConfusionSim(self.params, seed=self.seed)
        n = self.params.n_prey
        c = self.sim.xy
        self.hist = deque([c.copy() for _ in range(TRAIL)], maxlen=TRAIL)
        self.cam = self.sim.centroid().copy()
        self.half = 30.0
        self.prob_hist: list[float] = []
        for r in self.rings:
            r[0].set_visible(False)
        self._last_outcomes = 0
        self.last_msg = ""
        self.msg_age = 0
        self.draw()

    # ---------------------------------------------------------------- drawing
    @staticmethod
    def _triangles(xy: np.ndarray, theta: np.ndarray, size: float) -> np.ndarray:
        """(N, 3, 2) vertices of arrows pointing along ``theta``."""
        c, s = np.cos(theta), np.sin(theta)
        local = np.array([[1.0, 0.0], [-0.65, 0.45], [-0.65, -0.45]]) * size
        x = (
            xy[0][:, None]
            + local[None, :, 0] * c[:, None]
            - local[None, :, 1] * s[:, None]
        )
        y = (
            xy[1][:, None]
            + local[None, :, 0] * s[:, None]
            + local[None, :, 1] * c[:, None]
        )
        return np.stack([x, y], axis=2)

    def _camera(self) -> None:
        sim = self.sim
        cen = sim.centroid()
        r = np.percentile(np.hypot(sim.xy[0] - cen[0], sim.xy[1] - cen[1]), 92)
        half = max(24.0, 1.35 * r + 9.0)
        d = np.max(np.abs(sim.pred_xy - cen))
        if d < 60.0:
            half = max(half, 1.12 * d + 4.0)
        self.cam += 0.12 * (cen - self.cam)
        self.half += 0.08 * (half - self.half)

    def draw(self) -> None:
        P, sim, ax = self.p, self.sim, self.ax
        self._camera()
        cx, cy = self.cam
        h = self.half
        ax.set_xlim(cx - h, cx + h)
        ax.set_ylim(cy - h, cy + h)
        body = max(0.9, h / 17.0)  # arrow size in world units, so prey stay visible

        # motion-cue dots
        g = 5.0 if h < 32 else 10.0
        xs = np.arange(np.floor((cx - h) / g) * g, cx + h + g, g)
        ys = np.arange(np.floor((cy - h) / g) * g, cy + h + g, g)
        gx, gy = np.meshgrid(xs, ys)
        self.dots.set_offsets(np.c_[gx.ravel(), gy.ravel()])

        # prey
        self.hist.append(sim.xy.copy())
        odd = sim.is_odd
        base = np.array(to_rgba(P["prey"]))
        oddc = np.array(to_rgba(P["odd"]))
        face = np.tile(base, (sim.p.n_prey, 1))
        face[odd] = oddc
        edge = np.tile(np.array(to_rgba(P["bg"], 0.9)), (sim.p.n_prey, 1))
        edge[odd] = to_rgba(P["text"])
        self.prey.set_verts(self._triangles(sim.xy, sim.theta, body))
        self.prey.set_facecolor(face)
        self.prey.set_edgecolor(edge)
        self.prey.set_linewidths(np.where(odd, 1.6, 0.5))
        self.glow.set_offsets(sim.xy.T)
        self.glow.set_sizes(np.full(sim.p.n_prey, (body * 9) ** 2))
        self.glow.set_facecolor([oddc if o else base for o in odd])

        # trails (older = fainter)
        H = np.array(self.hist)  # (T, 2, N)
        segs = np.stack([H[:-1], H[1:]], axis=0)  # (2, T-1, 2, N)
        segs = segs.transpose(3, 1, 0, 2).reshape(-1, 2, 2)  # (N*(T-1), 2 pts, xy)
        alpha = np.tile(np.linspace(0.02, 0.65, TRAIL - 1), sim.p.n_prey)
        colr = np.repeat(face[:, :3], TRAIL - 1, axis=0)
        self.trails.set_segments(segs)
        self.trails.set_color(np.c_[colr, alpha])

        # predator, view cone, lock
        px, py = sim.pred_xy
        pbody = body * 1.9
        self.pred.set_verts(
            self._triangles(np.array([[px], [py]]), np.array([sim.pred_theta]), pbody)
        )
        self.pred.set_facecolor(P["predator"])
        self.pred.set_edgecolor(P["predator_edge"])
        fov = sim.p.fov_pred_deg
        deg = np.degrees(sim.pred_theta)
        self.cone.set_center((px, py))
        self.cone.set_radius(sim.p.sight)
        self.cone.set_theta1(deg - fov / 2)
        self.cone.set_theta2(deg + fov / 2)
        if sim.state in (LOCK, LOST) and sim.target >= 0:
            tx, ty = sim.xy[:, sim.target]
            self.lock_line.set_data([px, tx], [py, ty])
            self.lock_ring.set_center((tx, ty))
            self.lock_ring.set_radius(body * 1.8)
            self.lock_ring.set_visible(sim.state == LOCK)
            self.lock_line.set_alpha(1.0 if sim.state == LOCK else 0.25)
        else:
            self.lock_line.set_data([], [])
            self.lock_ring.set_visible(False)

        # event rings
        for ev in sim.events:
            self._spawn_ring(ev)
        sim.events.clear()
        for r in self.rings:
            patch, age, kind = r
            if kind is None:
                continue
            age += 1
            r[1] = age
            if age > EFFECT_LIFE:
                patch.set_visible(False)
                r[2] = None
                continue
            patch.set_radius(
                body
                * (1.5 + 0.9 * age)
                * (1.8 if kind in ("capture", "abort") else 1.0)
            )
            patch.set_alpha(1.0 - age / EFFECT_LIFE)

        # HUD
        st = sim.stats
        lock_txt = (
            f"   broken locks {sim.breaks} / {sim.p.max_breaks}"
            if sim.state in (LOCK, LOST)
            else ""
        )
        self.hud_state.set_text(STATE_TEXT[sim.state] + lock_txt)
        self.msg_age += 1
        self.hud_msg.set_text(self.last_msg if self.msg_age < 60 else "")
        self.scale_txt.set_text(
            f"view {2 * h:.0f} body lengths · t = {sim.t * sim.p.dt:.0f} s"
        )
        m = sim.school_metrics()
        done = st["attacks"] - (1 if sim.state in (LOCK, LOST) else 0)
        pc = st["captures"] / done if done > 0 else float("nan")
        v = self.hud_vals
        v["attacks"].set_text(f"{done}")
        v["captured"].set_text(f"{st['captures']}")
        v["wasted"].set_text(f"{st['aborts'] + st['timeouts']}")
        v["escaped"].set_text(f"{st['escapes']}")
        v["p"].set_text("n/a" if np.isnan(pc) else f"{100 * pc:.0f} %")
        v["look"].set_text(f"{m['lookalikes']:.1f}")
        v["nn"].set_text(f"{m['nn_distance']:.1f}")
        # running probability, one point per finished attack
        if done > len(self.prob_hist):
            self.prob_hist.append(pc)
            xs = np.arange(1, len(self.prob_hist) + 1)
            self.line_prob.set_data(xs, self.prob_hist)
            self.ax_prob.set_xlim(1, max(10, len(self.prob_hist)))

    def _spawn_ring(self, ev: dict) -> None:
        P = self.p
        kind = ev["type"]
        colour = {
            "break": P["text"],
            "capture": P["predator"],
            "escape": P["good"],
            "abort": P["muted"],
        }[kind]
        self.msg_age = 0
        self.last_msg = {
            "break": "A look-alike crossed the line of sight: lock broken",
            "capture": "Strike succeeded: prey captured",
            "escape": "Strike missed: the prey escaped",
            "abort": "Three broken locks: the pass is wasted",
        }[kind]
        for r in self.rings:
            if r[2] is None:
                r[0].set_center((ev["x"], ev["y"]))
                r[0].set_edgecolor(colour)
                r[0].set_visible(True)
                r[1], r[2] = 0, kind
                return

    # --------------------------------------------------------------- animation
    def update(self, _frame):
        if not self.paused:
            for _ in range(self.steps_per_frame):
                self.sim.step()
            self.draw()
        return []


def main(args=None) -> None:
    args = args if args is not None else _parse_args()
    params = HuntParams(n_prey=args.n, n_odd=args.odd, d_attract=args.cohesion)
    view = HuntView(
        params, seed=args.seed, theme=args.theme, steps_per_frame=args.speed
    )
    ani = animation.FuncAnimation(
        view.fig,
        view.update,
        frames=args.frames if args.save else None,
        interval=33,
        blit=False,
        cache_frame_data=False,
    )
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
