"""Interactive animation of a crowd leaving a room.

Run from the repository root:

    python amlab/collective_motion/crowd_exit_animation.py
    python amlab/collective_motion/crowd_exit_animation.py --theme light --v0 4 --column 0.8
    python amlab/collective_motion/crowd_exit_animation.py --save exit.gif --frames 300 --v0 4

What you see
------------
* Pedestrians as discs, coloured by the pressure they feel (calm blue to red).
* The door (Tech green) and, optionally, a column in front of it.
* A "pause at the door" counter: seconds since the last person got through.
  Long pauses are arches: people block each other and nobody moves.
* The number of people evacuated against time. Flat stretches shaded in red
  are pauses longer than 2 s.

Controls: desired speed (how much in a hurry, live), door width, column
distance (0 = no column), crowd size, playback speed; "Restart" and "Pause".
"""

from __future__ import annotations

import argparse
import sys

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
        "--frames", type=int, default=300, help="frames to render with --save"
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=150, help="number of pedestrians")
    ap.add_argument("--v0", type=float, default=3.0, help="desired speed in m/s")
    ap.add_argument("--door", type=float, default=1.0, help="door width in m")
    ap.add_argument(
        "--column",
        type=float,
        default=0.0,
        help="distance in m from the door to the nearest point of a column (0 = no column)",
    )
    ap.add_argument(
        "--speed", type=int, default=4, help="playback speed (1 = real time)"
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
from matplotlib.collections import EllipseCollection, LineCollection, PolyCollection
from matplotlib.colors import Normalize, to_rgba
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button, Slider

from amlab.collective_motion.crowd_exit import CrowdParams, CrowdSim
from amlab.collective_motion.ie_style import apply_theme, pressure_cmap, style_slider

STEPS_PER_FRAME_AT_1X = 17  # 1 frame = 1/30 s of simulated time at dt = 0.002 s
PRESSURE_MAX = 2500.0  # N/m: colour scale saturates here


class CrowdView:
    def __init__(
        self, params: CrowdParams, seed: int = 1, theme: str = "dark", speed: int = 4
    ):
        self.theme = theme
        self.p = apply_theme(theme)
        self.cmap = pressure_cmap(theme)
        self.norm = Normalize(0.0, PRESSURE_MAX)
        self.params = params
        self.seed = seed
        self.speed = speed
        self.paused = False
        self._build_figure()
        self.rebuild()

    # ------------------------------------------------------------------ layout
    def _build_figure(self) -> None:
        P = self.p
        self.fig = plt.figure(figsize=(13, 7.2), facecolor=P["bg"])
        self.ax = self.fig.add_axes([0.02, 0.05, 0.48, 0.90])
        ax = self.ax
        ax.set_facecolor(P["bg"])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)

        self.title = self.fig.text(
            0.54,
            0.945,
            "Faster is slower",
            fontsize=20,
            fontweight="bold",
            color=P["text"],
            va="center",
        )
        self.fig.text(
            0.54,
            0.905,
            "A crowd leaves through one door. Pushing harder does not help.",
            fontsize=10,
            color=P["muted"],
            va="center",
        )

        # HUD
        self.ax_hud = self.fig.add_axes([0.54, 0.60, 0.44, 0.27])
        self.ax_hud.set_facecolor(P["panel"])
        self.ax_hud.set_xticks([])
        self.ax_hud.set_yticks([])
        self.ax_hud.grid(False)
        for s in self.ax_hud.spines.values():
            s.set_color(P["grid"])
        self.ax_hud.set_xlim(0, 1)
        self.ax_hud.set_ylim(0, 1)
        self.hud_vals = {}
        y = 0.88
        for key, label in [
            ("time", "Time"),
            ("out", "Evacuated"),
            ("flow", "Flow through the door (last 10 s)"),
            ("press", "Pressure on people near the door"),
            ("clog", "Pause at the door"),
        ]:
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
            y -= 0.155
        # pressure colour bar (drawn as a small image) with text labels
        grad = np.linspace(0, 1, 128)[None, :]
        self.ax_hud.imshow(
            grad,
            extent=[0.04, 0.60, 0.015, 0.085],
            aspect="auto",
            cmap=self.cmap,
            zorder=3,
        )
        self.ax_hud.text(
            0.04, 0.12, "calm", color=P["muted"], fontsize=8, ha="left", va="bottom"
        )
        self.ax_hud.text(
            0.60,
            0.12,
            "crushing",
            color=P["muted"],
            fontsize=8,
            ha="right",
            va="bottom",
        )
        self.ax_hud.set_xlim(0, 1)
        self.ax_hud.set_ylim(0, 1)

        # evacuated vs time
        self.ax_cum = self.fig.add_axes([0.58, 0.335, 0.40, 0.19])
        self.ax_cum.set_xlabel("time (s)", fontsize=9)
        self.ax_cum.set_ylabel("people out", fontsize=9)
        self.ax_cum.tick_params(labelsize=8)
        self.ax_cum.set_title(
            "People evacuated (red: pauses longer than 2 s)", fontsize=10, loc="left"
        )
        (self.line_cum,) = self.ax_cum.plot(
            [], [], color=P["accent"], lw=2.2, drawstyle="steps-post"
        )
        self.pauses = PolyCollection(
            [], facecolor=to_rgba(P["bad"], 0.35), edgecolor="none"
        )
        self.ax_cum.add_collection(self.pauses)

        def mk(rect, label, lo, hi, init, step, fmt):
            axs = self.fig.add_axes(rect, facecolor=P["panel"])
            sl = Slider(axs, label, lo, hi, valinit=init, valstep=step, valfmt=fmt)
            style_slider(sl, P)
            return sl

        x0, w = 0.73, 0.23
        pr = self.params
        self.s_v0 = mk(
            [x0, 0.255, w, 0.022], "Desired speed (m/s)  ", 0.5, 8.0, pr.v0, 0.1, "%.1f"
        )
        self.s_door = mk(
            [x0, 0.215, w, 0.022], "Door width (m)  ", 0.8, 2.0, pr.door_w, 0.05, "%.2f"
        )
        self.s_col = mk(
            [x0, 0.175, w, 0.022],
            "Column distance (m)  ",
            0.0,
            2.5,
            pr.obs_gap if pr.obstacle else 0.0,
            0.1,
            "%.1f",
        )
        self.s_n = mk([x0, 0.135, w, 0.022], "People  ", 50, 250, pr.n, 10, "%d")
        self.s_speed = mk(
            [x0, 0.095, w, 0.022], "Playback speed  ", 1, 8, self.speed, 1, "%dx"
        )
        self.btn = Button(
            self.fig.add_axes([0.54, 0.035, 0.12, 0.04]),
            "Restart",
            color=P["slider_bg"],
            hovercolor=P["accent"],
        )
        self.btn_pause = Button(
            self.fig.add_axes([0.68, 0.035, 0.12, 0.04]),
            "Pause / play",
            color=P["slider_bg"],
            hovercolor=P["accent"],
        )
        for b in (self.btn, self.btn_pause):
            b.label.set_color(P["text"])
            b.label.set_fontsize(10)
        self.s_v0.on_changed(lambda v: setattr(self.sim.p, "v0", float(v)))
        self.s_speed.on_changed(lambda v: setattr(self, "speed", int(v)))
        for s in (self.s_door, self.s_col, self.s_n):
            s.on_changed(self._on_restart)
        self.btn.on_clicked(lambda _e: self._on_restart(None, bump=True))
        self.btn_pause.on_clicked(lambda _e: self._toggle_pause())
        self.fig.canvas.mpl_connect(
            "key_press_event", lambda e: self._toggle_pause() if e.key == " " else None
        )

    # ----------------------------------------------------------------- control
    def _on_restart(self, _val, bump: bool = False) -> None:
        if bump:
            self.seed += 1
        gap = float(self.s_col.val)
        self.params = self.params.with_(
            v0=float(self.s_v0.val),
            door_w=float(self.s_door.val),
            obstacle=gap > 0.0,
            obs_gap=max(gap, 0.05),
            n=int(self.s_n.val),
        )
        self.rebuild()

    def _toggle_pause(self) -> None:
        self.paused = not self.paused

    # ---------------------------------------------------------------- drawing
    def rebuild(self) -> None:
        P, ax, pr = self.p, self.ax, self.params
        self.sim = CrowdSim(pr, seed=self.seed)
        # clear static artists of the previous room
        for art in getattr(self, "_static", []):
            art.remove()
        self._static = []
        hw, H, wd = pr.room_w / 2, pr.room_h, pr.door_w / 2
        ax.set_xlim(-hw - 1.0, hw + 1.0)
        ax.set_ylim(-3.6, H + 0.8)
        outside = Rectangle(
            (-hw - 1.0, -3.6),
            pr.room_w + 2.0,
            3.6,
            facecolor=P["panel"],
            edgecolor="none",
            zorder=0,
        )
        floor = Rectangle(
            (-hw, 0), pr.room_w, H, facecolor=P["floor"], edgecolor="none", zorder=0
        )
        ax.add_patch(outside)
        ax.add_patch(floor)
        walls = LineCollection(
            self.sim.walls, colors=P["wall"], linewidths=4.0, capstyle="round", zorder=3
        )
        ax.add_collection(walls)
        door = ax.plot(
            [-wd, wd], [0, 0], color=P["good"], lw=4, ls=(0, (2, 2)), zorder=4
        )[0]
        jambs = ax.scatter([-wd, wd], [0, 0], s=60, color=P["good"], zorder=5)
        label = ax.text(
            0,
            -0.9,
            "↓  EXIT  ↓",
            color=P["good"],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            zorder=5,
        )
        self._static += [outside, floor, walls, door, jambs, label]
        if pr.obstacle:
            cx, cy = pr.obs_center
            col = Circle(
                (cx, cy),
                pr.obs_radius,
                facecolor=P["odd"],
                edgecolor=P["text"],
                lw=3.0,
                zorder=6,
            )
            ax.add_patch(col)
            self._static.append(col)

        if getattr(self, "people", None) is not None:
            self.people.remove()
        n = len(self.sim.pos)
        self.people = EllipseCollection(
            2 * self.sim.radius,
            2 * self.sim.radius,
            np.zeros(n),
            units="xy",
            offsets=self.sim.pos,
            offset_transform=ax.transData,
            edgecolors=to_rgba(P["bg"], 0.8),
            linewidths=0.8,
            zorder=7,
        )
        ax.add_collection(self.people)
        self.state_txt = getattr(self, "state_txt", None) or ax.text(
            0.02,
            0.985,
            "",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            fontweight="bold",
            color=P["text"],
            zorder=20,
        )
        self.ax_cum.set_xlim(0, 60)
        self.ax_cum.set_ylim(0, pr.n)
        self.draw()

    def _update_people(self) -> None:
        sim, P = self.sim, self.p
        n = len(sim.pos)
        if n == 0:
            self.people.set_offsets(np.zeros((0, 2)))
            return
        self.people.set_widths(2 * sim.radius)
        self.people.set_heights(2 * sim.radius)
        self.people.set_angles(np.zeros(n))
        self.people.set_offsets(sim.pos)
        rgba = self.cmap(self.norm(sim.pressure))
        outside = sim.pos[:, 1] < 0
        rgba[outside, 3] = 0.45  # people who made it are dimmed
        self.people.set_facecolor(rgba)

    def draw(self) -> None:
        sim, P = self.sim, self.p
        self._update_people()
        v = self.hud_vals
        v["time"].set_text(f"{sim.t:5.1f} s")
        v["out"].set_text(f"{sim.n_exited} / {sim.n_total}")
        v["flow"].set_text(f"{sim.flow_recent():.2f} people/s")
        v["press"].set_text(f"{sim.door_pressure():.0f} N/m")
        clog = sim.current_clog()
        v["clog"].set_text(f"{clog:.1f} s")
        v["clog"].set_color(P["bad"] if clog > 2.0 else P["text"])
        if sim.done:
            self.state_txt.set_text(f"✓  EVACUATED in {sim.exit_times[-1]:.0f} s")
        elif clog > 2.0:
            self.state_txt.set_text(
                f"▲  ARCH AT THE DOOR: {clog:.0f} s without anyone passing"
            )
        else:
            self.state_txt.set_text("▶  EVACUATING")
        # cumulative plot
        et = np.array(sim.exit_times)
        if len(et):
            xs = np.concatenate([[0.0], et, [sim.t]])
            ys = np.concatenate([[0], np.arange(1, len(et) + 1), [len(et)]])
            self.line_cum.set_data(xs, ys)
            allt = np.concatenate(
                [et, [sim.t]]
            )  # pauses are counted from the first exit on
            gaps = np.diff(allt)
            idx = np.nonzero(gaps > 2.0)[0]
            top = self.ax_cum.get_ylim()[1]
            verts = [
                [(allt[i], 0), (allt[i + 1], 0), (allt[i + 1], top), (allt[i], top)]
                for i in idx
            ]
            self.pauses.set_verts(verts)
        self.ax_cum.set_xlim(0, max(60.0, sim.t * 1.05))

    # --------------------------------------------------------------- animation
    def update(self, _frame):
        if not self.paused and not self.sim.done:
            for _ in range(STEPS_PER_FRAME_AT_1X * self.speed):
                self.sim.step()
            self.draw()
        return []


def main(args=None) -> None:
    args = args if args is not None else _parse_args()
    params = CrowdParams(
        n=args.n,
        v0=args.v0,
        door_w=args.door,
        obstacle=args.column > 0,
        obs_gap=max(args.column, 0.05),
    )
    view = CrowdView(params, seed=args.seed, theme=args.theme, speed=args.speed)
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
