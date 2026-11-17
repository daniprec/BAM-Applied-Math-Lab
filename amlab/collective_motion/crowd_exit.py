"""Crowd evacuation through a narrow door: faster-is-slower and the obstacle.

This is the physics-based social force model of Helbing, Farkas and Vicsek,
*Nature* 407, 487 (2000), which extends the social force model of
:doc:`social-force.qmd` (Helbing & Molnar, 1995) with *contact forces*:

    m dv_i/dt = m (v0 e_i - v_i) / tau                       (wish to move)
              + sum_j [ A exp((r_ij - d_ij)/B) + k g(r_ij - d_ij) ] n_ij   (social + body force)
              + sum_j  kappa g(r_ij - d_ij) (Dv_ji . t_ij) t_ij            (sliding friction)
              + the same terms for walls and for an optional column.

Here ``g(x) = x`` for ``x > 0`` (bodies touching) and 0 otherwise. Pedestrians
are discs. The room has one exit; whoever crosses the door line has escaped.

Two phenomena can be reproduced with this script:

* **Faster is slower.** When everybody pushes harder (larger desired speed
  ``v0``) arches form at the door, the flow drops and the evacuation takes
  *longer* (Helbing et al. 2000; experiments: Pastor et al. 2015).
* **A column in front of the door** can make the crowd flow better in silos
  and with sheep (Zuriguel et al. 2011, 2016). With people, an experiment with
  about 170 soldiers found *no change* in the flow but a reduction of the
  sideways rushes (Garcimartin et al. 2018). The model lets you explore why.

The parameters are the published ones, except that the time step and the
contact stiffness are chosen so that a run takes seconds instead of minutes
(see ``CrowdParams``).

Run ``python amlab/collective_motion/crowd_exit.py`` for a short self-test.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class CrowdParams:
    """Parameters of the evacuation. SI units (m, s, kg, N)."""

    # --- room and door -----------------------------------------------------
    n: int = 150
    room_w: float = 12.0
    room_h: float = 12.0
    door_w: float = 1.0
    # --- column (obstacle) in front of the door ----------------------------
    obstacle: bool = False
    obs_radius: float = 0.5
    obs_gap: float = 0.8  # distance from the door line to the nearest point of the column
    # --- pedestrians ---------------------------------------------------------
    v0: float = 1.5  # desired speed (m/s): "how much in a hurry"
    mass: float = 80.0
    r_min: float = 0.25
    r_max: float = 0.35
    tau: float = 0.5
    # --- interaction (Helbing et al. 2000) ------------------------------------
    A: float = 2000.0  # N
    B: float = 0.08  # m
    k: float = 1.2e5  # body force, kg / s^2
    kappa: float = 2.4e5  # sliding friction, kg / (m s)
    # --- numerics ----------------------------------------------------------------
    dt: float = 0.002  # s; 0.001 gives the same statistics at twice the cost

    def with_(self, **kw) -> "CrowdParams":
        return replace(self, **kw)

    @property
    def obs_center(self) -> tuple[float, float]:
        return (0.0, self.obs_gap + self.obs_radius)


class CrowdSim:
    """Discs in a room with one door.

    Attributes
    ----------
    pos, vel : (n_alive, 2) arrays of the pedestrians still in the simulation
    radius : (n_alive,) radii
    pressure : (n_alive,) contact force per perimeter length, in N/m
    crossed : (n_alive,) bool, True once the centre has crossed the door line
    exit_times : list of times (s) at which pedestrians crossed the door line
    t : current time (s)
    """

    CUTOFF = 0.5  # m: ignore the (exponentially small) social force beyond this gap

    def __init__(self, params: CrowdParams | None = None, seed: int = 0):
        self.p = params if params is not None else CrowdParams()
        self.seed = seed
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        p = self.p
        self.rng = np.random.default_rng(self.seed)
        self.t = 0.0
        self.exit_times: list[float] = []
        hw = p.room_w / 2.0
        wd = p.door_w / 2.0
        # wall segments: left, right, top and the two halves of the bottom wall
        self.walls = np.array(
            [
                [[-hw, 0.0], [-hw, p.room_h]],
                [[hw, 0.0], [hw, p.room_h]],
                [[-hw, p.room_h], [hw, p.room_h]],
                [[-hw, 0.0], [-wd, 0.0]],
                [[wd, 0.0], [hw, 0.0]],
            ]
        )
        self._place_pedestrians()
        n = len(self.pos)
        self.vel = np.zeros((n, 2))
        self.pressure = np.zeros(n)
        self.crossed = np.zeros(n, dtype=bool)
        self.n_total = n
        self.ever_pressure = 0.0

    def _place_pedestrians(self) -> None:
        """Random sequential addition: no overlaps at the start."""
        p = self.p
        hw = p.room_w / 2.0
        pos, rad = [], []
        tries = 0
        y_low = 1.2 + (p.obs_gap + 2 * p.obs_radius if p.obstacle else 0.0)
        while len(pos) < p.n and tries < 200000:
            tries += 1
            r = self.rng.uniform(p.r_min, p.r_max)
            x = self.rng.uniform(-hw + r + 0.05, hw - r - 0.05)
            y = self.rng.uniform(y_low, p.room_h - r - 0.05)
            if all(np.hypot(x - a, y - b) > r + c + 0.02 for (a, b), c in zip(pos, rad)):
                pos.append((x, y))
                rad.append(r)
        if len(pos) < p.n:
            raise ValueError("Room too small for this many pedestrians")
        self.pos = np.array(pos)
        self.radius = np.array(rad)

    # ----------------------------------------------------------------- forces
    def _contact(self, d, rij, n_vec, v_rel, social_scale=1.0):
        """Pair force for centre distance ``d`` and radii sum ``rij``.

        Returns the force on the first body and its contact (body-force) part.
        ``social_scale`` switches the exponential social force off (0) or on (1).
        """
        p = self.p
        g = rij - d
        overlap = np.maximum(g, 0.0)
        social = social_scale * p.A * np.exp(np.minimum(g, 0.5) / p.B)
        body = p.k * overlap
        t_vec = np.stack((-n_vec[:, 1], n_vec[:, 0]), axis=1)
        f_t = p.kappa * overlap * (v_rel * t_vec).sum(axis=1)
        return (social + body)[:, None] * n_vec + f_t[:, None] * t_vec, body

    def step(self) -> None:
        p = self.p
        pos, vel, r = self.pos, self.vel, self.radius
        n = len(pos)
        if n == 0:
            self.t += p.dt
            return

        # (1) wish to move: towards a point just outside the door
        e = np.array([0.0, -1.0]) - pos
        e[pos[:, 1] < 0.0] = (0.0, -1.0)
        e /= np.maximum(np.hypot(e[:, 0], e[:, 1]), 1e-9)[:, None]
        F = p.mass * (p.v0 * e - vel) / p.tau
        contact = np.zeros(n)

        # (2) pedestrians
        if n > 1:
            pairs = cKDTree(pos).query_pairs(2 * p.r_max + self.CUTOFF, output_type="ndarray")
            if len(pairs):
                i, j = pairs[:, 0], pairs[:, 1]
                dxy = pos[i] - pos[j]
                d = np.maximum(np.hypot(dxy[:, 0], dxy[:, 1]), 1e-9)
                n_vec = dxy / d[:, None]
                f, body = self._contact(d, r[i] + r[j], n_vec, vel[j] - vel[i])
                F[:, 0] += np.bincount(i, f[:, 0], n) - np.bincount(j, f[:, 0], n)
                F[:, 1] += np.bincount(i, f[:, 1], n) - np.bincount(j, f[:, 1], n)
                contact += np.bincount(i, body, n) + np.bincount(j, body, n)

        # (3) walls: nearest point on every segment
        a, b = self.walls[:, 0], self.walls[:, 1]
        ab = b - a
        tt = np.clip(((pos[:, None, :] - a) * ab).sum(2) / (ab * ab).sum(1), 0.0, 1.0)
        q = a + tt[..., None] * ab  # (n, S, 2)
        dv = pos[:, None, :] - q
        dist = np.maximum(np.hypot(dv[..., 0], dv[..., 1]), 1e-9)
        near = dist < r[:, None] + self.CUTOFF
        ni, si = np.nonzero(near)
        if len(ni):
            n_vec = dv[ni, si] / dist[ni, si][:, None]
            # The social force acts along wall *faces*. At a free end (the door jambs)
            # only the contact force acts, otherwise two jambs can repel a slow
            # pedestrian more strongly than it pushes, and it stalls in the doorway.
            face = (tt[ni, si] > 0.0) & (tt[ni, si] < 1.0)
            f, body = self._contact(dist[ni, si], r[ni], n_vec, -vel[ni], social_scale=face.astype(float))
            F[:, 0] += np.bincount(ni, f[:, 0], n)
            F[:, 1] += np.bincount(ni, f[:, 1], n)
            contact += np.bincount(ni, body, n)

        # (4) the column, a fixed disc
        if p.obstacle:
            c = np.array(p.obs_center)
            dv = pos - c
            d = np.maximum(np.hypot(dv[:, 0], dv[:, 1]), 1e-9)
            sel = np.nonzero(d < r + p.obs_radius + self.CUTOFF)[0]
            if len(sel):
                f, body = self._contact(d[sel], r[sel] + p.obs_radius, dv[sel] / d[sel][:, None], -vel[sel])
                F[sel] += f
                contact[sel] += body

        # (5) integrate (semi-implicit Euler) with a speed cap for numerical safety
        vel += p.dt * F / p.mass
        vmax = max(3.0, 1.6 * p.v0)
        speed = np.hypot(vel[:, 0], vel[:, 1])
        too_fast = speed > vmax
        if too_fast.any():
            vel[too_fast] *= (vmax / speed[too_fast])[:, None]
        pos += p.dt * vel
        self.pressure = contact / (2.0 * np.pi * r)
        self.t += p.dt

        # (6) bookkeeping: who crossed the door line, who can be removed
        newly = (pos[:, 1] < 0.0) & ~self.crossed
        if newly.any():
            self.crossed |= newly
            self.exit_times.extend([self.t] * int(newly.sum()))
        gone = pos[:, 1] < -2.5
        if gone.any():
            keep = ~gone
            self.pos, self.vel, self.radius = pos[keep], vel[keep], r[keep]
            self.crossed, self.pressure = self.crossed[keep], self.pressure[keep]

    # --------------------------------------------------------------- measures
    @property
    def n_exited(self) -> int:
        return len(self.exit_times)

    @property
    def done(self) -> bool:
        return self.n_exited >= self.n_total

    def flow_recent(self, window: float = 10.0) -> float:
        """Pedestrians per second through the door in the last ``window`` seconds."""
        if not self.exit_times:
            return 0.0
        w = min(window, self.t)
        recent = sum(1 for t in reversed(self.exit_times) if t > self.t - w)
        return recent / max(w, 1e-9)

    def door_pressure(self, radius: float = 2.0) -> float:
        """Mean contact pressure (N/m) of the people within ``radius`` m of the door."""
        if len(self.pos) == 0:
            return 0.0
        near = np.hypot(self.pos[:, 0], self.pos[:, 1]) < radius
        return float(self.pressure[near].mean()) if near.any() else 0.0

    def current_clog(self) -> float:
        """Seconds since the last pedestrian crossed the door line (0 before the first one)."""
        if not self.exit_times or self.done:  # the walk to the door is not a pause
            return 0.0
        return self.t - self.exit_times[-1]


# ---------------------------------------------------------------------------
# Numerical experiments
# ---------------------------------------------------------------------------
def evacuation_stats(exit_times, n_total: int) -> dict:
    """Summary of one evacuation from the list of exit times."""
    t = np.sort(np.asarray(exit_times, dtype=float))
    out = dict(n_exited=len(t), evac_time=float(t[-1]) if len(t) else float("nan"))
    if len(t) >= 10:
        lo, hi = int(0.1 * len(t)), int(0.9 * len(t)) - 1
        span = t[hi] - t[lo]
        out["flow"] = (hi - lo) / span if span > 0 else float("nan")  # persons / s, middle 80 %
        gaps = np.diff(t)
        out["max_gap"] = float(gaps.max())
        out["clog_fraction"] = float(gaps[gaps > 2.0].sum() / (t[-1] - t[0]))  # share of the time spent in pauses > 2 s
    else:
        out.update(flow=float("nan"), max_gap=float("nan"), clog_fraction=float("nan"))
    out["complete"] = len(t) >= n_total
    return out


def run_evacuation(params: CrowdParams, seed: int = 0, t_max: float = 300.0) -> dict:
    """Run one evacuation until everybody is out or ``t_max`` seconds have passed."""
    sim = CrowdSim(params, seed=seed)
    pressure_sum, pressure_n = 0.0, 0
    k = 0
    while not sim.done and sim.t < t_max:
        sim.step()
        k += 1
        if k % 100 == 0:
            pressure_sum += sim.door_pressure()
            pressure_n += 1
    out = evacuation_stats(sim.exit_times, sim.n_total)
    out["door_pressure"] = pressure_sum / max(pressure_n, 1)
    out["gaps"] = np.diff(np.sort(sim.exit_times))
    if not out["complete"]:
        out["evac_time"] = float(t_max)  # censored: report the cap
    return out


def sweep(param: str, values, base: CrowdParams | None = None, n_seeds: int = 3, t_max: float = 300.0) -> dict:
    """Mean and standard deviation over ``n_seeds`` rooms of the evacuation time, flow,
    pressure at the door and the share of runs in which everybody got out (``complete``)."""
    base = base if base is not None else CrowdParams()
    keys = ("evac_time", "flow", "max_gap", "clog_fraction", "door_pressure", "complete")
    res = {"values": np.array(list(values), dtype=float)}
    for key in keys:
        res[key] = np.zeros(len(res["values"]))
        res[key + "_std"] = np.zeros(len(res["values"]))
    for a, v in enumerate(res["values"]):
        runs = [run_evacuation(base.with_(**{param: v}), seed=s, t_max=t_max) for s in range(n_seeds)]
        for key in keys:
            vals = np.array([r[key] for r in runs], dtype=float)
            with warnings.catch_warnings():  # all-NaN (nobody got out) gives NaN, quietly
                warnings.simplefilter("ignore", RuntimeWarning)
                res[key][a] = np.nanmean(vals)
                res[key + "_std"][a] = np.nanstd(vals)
    return res


if __name__ == "__main__":
    import time

    for v0 in (0.8, 1.5, 3.0, 6.0):
        t0 = time.time()
        r = run_evacuation(CrowdParams(v0=v0), seed=0)
        print(f"v0 = {v0:3.1f} m/s  evac time = {r['evac_time']:6.1f} s  flow = {r['flow']:.2f} /s  "
              f"max gap = {r['max_gap']:.1f} s  pressure = {r['door_pressure']:.0f} N/m   ({time.time() - t0:.1f} s)")
