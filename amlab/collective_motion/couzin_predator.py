"""The confusion effect: a predator hunting a Couzin school.

This module *extends* ``couzin.py``. The prey follow exactly the zone rules of
Couzin et al. (2002) (repulsion > orientation + attraction, blind rear sector,
limited turning rate, Gaussian noise). On top of them we add

1. a **predator** with a finite field of view, a limited turning rate and a
   four-state hunting cycle (search, lock, lost, recover);
2. the **confusion effect**: while the predator is locked on a target, every
   other prey that crosses the line of sight breaks the lock. After
   ``max_breaks`` broken locks the pass is wasted. If the lock holds until the
   strike, the capture probability falls with the mean number ``n`` of
   look-alikes that crossed the line of sight during the approach,
   ``P = 1 / (1 + c * n)``;
3. the **oddity effect**: an "odd" prey (think of a marked or limping fish)
   attracts the predator, its lock cannot be broken, and it is captured with
   certainty;
4. a simple **evasion** reflex: prey close to the predator turn away from it.

The sources of these ideas are listed in ``confusion-effect.qmd``. The lock and
strike rules are teaching simplifications in the spirit of Landeau & Terborgh
(1986), Ioannou et al. (2008) and Olson et al. (2013); they are not a
quantitative fit to any data set.

Run ``python amlab/collective_motion/couzin_predator.py`` for a self-test and a
short cohesion sweep.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace

import numpy as np

sys.path.append(".")  # run from the repository root, like the other lab scripts

# Re-use the course code: same initial condition as the original lab.
from amlab.collective_motion.couzin import (
    couzin_equations,
    initialize_particles_couzin,
)

TWO_PI = 2.0 * np.pi

# Predator states
SEARCH, LOCK, LOST, RECOVER = "search", "lock", "lost", "recover"


def wrap_pi(a):
    """Wrap angles to [-pi, pi)."""
    return (a + np.pi) % TWO_PI - np.pi


# ---------------------------------------------------------------------------
# 1. Vectorised Couzin step (same rules as couzin.couzin_equations)
# ---------------------------------------------------------------------------
def couzin_headings(
    xy: np.ndarray,
    theta: np.ndarray,
    dt: float = 0.1,
    radius_repulsion: float = 1.0,
    radius_alignment: float = 6.0,
    radius_attraction: float = 14.0,
    noise_std: float = 0.05,
    perception_angle_deg: float = 270.0,
    turning_rate_deg=40.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """New headings after one Couzin step, computed with (N, N) arrays.

    Same rules as :func:`amlab.collective_motion.couzin.couzin_equations`; only
    the loop over individuals is replaced by array operations, which is 10 to
    50 times faster for the school sizes used here. ``turning_rate_deg`` may be
    a scalar or an array with one value per individual.
    """
    rng = np.random.default_rng() if rng is None else rng
    x, y = xy
    n = x.size
    half_fov = np.deg2rad(perception_angle_deg) / 2.0
    max_turn = np.deg2rad(turning_rate_deg) * dt

    dx = x[None, :] - x[:, None]  # dx[i, j] = x_j - x_i
    dy = y[None, :] - y[:, None]
    d = np.hypot(dx, dy)
    d_safe = np.where(d > 0.0, d, 1.0)
    ux, uy = dx / d_safe, dy / d_safe  # unit vector i -> j

    bearing = np.arctan2(dy, dx)
    visible = (d <= radius_attraction) & (
        np.abs(wrap_pi(bearing - theta[:, None])) <= half_fov
    )
    visible &= ~np.eye(n, dtype=bool)

    rep = visible & (d < radius_repulsion)
    ori = visible & (d >= radius_repulsion) & (d < radius_alignment)
    att = visible & (d >= radius_alignment) & (d < radius_attraction)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Repulsion: away from every neighbour in the zor
    rep_x, rep_y = -(rep * ux).sum(1), -(rep * uy).sum(1)
    # Orientation + attraction
    oa_x = (ori * cos_t[None, :]).sum(1) + (att * ux).sum(1)
    oa_y = (ori * sin_t[None, :]).sum(1) + (att * uy).sum(1)

    has_rep = rep.any(axis=1)
    dir_x = np.where(has_rep, rep_x, oa_x)
    dir_y = np.where(has_rep, rep_y, oa_y)
    lazy = np.hypot(dir_x, dir_y) < 1e-8  # nothing to react to: keep heading
    desired = np.where(lazy, theta, np.arctan2(dir_y, dir_x))

    desired = (desired + noise_std * rng.standard_normal(n)) % TWO_PI
    dh = np.clip(wrap_pi(desired - theta), -max_turn, max_turn)
    return (theta + dh) % TWO_PI


def couzin_step_vectorized(
    xy: np.ndarray,
    theta: np.ndarray,
    dt: float = 0.1,
    v0: float = 3.0,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised equivalent of ``couzin_equations`` (positions and headings)."""
    theta_new = couzin_headings(xy, theta, dt=dt, rng=rng, **kwargs)
    v = v0 * np.vstack((np.cos(theta_new), np.sin(theta_new)))
    return xy + dt * v, theta_new


def check_against_original(n: int = 60, seed: int = 1, steps: int = 200) -> float:
    """Largest position difference between the course code and the vectorised
    version when the noise is switched off (they must agree to rounding)."""
    np.random.seed(seed)
    xy, th = initialize_particles_couzin(n, box_size=25.0)
    xy_a, th_a = xy.copy(), th.copy()
    xy_b, th_b = xy.copy(), th.copy()
    worst = 0.0
    for _ in range(steps):
        xy_a, th_a = couzin_equations(xy_a, th_a, noise_std=0.0)
        xy_b, th_b = couzin_step_vectorized(xy_b, th_b, noise_std=0.0)
        worst = max(worst, float(np.abs(xy_a - xy_b).max()))
    return worst


# ---------------------------------------------------------------------------
# 2. Parameters
# ---------------------------------------------------------------------------
@dataclass
class HuntParams:
    """All parameters of the hunt. Lengths in body units, times in seconds."""

    # --- school (Couzin zones; cohesion = width of the zone of attraction) ---
    n_prey: int = 60
    n_odd: int = 0  # number of "odd" (marked) prey
    v_prey: float = 3.0
    r_rep: float = 1.0
    d_orient: float = 3.0  # width of the zone of orientation
    d_attract: float = 10.0  # width of the zone of attraction ("cohesion")
    noise: float = 0.05
    fov_deg: float = 270.0
    turn_deg: float = 40.0
    dt: float = 0.1
    # --- evasion reflex ---
    flee_radius: float = 4.0
    flee_turn_deg: float = 120.0
    flee_boost: float = 1.15
    # --- predator ---
    v_pred: float = 5.0
    turn_pred_deg: float = 120.0
    sight: float = 22.0
    fov_pred_deg: float = 140.0
    # --- confusion and oddity ---
    lock_tol_deg: float = 8.0  # a prey within this angle of the target breaks the lock
    clutter_depth: float = 8.0  # look-alikes up to this far behind the target still clutter the view
    max_breaks: int = 3  # broken locks before the pass is wasted
    relock_steps: int = 3  # steps the predator is "blind" after a break
    strike_radius: float = 0.9
    confusion_radius: float = 4.0  # radius used for the school metric "look-alikes"
    confusion: float = 1.0  # c in P = 1 / (1 + c n); 0 switches confusion off
    recover_time: float = 3.0
    max_attack_time: float = 10.0

    @property
    def r_orient(self) -> float:
        return self.r_rep + self.d_orient

    @property
    def r_attract(self) -> float:
        return self.r_rep + self.d_orient + self.d_attract

    def with_(self, **kw) -> "HuntParams":
        return replace(self, **kw)


def strike_probability(n_lookalikes: int, confusion: float = 1.0) -> float:
    """Capture probability of a strike that reaches its target.

    ``n_lookalikes`` is the number of other prey within the confusion radius of
    the target. One look-alike halves the chance, two cut it to one third, and
    so on (``c = 1``).
    """
    return 1.0 / (1.0 + confusion * n_lookalikes)


# ---------------------------------------------------------------------------
# 3. The simulation
# ---------------------------------------------------------------------------
class ConfusionSim:
    """A Couzin school hunted by one predator.

    Attributes (read them to draw the scene)
    ----------------------------------------
    xy, theta : prey positions (2, N) and headings (N,)
    pred_xy, pred_theta : predator position (2,) and heading
    state : one of ``search``, ``lock``, ``lost``, ``recover``
    target : index of the locked prey, or -1
    breaks : broken locks in the current attack
    is_odd : boolean mask of odd prey
    events : list of dicts ``{"type", "x", "y", "t"}`` for visual effects
             (``break``, ``capture``, ``escape``, ``abort``). Clear it after use.
    stats : dict with counters (attacks, captures, escapes, aborts, ...)
    """

    def __init__(self, params: HuntParams | None = None, seed: int = 0, burn_in: int = 300):
        self.p = params if params is not None else HuntParams()
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.burn_in = burn_in
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        p = self.p
        # Same initial condition as the course lab (uniform in a 25 x 25 box)
        np.random.seed(int(self.rng.integers(2**31 - 1)))
        self.xy, self.theta = initialize_particles_couzin(p.n_prey, box_size=25.0)
        self.is_odd = np.zeros(p.n_prey, dtype=bool)
        self.is_odd[: min(p.n_odd, p.n_prey)] = True
        self.hunting = False
        self.t = 0
        self.events: list[dict] = []
        self.stats = self._empty_stats()
        self.outcomes: list[tuple[int, str]] = []  # (step, outcome) of each attack
        self._reset_predator()
        for _ in range(self.burn_in):  # let the school form before the predator comes
            self._step_prey()
        self._place_predator()
        self.hunting = True
        self.t = 0

    @staticmethod
    def _empty_stats() -> dict:
        return dict(attacks=0, captures=0, escapes=0, aborts=0, timeouts=0, breaks=0, strikes=0)

    def _reset_predator(self) -> None:
        self.pred_xy = np.array([1e4, 1e4])
        self.pred_theta = 0.0
        self.state = SEARCH
        self.target = -1
        self.breaks = 0
        self._lost_steps = 0
        self._recover_steps = 0
        self._attack_steps = 0
        self._cone_sum = 0.0  # look-alikes seen in the line of sight (summed over the approach)
        self._cone_n = 0
        self._prev_cross = np.zeros(self.p.n_prey, dtype=bool)

    def _place_predator(self) -> None:
        c = self.xy.mean(axis=1)
        ang = self.rng.uniform(0, TWO_PI)
        self.pred_xy = c + 35.0 * np.array([np.cos(ang), np.sin(ang)])
        self.pred_theta = (ang + np.pi) % TWO_PI
        self.state = SEARCH

    # ------------------------------------------------------------ measurements
    def centroid(self) -> np.ndarray:
        return self.xy.mean(axis=1)

    def school_metrics(self) -> dict:
        """Polarization, nearest-neighbour distance and mean look-alike count."""
        x, y = self.xy
        d = np.hypot(x[None, :] - x[:, None], y[None, :] - y[:, None])
        np.fill_diagonal(d, np.inf)
        polar = float(np.hypot(np.cos(self.theta).mean(), np.sin(self.theta).mean()))
        return dict(
            polarization=polar,
            nn_distance=float(d.min(axis=1).mean()),
            lookalikes=float((d < self.p.confusion_radius).sum(axis=1).mean()),
        )

    # ------------------------------------------------------------------- prey
    def _step_prey(self) -> None:
        p = self.p
        n = p.n_prey
        theta_new = couzin_headings(
            self.xy,
            self.theta,
            dt=p.dt,
            radius_repulsion=p.r_rep,
            radius_alignment=p.r_orient,
            radius_attraction=p.r_attract,
            noise_std=p.noise,
            perception_angle_deg=p.fov_deg,
            turning_rate_deg=p.turn_deg,
            rng=self.rng,
        )
        speed = np.full(n, p.v_prey)
        if self.hunting and p.flee_radius > 0.0:
            rel = self.xy - self.pred_xy[:, None]
            near = np.hypot(rel[0], rel[1]) < p.flee_radius
            if near.any():
                away = np.arctan2(rel[1], rel[0])
                lim = np.deg2rad(p.flee_turn_deg) * p.dt
                dh = np.clip(wrap_pi(away - self.theta), -lim, lim)
                theta_new = np.where(near, (self.theta + dh) % TWO_PI, theta_new)
                speed = np.where(near, speed * p.flee_boost, speed)
        self.theta = theta_new
        self.xy = self.xy + p.dt * speed * np.vstack((np.cos(theta_new), np.sin(theta_new)))

    # --------------------------------------------------------------- predator
    def _visible_prey(self, dist, rel_ang) -> np.ndarray:
        p = self.p
        return (dist < p.sight) & (np.abs(rel_ang) < np.deg2rad(p.fov_pred_deg) / 2.0)

    def _select_target(self, vis: np.ndarray, dist: np.ndarray) -> int:
        """Odd prey first; otherwise the most isolated visible prey."""
        odd_vis = vis & self.is_odd
        if odd_vis.any():
            return int(np.argmin(np.where(odd_vis, dist, np.inf)))
        x, y = self.xy
        d = np.hypot(x[None, :] - x[:, None], y[None, :] - y[:, None])
        np.fill_diagonal(d, np.inf)
        isolation = d.min(axis=1)
        score = np.where(vis, isolation - 0.15 * dist, -np.inf)
        return int(np.argmax(score))

    def _log(self, kind: str, x: float, y: float) -> None:
        self.events.append(dict(type=kind, x=float(x), y=float(y), t=self.t))
        del self.events[:-60]

    def _end_attack(self, outcome: str) -> None:
        self.stats[{"capture": "captures", "escape": "escapes", "abort": "aborts", "timeout": "timeouts"}[outcome]] += 1
        self.outcomes.append((self.t, outcome))
        self.target = -1
        self.state = RECOVER
        self._recover_steps = int(self.p.recover_time / self.p.dt)

    def _predator_turn_and_move(self, desired: float, speed: float) -> None:
        p = self.p
        lim = np.deg2rad(p.turn_pred_deg) * p.dt
        self.pred_theta = (self.pred_theta + np.clip(wrap_pi(desired - self.pred_theta), -lim, lim)) % TWO_PI
        self.pred_xy = self.pred_xy + p.dt * speed * np.array([np.cos(self.pred_theta), np.sin(self.pred_theta)])

    def _step_predator(self) -> None:
        p = self.p
        px, py = self.pred_xy
        rel = self.xy - self.pred_xy[:, None]
        dist = np.hypot(rel[0], rel[1])
        bearing = np.arctan2(rel[1], rel[0])
        rel_ang = wrap_pi(bearing - self.pred_theta)

        if self.state == SEARCH:
            # Head for the nearest prey (a hunter follows what it can see or smell)
            k = int(np.argmin(dist))
            desired = float(bearing[k])
            vis = self._visible_prey(dist, rel_ang)
            if vis.any():
                self.target = self._select_target(vis, dist)
                self.state = LOCK
                self.breaks = 0
                self._attack_steps = 0
                self._cone_sum, self._cone_n = 0.0, 0
                self._prev_cross[:] = False
                self.stats["attacks"] += 1
            self._predator_turn_and_move(desired, p.v_pred)

        elif self.state == LOCK:
            t = self.target
            self._attack_steps += 1
            odd_target = bool(self.is_odd[t])
            # (a) Look-alikes in the line of sight. A *new* one crossing it resets the lock.
            if not odd_target:
                cross = (np.abs(wrap_pi(bearing - bearing[t])) < np.deg2rad(p.lock_tol_deg)) & (dist < dist[t] + p.clutter_depth)
                cross[t] = False
                self._cone_sum += float(cross.sum())
                self._cone_n += 1
                new = cross & ~self._prev_cross
                self._prev_cross = cross
                if new.any():
                    self.breaks += 1
                    self.stats["breaks"] += 1
                    self._log("break", self.xy[0, t], self.xy[1, t])
                    if self.breaks >= p.max_breaks:
                        self._log("abort", px, py)
                        self._end_attack("abort")
                    else:
                        self.state, self._lost_steps = LOST, p.relock_steps
                    self._predator_turn_and_move(self.pred_theta, p.v_pred)
                    return
            # (b) Strike?
            if dist[t] < p.strike_radius:
                self.stats["strikes"] += 1
                if odd_target:
                    prob = 1.0  # unbreakable lock, no confusion
                else:
                    n_mean = self._cone_sum / max(1, self._cone_n)
                    prob = strike_probability(n_mean, p.confusion)
                if self.rng.random() < prob:
                    self._log("capture", self.xy[0, t], self.xy[1, t])
                    self._respawn(t)
                    self._end_attack("capture")
                else:
                    self._log("escape", self.xy[0, t], self.xy[1, t])
                    self._end_attack("escape")
                self._predator_turn_and_move(self.pred_theta, p.v_pred)
                return
            # (c) Give up?
            if self._attack_steps * p.dt > p.max_attack_time:
                self._end_attack("timeout")
                self._predator_turn_and_move(self.pred_theta, p.v_pred)
                return
            self._predator_turn_and_move(float(bearing[t]), p.v_pred)

        elif self.state == LOST:
            self._lost_steps -= 1
            if self._lost_steps <= 0:
                vis = self._visible_prey(dist, rel_ang)
                if vis.any():
                    self.target = self._select_target(vis, dist)
                    self._prev_cross[:] = False
                    self.state = LOCK
                else:
                    self._end_attack("abort")
            self._predator_turn_and_move(self.pred_theta, p.v_pred)

        else:  # RECOVER: carry on straight, then loop back (search)
            self._recover_steps -= 1
            if self._recover_steps <= 0:
                self.state = SEARCH
            self._predator_turn_and_move(self.pred_theta, p.v_pred)

    def _respawn(self, i: int) -> None:
        """Keep N constant: a new prey joins the school next to a random member."""
        j = int(self.rng.choice(np.delete(np.arange(self.p.n_prey), i)))
        self.xy[:, i] = self.xy[:, j] + self.rng.normal(0.0, 1.5, 2)
        self.theta[i] = self.theta[j]

    # ------------------------------------------------------------------- step
    def step(self) -> None:
        if self.hunting:
            self._step_predator()
        self._step_prey()
        self.t += 1

    # --------------------------------------------------------------- reporting
    @property
    def capture_probability(self) -> float:
        """Captures per attack so far (nan before the first attack)."""
        a = self.stats["attacks"] - (1 if self.state in (LOCK, LOST) else 0)
        return self.stats["captures"] / a if a > 0 else float("nan")


# ---------------------------------------------------------------------------
# 4. Numerical experiments
# ---------------------------------------------------------------------------
def run_trial(params: HuntParams, steps: int = 3000, seed: int = 0) -> dict:
    """One hunt of ``steps`` time steps. Returns counters and school metrics."""
    sim = ConfusionSim(params, seed=seed)
    metrics = []
    for k in range(steps):
        sim.step()
        if k % 20 == 0:
            metrics.append(sim.school_metrics())
    out = dict(sim.stats)
    done = out["attacks"] - (1 if sim.state in (LOCK, LOST) else 0)
    out["finished_attacks"] = done
    out["capture_prob"] = out["captures"] / done if done > 0 else float("nan")
    for key in ("polarization", "nn_distance", "lookalikes"):
        out[key] = float(np.mean([m[key] for m in metrics]))
    return out


def sweep(param: str, values, base: HuntParams | None = None, n_seeds: int = 8, steps: int = 3000) -> dict:
    """Pool ``n_seeds`` hunts for every value of ``param``.

    Returns arrays ``values``, ``capture_prob`` (captures / attacks, pooled over
    seeds), ``attacks`` and the mean school metrics.
    """
    base = base if base is not None else HuntParams()
    keys = ("capture_prob", "attacks", "captures", "breaks", "polarization", "nn_distance", "lookalikes")
    rows = {k: [] for k in keys}
    for v in values:
        runs = [run_trial(base.with_(**{param: v}), steps=steps, seed=s) for s in range(n_seeds)]
        att = sum(r["finished_attacks"] for r in runs)
        cap = sum(r["captures"] for r in runs)
        rows["capture_prob"].append(cap / att if att else float("nan"))
        rows["attacks"].append(att)
        rows["captures"].append(cap)
        rows["breaks"].append(sum(r["breaks"] for r in runs))
        for k in ("polarization", "nn_distance", "lookalikes"):
            rows[k].append(float(np.mean([r[k] for r in runs])))
    out = {k: np.array(v) for k, v in rows.items()}
    out["values"] = np.array(list(values))
    return out


if __name__ == "__main__":
    import time

    print("Max |difference| between couzin_equations and the vectorised step "
          f"(noise = 0): {check_against_original():.2e}")
    t0 = time.time()
    res = sweep("d_attract", [0.0, 2.0, 4.0, 8.0, 12.0], n_seeds=4, steps=2500)
    print(f"sweep took {time.time() - t0:.1f} s")
    print(f"{'d_attract':>9} {'P(capture)':>10} {'attacks':>8} {'look-alikes':>11} {'nn dist':>8}")
    for i, v in enumerate(res["values"]):
        print(f"{v:9.1f} {res['capture_prob'][i]:10.2f} {res['attacks'][i]:8d} "
              f"{res['lookalikes'][i]:11.1f} {res['nn_distance'][i]:8.2f}")
