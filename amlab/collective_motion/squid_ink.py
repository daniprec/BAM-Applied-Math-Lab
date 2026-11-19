"""A school of fish hunts one squid that can release ink.

This module turns the prototype ``experiments/squid_ink_flocking.py`` into a
model that can be measured: it adds a capture rule, a time limit, parameter
sweeps and a self-test. The physics is the prototype's.

The model
---------
* **Fish** (predators) follow Vicsek-style rules: alignment, cohesion, short-range
  separation and pursuit of the squid. Their heading turns at a limited rate and
  carries Gaussian noise.
* **Squid** (prey) cruises, flees from the nearest fish when it comes within the
  flight distance, and releases a compact bolus of ink when a fish is that close
  and at least ``reload_time`` has passed since the last release. The ink supply is
  **finite**: the ink sac does not refill within a hunt (in a cuttlefish it takes
  weeks), and each release spends the fraction ``use_fraction`` of what is left, so
  successive boluses get thinner until the sac is empty.
* **Ink** is a scalar field c on a periodic grid with a simple spreading rule. In
  every time step each cell passes the fraction ``spread`` of its ink to its four
  neighbours (``spread = 0.5``: half stays, half goes out) and every cell whose
  ink falls below ``threshold`` is reset to exactly 0. The cloud therefore
  spreads, thins out and finally disappears. ``thickness`` is the peak amount
  released: a thicker bolus spreads farther and lasts longer.
* **Vision.** The pursuit weight of a fish is multiplied by the transmission
  T = exp(-(a + kappa * mean(c along the sightline)) * distance) (Beer-Lambert law).
  ``a`` is the attenuation of clear water, which limits the visual range, and the ink
  adds ``kappa * c``. In a dense cloud T is close to 0 and the fish loses the
  squid and drifts with the school.
* **Capture** happens when a fish comes within ``capture_radius`` of the squid.
* **The squid's decision** (optional, off by default). When it releases ink the squid also
  commits, for ``jet_time``, to the heading "straight away from the nearest fish, turned by
  ``escape_angle`` to a random side" and swims ``jet_boost`` times faster than in flight.
  ``hide_time`` > 0 makes it hold still inside the cloud instead. With the defaults
  (angle 0, boost 1, no hiding) the squid behaves exactly as in the model without a decision.

Lengths are in body lengths (BL), time in arbitrary time units (t.u.). This is a
qualitative teaching model, not a fit to squid data.

Run from the repository root::

    python amlab/collective_motion/squid_ink.py     # self-test and a short sweep
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


def wrap_delta(delta: np.ndarray, length: float) -> np.ndarray:
    """Shortest periodic displacement on a square domain of side ``length``."""
    return (delta + length / 2) % length - length / 2


def wrap_pi(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class InkParams:
    """All parameters of the model (body lengths and time units)."""

    n_fish: int = 24
    domain: float = 40.0  # side of the periodic square
    grid: int = 100  # ink grid is grid x grid
    dt: float = 0.12
    # ink
    spread: float = 0.5  # fraction of a cell's ink passed to its 4 neighbours each step
    threshold: float = 0.002  # cells with less ink than this are reset to 0
    ink_steps: int = 3  # spreading steps per time step
    thickness: float = 3.0  # peak amount of ink in a fresh bolus
    bolus_width: float = 0.9  # Gaussian sigma of a fresh bolus (BL)
    kappa: float = 2.8  # opacity: optical depth per unit concentration per BL
    water: float = 0.05  # attenuation of clear water per BL (visual range 1/water = 20 BL)
    reload_time: float = 5.0  # minimum time between two releases
    use_fraction: float = 0.6  # share of the remaining ink spent per release (0 = unlimited ink)
    empty_below: float = 0.05  # the sac counts as empty when less than this share is left
    ink: bool = True  # False = the squid never releases ink
    # fish
    fish_speed: float = 0.70  # clearly faster than the squid (cruise 0.30, flight 0.45)
    fish_turn: float = 0.20  # max turn per step (rad)
    fish_noise: float = 0.035
    r_separate: float = 1.15
    r_school: float = 5.0
    w_align: float = 1.1
    w_cohere: float = 0.35
    w_separate: float = 1.4
    w_pursue: float = 1.9
    # squid
    squid_cruise: float = 0.30
    squid_flee: float = 0.45
    flight_distance: float = 8.0
    squid_turn: float = 0.33
    squid_noise: float = 0.045
    # outcome
    capture_radius: float = 0.8
    # ---- how the fish react to the ink, and where the hunt starts ----
    search: bool = True  # a fish that loses the squid heads for the place where it last saw it
    see_threshold: float = 0.2  # transmission above which a fish counts as "seeing" the squid
    search_gain: float = 0.8  # pursuit weight toward the last-known position
    search_radius: float = 1.5  # a fish gives up once it is this close to that place
    decoy: bool = False  # fish chase the densest ink blob (a decoy) instead of the squid
    decoy_range: float = 10.0
    decoy_min: float = 0.2  # blobs thinner than this are not mistaken for a squid
    random_start: bool = True  # random school position and squid heading in every hunt
    # ---- the squid's decision after a release ----
    escape_angle: float = 0.0  # degrees: after inking, leave the flight line by this angle (0 = straight away)
    jet_boost: float = 1.0  # speed factor of the escape jet (1 = no jet)
    jet_time: float = 2.0  # duration of the jet (t.u.)
    hide_time: float = 0.0  # t.u. the squid holds still inside its own cloud after a release (0 = never)


class SquidInkSimulation:
    """One hunt. Call :meth:`step` repeatedly; stop when ``captured`` is True."""

    def __init__(self, params: InkParams | None = None, seed: int = 4) -> None:
        p = self.p = params or InkParams()
        self.rng = np.random.default_rng(seed)
        self.dx = p.domain / p.grid
        if not 0.0 <= p.spread <= 1.0:
            raise ValueError("spread must lie in [0, 1]")
        if p.random_start:
            self.squid = self.rng.uniform(0, p.domain, 2)
            angle = self.rng.uniform(-np.pi, np.pi)
            centre = self.squid + self.rng.uniform(11.0, 16.0) * np.array([np.cos(angle), np.sin(angle)])
            self.squid_heading = self.rng.uniform(-np.pi, np.pi)
        else:
            centre = np.array([p.domain * 0.68, p.domain * 0.52])
            self.squid = np.array([p.domain * 0.28, p.domain * 0.42])
            self.squid_heading = -0.15
        self.fish = (centre + self.rng.normal(0, 2.6, (p.n_fish, 2))) % p.domain
        self.fish_heading = self.rng.uniform(-np.pi, np.pi, p.n_fish)
        self.last_seen = np.tile(self.squid, (p.n_fish, 1))  # place where each fish last saw the squid
        self.searching = np.ones(p.n_fish, bool)
        self.ink = np.zeros((p.grid, p.grid))
        self.transmission = np.ones(p.n_fish)
        self.cooldown = 0.0
        self.stock = 1.0  # share of a full ink sac that is left
        self.time = 0.0
        self.releases = 0
        self.jet_left = 0.0  # remaining jet time
        self.hide_left = 0.0  # remaining hiding time
        self.jet_heading = 0.0
        # a separate stream for the squid's coin toss, so that the fish noise is identical
        # in paired runs that differ only in the decision
        self.rng_squid = np.random.default_rng(seed + 7_000_003)
        self.captured = False
        self.capture_time = np.nan
        self.closest = float(np.linalg.norm(wrap_delta(self.fish - self.squid, p.domain), axis=1).min())
        self.release_log: list[tuple[float, float, float]] = []  # (time, x, y)
        coords = np.arange(p.grid) * self.dx
        self._gx, self._gy = coords[None, :], coords[:, None]

    # ---------------------------------------------------------------- ink
    def _sample_ink(self, points: np.ndarray) -> np.ndarray:
        """Bilinear sample of the periodic ink field at continuous positions."""
        g = self.p.grid
        gx = (points[..., 0] % self.p.domain) / self.dx
        gy = (points[..., 1] % self.p.domain) / self.dx
        x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
        fx, fy = gx - x0, gy - y0
        x1, y1 = (x0 + 1) % g, (y0 + 1) % g
        x0, y0 = x0 % g, y0 % g
        return (
            self.ink[y0, x0] * (1 - fx) * (1 - fy)
            + self.ink[y0, x1] * fx * (1 - fy)
            + self.ink[y1, x0] * (1 - fx) * fy
            + self.ink[y1, x1] * fx * fy
        )

    def _release_ink(self) -> None:
        """Add a Gaussian bolus at the squid's position.

        The peak is ``thickness`` times the share of ink left in the sac, which then
        falls by the factor ``1 - use_fraction``.
        """
        p = self.p
        rx = wrap_delta(self._gx - self.squid[0], p.domain)
        ry = wrap_delta(self._gy - self.squid[1], p.domain)
        self.ink += p.thickness * self.stock * np.exp(-(rx**2 + ry**2) / (2 * p.bolus_width**2))
        self.stock *= 1.0 - p.use_fraction
        self.releases += 1
        self.release_log.append((self.time, self.squid[0], self.squid[1]))

    def _spread_ink(self) -> None:
        """Spreading rule: pass ``spread`` of each cell's ink to its 4 neighbours, then cut off.

        Each cell keeps ``1 - spread`` of its ink and gives ``spread / 4`` to each
        neighbour. The total ink is conserved until the cut-off: cells below
        ``threshold`` are reset to exactly 0 and their ink is lost.
        """
        p = self.p
        c = self.ink
        for _ in range(p.ink_steps):
            nb = np.roll(c, 1, 0) + np.roll(c, -1, 0) + np.roll(c, 1, 1) + np.roll(c, -1, 1)
            c = (1.0 - p.spread) * c + (p.spread / 4.0) * nb
            c[c < p.threshold] = 0.0
        self.ink = c

    # ------------------------------------------------------------- vision
    def sight_transmission(self) -> np.ndarray:
        """Beer-Lambert transmission along each fish-to-squid sightline."""
        p = self.p
        rays = wrap_delta(self.squid[None, :] - self.fish, p.domain)
        distance = np.linalg.norm(rays, axis=1)
        if not self.ink.any():
            return np.exp(-p.water * distance)
        n_samples = max(2, int(np.ceil(distance.max() / (self.dx * 1.5))))
        fractions = np.linspace(0.08, 0.92, n_samples)
        pts = self.fish[:, None, :] + rays[:, None, :] * fractions[None, :, None]
        mean_c = self._sample_ink(pts).mean(axis=1)
        return np.exp(-(p.water + p.kappa * mean_c) * distance)

    # --------------------------------------------------------------- fish
    def _targets(self, transmission):
        """Where each fish steers and with what weight.

        Original rule: toward the squid, with weight equal to the transmission.
        ``search``: a fish that has lost the squid goes to the place where it last saw it
        and gives up on arrival. ``decoy``: fish chase the densest ink blob in range.
        """
        p = self.p
        n = p.n_fish
        target = np.tile(self.squid, (n, 1))
        weight = transmission.copy()
        if p.search:
            seen = transmission > p.see_threshold
            self.last_seen[seen] = self.squid
            self.searching |= seen
            gap = np.linalg.norm(wrap_delta(self.last_seen - self.fish, p.domain), axis=1)
            self.searching &= ~(~seen & (gap < p.search_radius))
            target = np.where(seen[:, None], target, self.last_seen)
            weight = np.where(seen, transmission, np.where(self.searching, p.search_gain, 0.0))
        if p.decoy and self.ink.max() > p.decoy_min:
            k = min(60, self.ink.size)
            idx = np.argpartition(self.ink.ravel(), -k)[-k:]
            vals = self.ink.ravel()[idx]
            keep = vals > p.decoy_min
            idx, vals = idx[keep], vals[keep]
            by, bx = np.divmod(idx, p.grid)
            pos = np.column_stack((bx, by)) * self.dx
            d = np.linalg.norm(wrap_delta(pos[None] - self.fish[:, None], p.domain), axis=2)
            score = np.where(d < p.decoy_range, vals[None, :], -1.0)
            j = score.argmax(axis=1)
            has = score[np.arange(n), j] > 0
            target = np.where(has[:, None], pos[j], target)
            weight = np.where(has, 1.0, weight)
        return target, weight

    def _fish_update(self, transmission: np.ndarray) -> None:
        p = self.p
        delta = wrap_delta(self.fish[None, :, :] - self.fish[:, None, :], p.domain)
        dist = np.linalg.norm(delta, axis=2)
        not_self = ~np.eye(p.n_fish, dtype=bool)
        near = (dist < p.r_separate) & not_self
        school = (dist < p.r_school) & not_self
        n_school = np.maximum(school.sum(axis=1), 1)[:, None]
        away = -(near[..., None] * delta / np.maximum(dist[..., None], 1e-8)).sum(axis=1)
        cohesion = (school[..., None] * delta).sum(axis=1) / n_school
        ax = (school * np.cos(self.fish_heading)[None, :]).sum(axis=1)
        ay = (school * np.sin(self.fish_heading)[None, :]).sum(axis=1)
        alignment = np.column_stack((ax, ay)) / n_school
        target, weight = self._targets(transmission)
        pursuit = wrap_delta(target - self.fish, p.domain)
        pursuit /= np.maximum(np.linalg.norm(pursuit, axis=1)[:, None], 1e-8)
        v = p.w_align * alignment + p.w_cohere * cohesion + p.w_separate * away
        v += (p.w_pursue * weight)[:, None] * pursuit
        desired = np.arctan2(v[:, 1], v[:, 0])
        turn = np.clip(wrap_pi(desired - self.fish_heading), -p.fish_turn, p.fish_turn)
        turn += self.rng.normal(0.0, p.fish_noise, p.n_fish)
        self.fish_heading = (self.fish_heading + turn) % (2 * np.pi)

    # -------------------------------------------------------------- squid
    def _start_decision(self, away: float) -> None:
        """The squid's decision after a release: where to jet, and whether to hide first.

        ``away`` is the heading straight away from the nearest fish. The squid commits to
        the heading ``away`` +/- ``escape_angle`` (side chosen at random) for ``jet_time``.
        """
        p = self.p
        if p.escape_angle <= 0:
            side = 0.0
        else:
            side = self.rng_squid.choice((-1.0, 1.0))
        self.jet_heading = away + side * np.radians(p.escape_angle)
        self.jet_left = p.jet_time if (p.jet_boost > 1.0 or p.escape_angle > 0) else 0.0
        self.hide_left = p.hide_time

    def _squid_update(self) -> None:
        p = self.p
        rel = wrap_delta(self.fish - self.squid, p.domain)
        d = np.linalg.norm(rel, axis=1)
        k = int(np.argmin(d))
        self.closest = float(d[k])
        threatened = d[k] < p.flight_distance
        speed = p.squid_cruise
        if self.hide_left > 0:  # inside the cloud: hold still
            self.hide_left -= p.dt
            speed = 0.0
        elif self.jet_left > 0:  # committed escape jet
            turn = np.clip(wrap_pi(self.jet_heading - self.squid_heading), -p.squid_turn, p.squid_turn)
            self.squid_heading = (self.squid_heading + turn) % (2 * np.pi)
            self.jet_left -= p.dt
            speed = p.squid_flee * p.jet_boost
        elif threatened:
            desired = np.arctan2(-rel[k, 1], -rel[k, 0])
            turn = np.clip(wrap_pi(desired - self.squid_heading), -p.squid_turn, p.squid_turn)
            self.squid_heading = (self.squid_heading + turn) % (2 * np.pi)
            if p.ink and self.cooldown <= 0 and self.stock >= p.empty_below:
                self._release_ink()
                self.cooldown = p.reload_time
                self._start_decision(desired)
            speed = p.squid_flee
        else:
            self.squid_heading = (self.squid_heading + self.rng.normal(0, p.squid_noise)) % (2 * np.pi)
        self.squid = (
            self.squid + speed * p.dt * np.array([np.cos(self.squid_heading), np.sin(self.squid_heading)])
        ) % p.domain

    # --------------------------------------------------------------- step
    def step(self) -> None:
        p = self.p
        self._spread_ink()
        self.cooldown = max(0.0, self.cooldown - p.dt)
        self.transmission = self.sight_transmission()
        self._fish_update(self.transmission)
        vel = np.column_stack((np.cos(self.fish_heading), np.sin(self.fish_heading)))
        self.fish = (self.fish + p.fish_speed * p.dt * vel) % p.domain
        self._squid_update()
        self.time += p.dt
        d = np.linalg.norm(wrap_delta(self.fish - self.squid, p.domain), axis=1).min()
        self.closest = float(d)
        if d < p.capture_radius and not self.captured:
            self.captured = True
            self.capture_time = self.time

    # ------------------------------------------------------------ metrics
    def school_metrics(self) -> dict[str, float]:
        """Polarisation and mean nearest-neighbour distance of the school."""
        pol = float(np.hypot(np.cos(self.fish_heading).mean(), np.sin(self.fish_heading).mean()))
        delta = wrap_delta(self.fish[None] - self.fish[:, None], self.p.domain)
        d = np.linalg.norm(delta, axis=2) + np.eye(len(self.fish)) * 1e9
        return {"polarisation": pol, "nn_distance": float(d.min(axis=1).mean())}


# ------------------------------------------------------------ experiments
def hunt(params: InkParams, seed: int, t_max: float = 250.0) -> dict[str, float]:
    """Run one hunt until capture or ``t_max``. Return the outcome."""
    sim = SquidInkSimulation(params, seed=seed)
    n_steps = int(round(t_max / params.dt))
    for _ in range(n_steps):
        sim.step()
        if sim.captured:
            break
    return {
        "captured": float(sim.captured),
        "time": sim.capture_time if sim.captured else t_max,
        "releases": float(sim.releases),
    }


def sweep(
    name: str, values, base: InkParams | None = None, n_seeds: int = 10, t_max: float = 250.0
) -> dict[str, np.ndarray]:
    """Vary one parameter. Return the capture probability, its counts and the mean ink use.

    ``capture_prob`` is the fraction of ``n_seeds`` hunts in which the squid was
    caught within ``t_max``; ``k`` and ``n`` are the counts behind it. ``mean_time``
    is the mean time to capture (``t_max`` when the squid was not caught) with its
    standard error ``sem_time``. The *delay* gained by the ink is read from it.
    """
    base = base or InkParams()
    out = {
        "values": np.asarray(values, float), "k": [], "n": [], "capture_prob": [],
        "mean_time": [], "sem_time": [], "releases": [],
    }  # fmt: skip
    for v in values:
        pv = replace(base, **{name: type(getattr(base, name))(v)})
        res = [hunt(pv, seed, t_max) for seed in range(n_seeds)]
        cap = np.array([r["captured"] for r in res])
        out["k"].append(cap.sum())
        out["n"].append(len(cap))
        out["capture_prob"].append(cap.mean())
        times = np.array([r["time"] for r in res])
        out["mean_time"].append(times.mean())
        out["sem_time"].append(times.std(ddof=1) / np.sqrt(len(times)) if len(times) > 1 else 0.0)
        out["releases"].append(np.mean([r["releases"] for r in res]))
    return {k: np.asarray(v) for k, v in out.items()}


# --------------------------------------------------------------- self-test
def self_test() -> None:
    """Checks that must hold. Raises AssertionError if one fails."""
    # 1. same seed, same hunt
    a, b = SquidInkSimulation(seed=1), SquidInkSimulation(seed=1)
    for _ in range(200):
        a.step(); b.step()
    assert np.allclose(a.fish, b.fish) and np.allclose(a.ink, b.ink)

    # 2. the spreading rule conserves ink when there is no cut-off, never makes it
    #    negative, spreads it symmetrically, and the cut-off makes the cloud vanish
    s = SquidInkSimulation(InkParams(threshold=0.0), seed=2)
    s._release_ink()
    m0 = s.ink.sum()
    for _ in range(300):
        s._spread_ink()
    assert abs(s.ink.sum() / m0 - 1) < 1e-9, "spreading must conserve ink without a cut-off"
    assert s.ink.min() >= 0
    s = SquidInkSimulation(InkParams(), seed=2)
    s._release_ink()
    s.squid = np.array([20.0, 20.0])
    s.ink[:] = 0.0
    s.ink[50, 50] = 1.0
    s._spread_ink()  # 3 steps; the ink must have moved symmetrically to the neighbours
    assert np.allclose(s.ink, s.ink.T), "a point source must spread symmetrically in x and y"
    for _ in range(4000):
        s._spread_ink()
    assert not s.ink.any(), "the cut-off must make the cloud disappear"
    s = SquidInkSimulation(InkParams(), seed=2)
    s._release_ink()
    assert s.ink.sum() > 0
    peak_before = s.ink.max()
    s._spread_ink()
    assert s.ink.max() < peak_before, "spreading must dilute the peak"

    # 3. in clear water the transmission is the water term alone; ink on the sightline lowers it
    s = SquidInkSimulation(seed=3)
    d = np.linalg.norm(wrap_delta(s.squid[None, :] - s.fish, s.p.domain), axis=1)
    clear = s.sight_transmission()
    assert np.allclose(clear, np.exp(-s.p.water * d))
    s._release_ink()
    assert np.all(s.sight_transmission() <= clear + 1e-12) and s.sight_transmission().min() < clear.min()

    # 4. transmission falls as the opacity kappa grows
    t = []
    for kappa in (0.5, 2.8, 8.0):
        s = SquidInkSimulation(InkParams(kappa=kappa), seed=3)
        s._release_ink()
        t.append(s.sight_transmission().mean())
    assert t[0] > t[1] > t[2]

    # 5. periodic wrap
    assert abs(wrap_delta(np.array([39.0]), 40.0)[0] + 1.0) < 1e-12

    # 6. the squid never releases twice inside the reload time
    s = SquidInkSimulation(seed=4)
    for _ in range(1500):
        s.step()
    times = [r[0] for r in s.release_log]
    assert all(b - a_ >= s.p.reload_time - 1e-9 for a_, b in zip(times, times[1:]))
    # 7. a finite ink supply: boluses get thinner and the sac runs empty
    s = SquidInkSimulation(InkParams(), seed=5)
    peaks = []
    for _ in range(3):
        s.ink[:] = 0.0
        s._release_ink()
        peaks.append(s.ink.max())
    assert peaks[0] > peaks[1] > peaks[2]
    assert abs(peaks[1] / peaks[0] - (1 - s.p.use_fraction)) < 0.02
    s = SquidInkSimulation(InkParams(), seed=4)
    for _ in range(2500):
        s.step()
        if s.captured:
            break
    n_max = int(np.ceil(np.log(s.p.empty_below) / np.log(1 - s.p.use_fraction)))
    assert s.releases <= n_max, "the squid cannot release more boluses than its sac allows"
    s = SquidInkSimulation(InkParams(use_fraction=0.0), seed=4)
    for _ in range(1500):
        s.step()
        if s.captured:
            break
    assert s.stock == 1.0, "use_fraction = 0 means unlimited ink"

    # 8. the decision system. Defaults leave the old behaviour untouched; with a jet the squid is
    #    faster after a release; the escape angle is exact and the side is a fair coin toss
    a = SquidInkSimulation(InkParams(), seed=6)
    b = SquidInkSimulation(InkParams(jet_boost=1.0, escape_angle=0.0), seed=6)
    for _ in range(300):
        a.step(); b.step()
    assert np.allclose(a.squid, b.squid) and np.allclose(a.fish, b.fish), "defaults must be the old model"
    s = SquidInkSimulation(InkParams(jet_boost=2.0, jet_time=2.0, escape_angle=90.0), seed=7)
    away = 0.3
    sides = []
    for _ in range(200):
        s._start_decision(away)
        sides.append(np.sign(wrap_pi(s.jet_heading - away)))
        assert abs(abs(wrap_pi(s.jet_heading - away)) - np.pi / 2) < 1e-9
    assert 60 < sum(1 for x in sides if x > 0) < 140, "the side must be random"
    s = SquidInkSimulation(InkParams(jet_boost=2.0, jet_time=2.0), seed=8)
    steps_jet = 0
    for _ in range(400):
        before = s.squid.copy()
        jetting = s.jet_left > 0
        s.step()
        if jetting:
            steps_jet += 1
            assert abs(np.linalg.norm(wrap_delta(s.squid - before, s.p.domain)) - s.p.squid_flee * 2.0 * s.p.dt) < 1e-9
        if s.captured:
            break
    assert s.releases == 0 or steps_jet >= 1, "after a release the squid must jet"
    s = SquidInkSimulation(InkParams(hide_time=3.0), seed=8)
    s._start_decision(0.0)
    p0 = s.squid.copy()
    s._squid_update()
    assert np.allclose(s.squid, p0), "a hiding squid holds still"
    print("self-test passed")


if __name__ == "__main__":
    self_test()
    base = InkParams()
    for label, par in [("no ink", replace(base, ink=False)), ("ink", base)]:
        res = [hunt(par, seed, 250.0) for seed in range(6)]
        print(f"{label:7s} captured {int(sum(r['captured'] for r in res))}/6, "
              f"mean time to capture {np.mean([r['time'] for r in res]):.0f}, "
              f"mean ink releases {np.mean([r['releases'] for r in res]):.1f}")
