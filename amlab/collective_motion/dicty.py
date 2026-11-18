"""Dictyostelium aggregation: amoebae that relay a cAMP signal and walk toward it.

One sentence: *amoebae pass a signal along and walk toward it, so they gather
without a leader.*

The model joins two ideas of the course: agents that move and interact locally
(Module 2) and an excitable reaction-diffusion field (Module 4). It is a
**teaching model**; the parameters are in model units, not in micrometres and
minutes, and are not fitted to data.

Field (x, y on an ``n x n`` grid with no-flux walls, grid spacing ``dx = 1``)
---------------------------------------------------------------------------
``u`` is the extracellular cAMP and ``v`` a slow recovery variable (the
refractory state of the relaying cells). The kinetics are of Barkley /
FitzHugh-Nagumo type, a cubic activator with a linear slow inhibitor::

    du/dt = D lap(u) + k g(x) u (1 - u) (u - theta - v) - mu u
    dv/dt = eps (u - v)

* ``D`` diffusion coefficient of cAMP,
* ``k`` reaction rate (secretion and relay),
* ``theta`` excitation threshold (the **excitability dial**: low = easy to excite),
* ``mu`` degradation of cAMP (phosphodiesterase),
* ``eps`` rate of the recovery variable (small = slow recovery, long refractory time),
* ``g(x)`` the **relay capacity** of the cells at x. It is the local cell density
  smoothed over a few grid sites and divided by ``rho_c``, then clipped to [0, 1]:
  where there are no cells the medium cannot relay, only degrade. This is a
  coarse-grained version of "a cell that sees cAMP above threshold secretes
  cAMP" (Martiel and Goldbeter 1987; Palsson and Cox 1996).

Spontaneous firing: every cell fires at random with a small rate
``fire_rate`` per cell and unit time, which adds a Gaussian bump of cAMP
(radius 3.5 sites). The bump grows into a wave only if the medium there has
recovered (v small) and has enough cells (g large). Dense places therefore
become pacemakers, as in the real aggregation.

Front speed. With ``v = 0``, ``mu = 0`` and ``g = 1`` the field equation is the
Nagumo equation, whose travelling front has the exact speed::

    c = sqrt(D k / 2) (1 - 2 theta)                                   (1)

so ``c`` grows like ``sqrt(D k)``: a length scale ``sqrt(D / k)`` (front width)
and a time scale ``1 / k`` can only be combined into a speed in one way.

Cells (positions in grid units, free to move on the continuous square)
----------------------------------------------------------------------
Each cell has a heading ``h`` and moves with a small speed ``v0`` (persistent
random walk, rotational diffusion ``d_rot``). It senses the **rate of change**
of cAMP at its own position, ``(u - u_prev) / dt``. While cAMP is *rising*
(rate above ``rise``), that is, at the front of a wave, the cell turns toward
``grad u`` and switches on **directed motion with speed** ``v_chem`` for a time
``t_chem`` (the cell keeps walking after the wave has passed). Behind the wave
cAMP falls and the cell ignores the gradient. Because ``grad u`` points toward
the source while cAMP rises, every wave pulls the cells one step toward its
source. The cells are points: there is no volume exclusion and no cell
adhesion, so clusters can become very dense.

Simplifications (see the lesson page): no cAMP-induced cAMP synthesis cycle of
the receptors (no adaptation), a coarse-grained relay instead of cell-level
signalling, no cell differentiation, point cells, no streams-by-adhesion.

Run from the repository root::

    python amlab/collective_motion/dicty.py      # self-test and a short sweep
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class DictyParams:
    """All parameters (model units: space in grid spacings, time in 1/k)."""

    n: int = 100  # grid is n x n (set ``ny`` to use a rectangle)
    ny: int = 0  # 0 means ny = n
    dt: float = 0.2
    # field
    D: float = 1.0
    k: float = 1.0
    theta: float = 0.15  # excitation threshold: the excitability dial
    mu: float = 0.005
    gamma: float = 1.2  # strength of the inhibition by v
    eps: float = 0.025
    # cells
    density: float = 0.30  # cells per grid site (n * n * density cells in all)
    rho_c: float = 0.06  # smoothed density at which relay capacity reaches 1
    v0: float = 0.02  # speed of the random walk
    d_rot: float = 0.05  # rotational diffusion of the heading (1 / time)
    v_chem: float = 0.3  # speed of directed motion after a front (chemotactic strength)
    t_chem: float = 6.0  # time the directed motion lasts after the last rise
    rise: float = 0.04  # rate of rise of cAMP that counts as "a front" (1 / time)
    fire_rate: float = 4e-5  # spontaneous firing rate per cell and time unit
    fire_radius: float = 3.5
    chemotaxis: bool = True  # False = control: cells ignore cAMP
    cells: bool = True  # False = field only, relay capacity g = 1 everywhere
    mean_every: int = 5  # density map is refreshed every this many steps

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n, self.ny or self.n)

    @property
    def n_cells(self) -> int:
        nx, ny = self.shape
        return int(round(self.density * nx * ny))


def stability_limit(p: DictyParams) -> float:
    """Largest time step of the explicit Euler scheme (diffusion limit dx^2 / (4 D))."""
    return 1.0 / (4.0 * p.D)


def _blur(a: np.ndarray, passes: int = 2) -> np.ndarray:
    """Smooth with the [1, 2, 1] / 4 kernel along both axes (edge values are mirrored)."""
    for _ in range(passes):
        for ax in (0, 1):
            s = np.swapaxes(a, 0, ax)
            b = s.copy() * 2.0
            b[1:] += s[:-1]
            b[:-1] += s[1:]
            b[0] += s[0]
            b[-1] += s[-1]
            a = np.swapaxes(b * 0.25, 0, ax)
    return a


class DictySimulation:
    """Cells and the excitable cAMP field. ``step()`` advances one time step ``dt``."""

    def __init__(self, params: DictyParams | None = None, seed: int = 1) -> None:
        self.p = params or DictyParams()
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        nx, ny = self.p.shape
        self.nx, self.ny = nx, ny
        self.time = 0.0
        self.steps = 0
        self.u = np.zeros((nx, ny))
        self.v = np.zeros((nx, ny))
        self.lap = np.zeros((nx, ny))
        nc = self.p.n_cells
        self.x = self.rng.uniform(0, nx, nc)
        self.y = self.rng.uniform(0, ny, nc)
        self.h = self.rng.uniform(-np.pi, np.pi, nc)
        self.timer = np.zeros(nc)  # time left of directed motion
        self.u_prev = np.zeros(nc)
        self.active = np.zeros(nc, dtype=bool)  # drawn "moving toward the wave"
        self.rho = np.zeros((nx, ny))
        self.gain = np.ones((nx, ny))
        self.n_fired = 0
        self._ix = np.zeros(nc, dtype=int)
        self._iy = np.zeros(nc, dtype=int)
        self._refresh_density()
        xs = np.arange(nx)[:, None]
        ys = np.arange(ny)[None, :]
        self._xs, self._ys = xs, ys

    # ---------------------------------------------------------------- cells
    def _sites(self):
        ix = np.minimum(self.x.astype(int), self.nx - 1)
        iy = np.minimum(self.y.astype(int), self.ny - 1)
        return ix, iy

    def _refresh_density(self) -> None:
        if not self.p.cells:
            self.gain[:] = 1.0
            return
        ix, iy = self._sites()
        count = np.bincount(ix * self.ny + iy, minlength=self.nx * self.ny).reshape(self.nx, self.ny)
        self.rho = _blur(count.astype(float), 2)
        self.gain = np.clip(self.rho / self.p.rho_c, 0.0, 1.0)

    def _fire(self) -> None:
        """Spontaneous firing: Poisson number of bumps, placed at random cells."""
        p = self.p
        if not p.cells or len(self.x) == 0 or p.fire_rate <= 0:
            return
        m = self.rng.poisson(p.fire_rate * len(self.x) * p.dt)
        for _ in range(m):
            c = self.rng.integers(len(self.x))
            cx, cy = self.x[c], self.y[c]
            bump = np.exp(-(((self._xs + 0.5 - cx) ** 2 + (self._ys + 0.5 - cy) ** 2) / (2 * p.fire_radius**2)))
            self.u += bump
            self.n_fired += 1
        if m:
            np.minimum(self.u, 1.0, out=self.u)

    def _move_cells(self) -> None:
        p = self.p
        nc = len(self.x)
        if nc == 0:
            return
        ix, iy = self._sites()
        u = self.u
        u_here = u[ix, iy]
        du = (u_here - self.u_prev) / p.dt
        self.u_prev = u_here.copy()
        # heading noise (persistent random walk)
        self.h += np.sqrt(2.0 * p.d_rot * p.dt) * self.rng.standard_normal(nc)
        speed = np.full(nc, p.v0)
        if p.chemotaxis:
            rising = du > p.rise
            ixm, ixp = np.maximum(ix - 1, 0), np.minimum(ix + 1, self.nx - 1)
            iym, iyp = np.maximum(iy - 1, 0), np.minimum(iy + 1, self.ny - 1)
            gx = u[ixp, iy] - u[ixm, iy]
            gy = u[ix, iyp] - u[ix, iym]
            ok = rising & (np.hypot(gx, gy) > 1e-6)
            self.h[ok] = np.arctan2(gy[ok], gx[ok])
            self.timer[rising] = p.t_chem
            self.active = self.timer > 0
            speed[self.active] = p.v_chem
            self.timer = np.maximum(self.timer - p.dt, 0.0)
        self.x += speed * np.cos(self.h) * p.dt
        self.y += speed * np.sin(self.h) * p.dt
        # reflecting walls
        for pos, hi, axis in ((self.x, self.nx, 0), (self.y, self.ny, 1)):
            lo_m = pos < 0
            hi_m = pos >= hi
            pos[lo_m] = -pos[lo_m]
            pos[hi_m] = 2 * hi - pos[hi_m] - 1e-9
            flip = lo_m | hi_m
            if axis == 0:
                self.h[flip] = np.pi - self.h[flip]
            else:
                self.h[flip] = -self.h[flip]

    # ---------------------------------------------------------------- field
    def _laplacian(self, a: np.ndarray) -> np.ndarray:
        lap = self.lap
        lap[:] = 0.0
        d = a[1:] - a[:-1]
        lap[:-1] += d
        lap[1:] -= d
        d = a[:, 1:] - a[:, :-1]
        lap[:, :-1] += d
        lap[:, 1:] -= d
        return lap

    def _step_field(self) -> None:
        p = self.p
        u, v = self.u, self.v
        lap = self._laplacian(u)
        react = p.k * self.gain * u * (1.0 - u) * (u - p.theta - p.gamma * v) - p.mu * u
        du = p.D * lap + react
        dv = p.eps * (u - v)
        self.u = u + p.dt * du
        self.v = v + p.dt * dv

    def step(self) -> None:
        p = self.p
        if self.steps % p.mean_every == 0:
            self._refresh_density()
        self._fire()
        self._step_field()
        if p.cells:
            self._move_cells()
        self.steps += 1
        self.time += p.dt

    def run(self, t: float) -> None:
        for _ in range(int(round(t / self.p.dt))):
            self.step()

    # -------------------------------------------------------------- metrics
    def aggregation(self, block: int = 5, factor: float = 4.0) -> dict[str, float]:
        """Share of cells in crowded blocks and number of crowded clusters.

        The square is cut into ``block x block`` blocks. A block is *crowded* if it
        holds more than ``factor`` times the mean number of cells per block. The
        aggregated share is the fraction of all cells in crowded blocks; a cluster is
        a group of crowded blocks that touch along an edge.
        """
        nx, ny = self.nx // block, self.ny // block
        ix, iy = self._sites()
        cnt = np.zeros((nx, ny))
        np.add.at(cnt, (np.minimum(ix // block, nx - 1), np.minimum(iy // block, ny - 1)), 1.0)
        mean = len(self.x) / (nx * ny)
        dense = cnt > factor * mean
        share = float(cnt[dense].sum() / max(len(self.x), 1))
        return {"share": share, "clusters": float(_count_components(dense))}

    def wave_fraction(self) -> float:
        """Fraction of the grid that is excited (u above 0.5)."""
        return float((self.u > 0.5).mean())

    def finite(self) -> bool:
        return bool(np.isfinite(self.u).all() and np.isfinite(self.v).all() and np.isfinite(self.x).all())


def _count_components(mask: np.ndarray) -> int:
    """Number of 4-connected groups of True cells (small flood fill)."""
    seen = np.zeros_like(mask, dtype=bool)
    n = 0
    nx, ny = mask.shape
    for i in range(nx):
        for j in range(ny):
            if mask[i, j] and not seen[i, j]:
                n += 1
                stack = [(i, j)]
                seen[i, j] = True
                while stack:
                    a, b = stack.pop()
                    for c, d in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                        if 0 <= c < nx and 0 <= d < ny and mask[c, d] and not seen[c, d]:
                            seen[c, d] = True
                            stack.append((c, d))
    return n


# ------------------------------------------------------------ experiments
def run(params: DictyParams, seed: int, t_max: float = 1500.0, share_goal: float = 0.5,
        sample: float = 20.0) -> dict[str, float]:
    """Run one experiment. Return the time to aggregate and the final state.

    ``t_agg`` is the first sampled time at which at least ``share_goal`` of the cells
    sit in crowded blocks, or ``t_max`` if that never happens (``aggregated`` is then 0).
    """
    sim = DictySimulation(params, seed=seed)
    per = int(round(sample / params.dt))
    t_agg, hit = t_max, 0.0
    m = sim.aggregation()
    for _ in range(int(round(t_max / sample))):
        for _ in range(per):
            sim.step()
        m = sim.aggregation()
        if not hit and m["share"] >= share_goal:
            t_agg, hit = sim.time, 1.0
            break
    # keep running to t_max so that the final cluster count is comparable across runs
    rest = int(round((t_max - sim.time) / params.dt))
    for _ in range(max(rest, 0)):
        sim.step()
    m = sim.aggregation()
    return {"t_agg": t_agg, "aggregated": hit, "share": m["share"], "clusters": m["clusters"], "fired": float(sim.n_fired)}


def sweep(name: str, values, base: DictyParams | None = None, n_seeds: int = 20, t_max: float = 1500.0):
    """Vary one parameter; return the mean over ``n_seeds`` runs for each value."""
    base = base or DictyParams()
    out = {"values": np.asarray(values, float), "t_agg": [], "aggregated": [], "share": [], "clusters": []}
    for v in values:
        pv = replace(base, **{name: type(getattr(base, name))(v)})
        res = [run(pv, s, t_max) for s in range(n_seeds)]
        for key in ("t_agg", "aggregated", "share", "clusters"):
            out[key].append(np.mean([r[key] for r in res]))
    return {k: np.asarray(v) for k, v in out.items()}


def front_position(u: np.ndarray, level: float = 0.5) -> float:
    """Position (grid units) of the right-most crossing of ``level`` along the middle row."""
    row = u[:, u.shape[1] // 2]
    idx = np.where(row >= level)[0]
    if len(idx) == 0:
        return float("nan")
    i = int(idx.max())
    if i + 1 >= len(row):
        return float(i)
    f = (row[i] - level) / (row[i] - row[i + 1] + 1e-12)
    return float(i + f)


def front_speed(D: float, k: float = 1.0, theta: float = 0.15, length: int = 400, dt: float | None = None,
                t0: float = 40.0, t1: float = 120.0) -> float:
    """Measured speed of a front in the pure Nagumo equation (v = 0, mu = 0, g = 1).

    A strip of length ``length`` and width 5 starts with u = 1 on the left; the front
    position is measured at two times and the speed is the slope.
    """
    p = DictyParams(n=length, ny=5, D=D, k=k, theta=theta, mu=0.0, eps=0.0, cells=False, density=0.0,
                    dt=dt or min(0.2, 0.5 * stability_limit(DictyParams(D=D))))
    sim = DictySimulation(p, seed=0)
    sim.u[:6, :] = 1.0
    sim.run(t0)
    a = front_position(sim.u)
    sim.run(t1 - t0)
    b = front_position(sim.u)
    return (b - a) / (t1 - t0)


def nagumo_speed(D: float, k: float, theta: float) -> float:
    """Exact front speed of u_t = D u_xx + k u (1 - u)(u - theta): sqrt(D k / 2) (1 - 2 theta)."""
    return float(np.sqrt(D * k / 2.0) * (1.0 - 2.0 * theta))


def wave_period(params: DictyParams, seed: int = 0, t_max: float = 1200.0) -> float:
    """Mean time between two excitations at the centre of the grid (nan if fewer than 3).

    A pacemaker (a bump at the centre) is not used: the first excitation is set by a
    bump and later ones by spontaneous firing, so this is the period of what the
    medium does on its own in the stated conditions.
    """
    sim = DictySimulation(params, seed=seed)
    cx, cy = sim.nx // 2, sim.ny // 2
    above = False
    times = []
    for _ in range(int(round(t_max / params.dt))):
        sim.step()
        a = sim.u[cx - 2 : cx + 3, cy - 2 : cy + 3].mean() > 0.5
        if a and not above:
            times.append(sim.time)
        above = a
    return float(np.mean(np.diff(times))) if len(times) >= 3 else float("nan")


# --------------------------------------------------------------- self-test
def self_test() -> None:
    """Checks that must hold. Raises AssertionError if one fails."""
    # 1. same seed, same run; different seed, different run
    a, b, c = (DictySimulation(seed=s) for s in (1, 1, 2))
    for _ in range(300):
        a.step(); b.step(); c.step()
    assert np.array_equal(a.u, b.u) and np.array_equal(a.x, b.x), "same seed must give the same run"
    assert not np.array_equal(a.x, c.x), "different seeds must differ"

    # 2. stability of the explicit scheme: dt D / dx^2 must be below 1/4, and a run at
    #    the limit stays bounded while one clearly above it blows up
    p = DictyParams()
    assert p.dt <= stability_limit(p), "default time step violates dt <= dx^2 / (4 D)"
    for D, ok in ((1.0, True), (2.0, False)):
        q = DictyParams(n=40, D=D, dt=0.2, density=0.3, fire_rate=0.0, cells=False)
        s = DictySimulation(q, seed=0)
        s.u[18:22, 18:22] = 1.0
        with np.errstate(all="ignore"):
            for _ in range(400):
                s.step()
        bounded = np.isfinite(s.u).all() and np.abs(s.u).max() < 2.0
        assert bounded == ok, f"dt D/dx^2 = {0.2 * D:.2f}: expected stable={ok}, got {bounded}"

    # 3. no NaN, fields stay in the physical range, cells stay in the box, over a long run
    s = DictySimulation(DictyParams(), seed=3)
    s.run(400.0)
    assert s.finite()
    assert -1e-6 <= s.u.min() and s.u.max() <= 1.0 + 1e-6, "u must stay in [0, 1]"
    assert s.v.min() >= -1e-6 and s.v.max() <= 1.0 + 1e-6
    assert s.x.min() >= 0 and s.x.max() < s.nx and s.y.min() >= 0 and s.y.max() < s.ny
    assert s.n_fired > 0, "cells must fire spontaneously"

    # 4. diffusion alone conserves the total (no-flux walls)
    q = DictyParams(n=30, k=0.0, mu=0.0, eps=0.0, cells=False, density=0.0)
    s = DictySimulation(q, seed=0)
    s.u[10:14, 10:14] = 1.0
    m0 = s.u.sum()
    s.run(60.0)
    assert abs(s.u.sum() / m0 - 1) < 1e-9, "diffusion with no-flux walls must conserve u"

    # 5. front speed against the exact Nagumo speed sqrt(D k / 2) (1 - 2 theta)
    for D in (0.5, 1.0, 2.0, 4.0):
        c, c_exact = front_speed(D), nagumo_speed(D, 1.0, 0.15)
        assert abs(c / c_exact - 1) < 0.08, f"D={D}: measured {c:.3f}, exact {c_exact:.3f}"
    ratio = front_speed(4.0) / front_speed(1.0)
    assert abs(ratio - 2.0) < 0.1, f"speed must double when D is multiplied by 4 (got {ratio:.2f})"

    # 6. refractoriness: a bump at the centre of a recovering medium (80 time units after the
    #    first wave) does not start a new wave, the same bump after full recovery does
    q = DictyParams(n=60, ny=60, cells=False, density=0.0, fire_rate=0.0)
    gx, gy = np.meshgrid(np.arange(60), np.arange(60), indexing="ij")
    disk = (gx - 30) ** 2 + (gy - 30) ** 2 < 144
    area = {}
    for wait in (80.0, 200.0):
        s = DictySimulation(q, seed=0)
        s.u[25:35, 25:35] = 1.0
        s.run(wait)
        s.u[25:35, 25:35] = 1.0
        s.run(15.0)
        area[wait] = int((s.u[disk] > 0.5).sum())
    assert area[200.0] > 100 and area[80.0] < 0.1 * area[200.0], f"refractory period not seen: {area}"
    # 7. chemotaxis moves cells toward a source: a pulse that spreads from the centre
    #    pulls cells in
    q = DictyParams(n=60, ny=60, density=0.3, fire_rate=0.0, v0=0.0)
    s = DictySimulation(q, seed=0)
    r0 = np.hypot(s.x - 30, s.y - 30).mean()
    for _ in range(3):
        s.u[28:32, 28:32] = 1.0
        s.run(60.0)
    r1 = np.hypot(s.x - 30, s.y - 30).mean()
    assert r1 < r0, f"cells must move toward the wave source ({r0:.1f} -> {r1:.1f})"
    q = replace(q, chemotaxis=False)
    s = DictySimulation(q, seed=0)
    for _ in range(3):
        s.u[28:32, 28:32] = 1.0
        s.run(60.0)
    assert abs(np.hypot(s.x - 30, s.y - 30).mean() - r0) < 1.0, "control cells must not move toward the source"
    print("self-test passed")


if __name__ == "__main__":
    import time

    self_test()
    t = time.time()
    for ch in (False, True):
        res = [run(replace(DictyParams(), chemotaxis=ch), s, t_max=600.0) for s in range(3)]
        print(f"chemotaxis {ch!s:5}: share {np.mean([r['share'] for r in res]):.2f}, "
              f"clusters {np.mean([r['clusters'] for r in res]):.1f}")
    print(f"{time.time() - t:.0f} s")
