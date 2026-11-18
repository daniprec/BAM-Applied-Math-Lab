"""Numerical experiments for the confusion effect, the crowd exit and the squid ink.

Runs the parameter sweeps used in ``confusion-effect.qmd``, ``crowd-exit.qmd`` and
``squid-ink.qmd``,
saves the numbers as CSV files in ``data/`` and the figures in ``img/``.

    python amlab/collective_motion/run_experiments.py confusion          # about 2 min
    python amlab/collective_motion/run_experiments.py crowd              # about 15 min
    python amlab/collective_motion/run_experiments.py squid --seeds 40   # about 5 min
    python amlab/collective_motion/run_experiments.py all --seeds 3      # fewer repeats, faster
    python amlab/collective_motion/run_experiments.py crowd --plot-only  # redraw from the saved CSV files

Every number is a mean over independent random runs (seeds). Use more seeds
(``--seeds 10``) for smoother curves.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.append(".")  # run from the repository root, like the other lab scripts

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from amlab.collective_motion import couzin_predator as cp
from amlab.collective_motion import crowd_exit as ce
from amlab.collective_motion import squid_ink as si
from amlab.collective_motion.ie_style import apply_theme

DATA, IMG = Path("data"), Path("img")


def save_csv(name: str, res: dict, keys: list[str]) -> None:
    DATA.mkdir(exist_ok=True)
    with open(DATA / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for row in zip(*[res[k] for k in keys]):
            w.writerow([f"{v:.6g}" for v in row])


def load_csv(name: str) -> dict:
    with open(DATA / name) as f:
        r = list(csv.reader(f))
    return {k: np.array([float(row[i]) for row in r[1:]]) for i, k in enumerate(r[0])}


def wilson(k, n, z=1.96):
    """95 % Wilson interval for a proportion k / n."""
    n = np.maximum(n, 1)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


# ---------------------------------------------------------------------------
# Confusion effect
# ---------------------------------------------------------------------------
def confusion(seeds: int, plot_only: bool) -> None:
    keys = [
        "values",
        "capture_prob",
        "attacks",
        "captures",
        "lookalikes",
        "nn_distance",
    ]
    if not plot_only:
        steps = 3000
        print("confusion: cohesion sweep")
        r1 = cp.sweep(
            "d_attract", [0, 2, 4, 6, 8, 10, 12, 14], n_seeds=seeds, steps=steps
        )
        save_csv("confusion_cohesion.csv", r1, keys)
        print("confusion: odd prey sweep")
        r2 = cp.sweep(
            "n_odd", [0, 1, 2, 3], base=cp.HuntParams(), n_seeds=seeds, steps=steps
        )
        save_csv("confusion_odd.csv", r2, keys)
        print("confusion: group size sweep")
        r3 = cp.sweep(
            "n_prey",
            [10, 20, 40, 80, 120],
            base=cp.HuntParams(),
            n_seeds=seeds,
            steps=steps,
        )
        save_csv("confusion_groupsize.csv", r3, keys)
    r1, r2, r3 = (
        load_csv(f"confusion_{n}.csv") for n in ("cohesion", "odd", "groupsize")
    )

    P = apply_theme("light")
    IMG.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, r, xl, title in [
        (axs[0], r1, "cohesion: width of the zone of attraction", "Cohesion"),
        (axs[1], r2, "number of odd prey", "Oddity"),
        (axs[2], r3, "school size", "School size"),
    ]:
        lo, hi = wilson(r["captures"], r["attacks"])
        ax.fill_between(r["values"], lo, hi, color=P["accent"], alpha=0.18, lw=0)
        ax.plot(r["values"], r["capture_prob"], "o-", color=P["prey"], lw=2.2, ms=6)
        ax.set_xlabel(xl)
        if title == "Oddity":
            ax.set_xticks(r["values"])  # whole numbers of odd prey only
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left")
    axs[0].set_ylabel("captures per attack")
    fig.savefig(IMG / "confusion_sweeps.png", dpi=160)
    print("wrote", IMG / "confusion_sweeps.png")


# ---------------------------------------------------------------------------
# Crowd exit
# ---------------------------------------------------------------------------
def crowd(seeds: int, plot_only: bool) -> None:
    keys = [
        "values",
        "evac_time",
        "evac_time_std",
        "flow",
        "flow_std",
        "door_pressure",
        "door_pressure_std",
        "max_gap",
        "max_gap_std",
        "clog_fraction",
        "clog_fraction_std",
        "complete",
        "complete_std",
    ]
    if not plot_only:
        print("crowd: desired-speed sweep (faster is slower)")
        r1 = ce.sweep("v0", [0.8, 1.5, 2.5, 4.0, 6.0, 8.0], n_seeds=seeds)
        save_csv("crowd_speed.csv", r1, keys)
        print("crowd: column sweep at v0 = 4 m/s")
        base = ce.CrowdParams(v0=4.0)
        r0 = ce.sweep("v0", [4.0], base=base, n_seeds=seeds)
        r2 = ce.sweep(
            "obs_gap",
            [0.2, 0.5, 0.8, 1.2, 2.0],
            base=base.with_(obstacle=True),
            n_seeds=seeds,
        )
        # the first row of the file is the room without a column (gap = -1)
        merged = {k: np.concatenate([r0[k], r2[k]]) for k in keys}
        merged["values"] = np.concatenate([[-1.0], r2["values"]])
        save_csv("crowd_column.csv", merged, keys)
    r1, r2 = load_csv("crowd_speed.csv"), load_csv("crowd_column.csv")

    P = apply_theme("light")
    IMG.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    axs[0].errorbar(
        r1["values"],
        r1["evac_time"],
        r1["evac_time_std"],
        fmt="o-",
        color=P["prey"],
        lw=2.2,
        capsize=3,
    )
    axs[0].set_xlabel("desired speed (m/s)")
    axs[0].set_ylabel("evacuation time (s)")
    axs[0].set_title("Everybody out", loc="left")
    axs[1].errorbar(
        r1["values"],
        r1["flow"],
        r1["flow_std"],
        fmt="o-",
        color=P["prey"],
        lw=2.2,
        capsize=3,
    )
    axs[1].set_xlabel("desired speed (m/s)")
    axs[1].set_ylabel("flow (people / s)")
    axs[1].set_title("Flow through the door", loc="left")
    axs[2].semilogy(
        r1["values"],
        np.maximum(r1["door_pressure"], 1.0),
        "o-",
        color=P["predator"],
        lw=2.2,
    )
    axs[2].set_xlabel("desired speed (m/s)")
    axs[2].set_ylabel("pressure near the door (N/m)")
    axs[2].set_title("Crushing", loc="left")
    fig.savefig(IMG / "crowd_faster_is_slower.png", dpi=160)
    print("wrote", IMG / "crowd_faster_is_slower.png")

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    labels = ["none" if v < 0 else f"{v:.1f}" for v in r2["values"]]
    xs = np.arange(len(labels))
    stuck = (
        r2["complete"] < 1.0
    )  # at least one run did not finish within the time limit
    colours = [P["muted"]] + [P["prey"]] * (len(labels) - 1)
    for ax, key, ylabel in [
        (axs[0], "flow", "flow (people / s)"),
        (axs[1], "door_pressure", "pressure near the door (N/m)"),
    ]:
        bars = ax.bar(
            xs,
            np.nan_to_num(r2[key]),
            yerr=np.nan_to_num(r2[key + "_std"]),
            color=colours,
            capsize=3,
        )
        for b, s_ in zip(bars, stuck):
            if s_:
                b.set_hatch("///")
                b.set_edgecolor(P["predator"])
        ax.set_ylabel(ylabel)
        ax.set_xticks(xs, labels)
        ax.set_xlabel("distance from the door to the column (m)")
    axs[0].set_title("Flow through the door", loc="left")
    axs[1].set_title("Pressure near the door", loc="left")
    axs[0].text(
        0.02,
        0.97,
        "hatched: some runs did not finish",
        transform=axs[0].transAxes,
        va="top",
        fontsize=8,
        color=P["muted"],
    )
    fig.savefig(IMG / "crowd_column.png", dpi=160)
    print("wrote", IMG / "crowd_column.png")


# ---------------------------------------------------------------------------
# Squid ink
# ---------------------------------------------------------------------------
def _squid_hunt(job):
    sweep, value, params, seed = job
    r = si.hunt(params, seed, 250.0)
    return sweep, value, seed, r["time"], r["captured"], r["releases"]


def squid(seeds: int, plot_only: bool) -> None:
    """Squid ink: one long CSV with every hunt, then means, intervals and survival curves."""
    from dataclasses import replace
    from multiprocessing import Pool

    base = si.InkParams()
    out = DATA / "squid_ink_sweeps.csv"
    if not plot_only:
        runs = {
            "tactic": [  # value = index into TACTICS
                (0, replace(base, ink=False)),
                (1, replace(base, search=False)),
                (2, base),
                (3, replace(base, decoy=True)),
                (4, replace(base, use_fraction=0.0)),
            ],
            "opacity": [(v, replace(base, kappa=v)) for v in [0.0, 0.7, 1.4, 2.8, 5.6, 8.0]],
            "inkuse": [(v, replace(base, use_fraction=v)) for v in [0.2, 0.4, 0.6, 0.8]],
            "school": [(v, replace(base, n_fish=int(v))) for v in [8, 16, 24, 40, 60]],
            # same product kappa * thickness = 8.4: is the effect one dial or two?
            "product": [(k, replace(base, kappa=k, thickness=8.4 / k)) for k in [1.4, 2.8, 8.4]],
        }
        jobs = [(s, v, p, seed) for s, lst in runs.items() for v, p in lst for seed in range(seeds)]
        print(f"squid: {len(jobs)} hunts")
        with Pool() as pool:
            rows = pool.map(_squid_hunt, jobs, chunksize=8)
        DATA.mkdir(exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sweep", "value", "seed", "time", "captured", "releases"])
            w.writerows(rows)
    with open(out) as f:
        data = list(csv.DictReader(f))

    def group(sweep):
        """value -> array of capture times (250 where the squid was not caught)."""
        g = {}
        for r in data:
            if r["sweep"] == sweep:
                g.setdefault(float(r["value"]), []).append((float(r["time"]), float(r["captured"])))
        return {v: np.array(x) for v, x in sorted(g.items())}

    def mean_ci(x):
        t = x[:, 0]
        return t.mean(), 1.96 * t.std(ddof=1) / np.sqrt(len(t))

    P = apply_theme("light")
    IMG.mkdir(exist_ok=True)
    tactics = group("tactic")
    labels = ["no ink", "ink,\nblind fish", "ink,\nfish search", "decoy", "unlimited\nink"]
    cols = [P["bad"], "#FF7B31", P["prey"], P["accent"], "#888A8C"]
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.6), constrained_layout=True)
    ax = axs[0, 0]
    for k, (v, x) in enumerate(tactics.items()):
        m, ci = mean_ci(x)
        ax.bar(k, m, yerr=ci, color=cols[k], capsize=3, alpha=0.9)
        if x[:, 1].mean() < 1:
            ax.text(k, m + ci + 6, f"{int(x[:, 1].sum())}/{len(x)}\ncaught", ha="center", fontsize=7.5, color=P["muted"])
    ax.set_xticks(range(5), labels, fontsize=8)
    ax.set_ylabel("mean time to capture (t.u.)")
    ax.set_title("Tactics", loc="left")
    ref = mean_ci(tactics[0.0])[0]
    for ax, name, xl in [
        (axs[0, 1], "opacity", "ink opacity \u03ba"),
        (axs[1, 0], "inkuse", "share of the remaining ink used per release"),
        (axs[1, 1], "school", "school size"),
    ]:
        g = group(name)
        xs = np.array(list(g))
        ms, cis = np.array([mean_ci(x) for x in g.values()]).T
        ax.fill_between(xs, ms - cis, ms + cis, color=P["accent"], alpha=0.18, lw=0)
        ax.plot(xs, ms, "o-", color=P["prey"], lw=2.2, ms=6)
        ax.axhline(ref, color=P["bad"], lw=1.6, ls="--", label="no ink")
        ax.set_xlabel(xl)
        ax.set_ylabel("mean time to capture (t.u.)")
        ax.set_ylim(bottom=0)
    axs[0, 1].legend(loc="lower right", fontsize=8)
    fig.savefig(IMG / "squid_ink_sweeps.png", dpi=160)
    print("wrote", IMG / "squid_ink_sweeps.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    for k, (v, x) in enumerate(tactics.items()):
        t = np.sort(x[:, 0][x[:, 1] > 0])
        steps = np.concatenate(([0.0], t, [250.0]))
        free = np.concatenate(([1.0], 1 - np.arange(1, len(t) + 1) / len(x), [1 - len(t) / len(x)]))
        ax.step(steps, free, where="post", color=cols[k], lw=2.2, label=labels[k].replace("\n", " "))
    ax.set_xlabel("time (t.u.)")
    ax.set_ylabel("share of squid not yet caught")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    fig.savefig(IMG / "squid_ink_survival.png", dpi=160)
    print("wrote", IMG / "squid_ink_survival.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("which", choices=["confusion", "crowd", "squid", "all"])
    ap.add_argument(
        "--seeds", type=int, default=6, help="independent runs per parameter value"
    )
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw the figures from the saved CSV files",
    )
    a = ap.parse_args()
    if a.which in ("confusion", "all"):
        confusion(a.seeds, a.plot_only)
    if a.which in ("crowd", "all"):
        crowd(a.seeds, a.plot_only)
    if a.which in ("squid", "all"):
        squid(a.seeds, a.plot_only)
