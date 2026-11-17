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
def squid(seeds: int, plot_only: bool) -> None:
    """Time to capture of the squid against opacity, thickness, ink use and school size."""
    keys = ["values", "k", "n", "capture_prob", "mean_time", "sem_time", "releases"]
    t_max = 250.0
    base = si.InkParams()
    out = DATA / "squid_ink_sweeps.csv"
    if not plot_only:
        runs = {
            "noink": ("n_fish", [base.n_fish], si.InkParams(ink=False)),
            "unlimited": ("n_fish", [base.n_fish], si.InkParams(use_fraction=0.0)),
            "opacity": ("kappa", [0.0, 0.7, 1.4, 2.8, 5.6, 8.0], base),
            "thickness": ("thickness", [1.0, 2.0, 3.0, 6.0, 10.0], base),
            "inkuse": ("use_fraction", [0.2, 0.4, 0.6, 0.8], base),
            "school": ("n_fish", [8, 16, 24, 40, 60], base),
        }
        DATA.mkdir(exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sweep", *keys])
            for name, (param, values, b) in runs.items():
                print(f"squid: {name}")
                res = si.sweep(param, values, base=b, n_seeds=seeds, t_max=t_max)
                for row in zip(*[res[k] for k in keys]):
                    w.writerow([name, *[f"{v:.6g}" for v in row]])
    with open(out) as f:
        rows = list(csv.DictReader(f))

    def load_sweep(name):
        sel = [r for r in rows if r["sweep"] == name]
        return {k: np.array([float(r[k]) for r in sel]) for k in keys}

    r0, ru = load_sweep("noink"), load_sweep("unlimited")

    P = apply_theme("light")
    IMG.mkdir(exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 6.4), constrained_layout=True, sharey=True)
    top = 0.0
    for ax, name, xl in [
        (axs[0, 0], "opacity", "ink opacity \u03ba"),
        (axs[0, 1], "thickness", "ink thickness (peak of the first bolus)"),
        (axs[1, 0], "inkuse", "share of the remaining ink used per release"),
        (axs[1, 1], "school", "school size"),
    ]:
        r = load_sweep(name)
        ax.fill_between(
            r["values"], r["mean_time"] - 1.96 * r["sem_time"], r["mean_time"] + 1.96 * r["sem_time"],
            color=P["accent"], alpha=0.18, lw=0,
        )  # fmt: skip
        ax.plot(r["values"], r["mean_time"], "o-", color=P["prey"], lw=2.2, ms=6, label="squid with ink")
        ax.axhline(r0["mean_time"][0], color=P["bad"], lw=1.6, ls="--", label="no ink")
        top = max(top, float((r["mean_time"] + 1.96 * r["sem_time"]).max()))
        ax.set_xlabel(xl)
    for ax in axs.flat:
        ax.set_ylim(0, 1.08 * top)
    for ax in axs[:, 0]:
        ax.set_ylabel("mean time to capture (t.u.)")
    k_u, n_u = int(ru["k"][0]), int(ru["n"][0])
    axs[1, 0].text(
        0.98, 0.04, f"Unlimited ink (not shown): only {k_u} of {n_u} squid\ncaught within 250 t.u.",
        transform=axs[1, 0].transAxes, ha="right", va="bottom", fontsize=8, color=P["muted"],
    )  # fmt: skip
    axs[1, 0].legend(loc="upper right", fontsize=8)
    fig.savefig(IMG / "squid_ink_sweeps.png", dpi=160)
    print("wrote", IMG / "squid_ink_sweeps.png")


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
