"""Experiments for the Dictyostelium lesson (``dicty.qmd``).

Run from the repository root::

    python amlab/collective_motion/run_dicty_experiments.py run --seeds 20     # about 15 min on 2 cores
    python amlab/collective_motion/run_dicty_experiments.py run --seeds 3      # quick look
    python amlab/collective_motion/run_dicty_experiments.py plot               # redraw from the saved CSV files

Writes ``data/dicty_runs.csv`` (every single run), ``data/dicty_front_speed.csv`` and
``img/dicty_experiment.png``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

sys.path.append(".")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from amlab.collective_motion.dicty import DictyParams, front_speed, nagumo_speed, run
from amlab.collective_motion.ie_style import apply_theme

DATA, IMG = Path("data"), Path("img")
T_MAX = 1500.0
DENSITIES = [0.05, 0.10, 0.20, 0.30, 0.45, 0.60]
THETAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
D_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def _job(args):
    sweep, value, chem, seed = args
    base = DictyParams(chemotaxis=chem)
    p = replace(base, density=value) if sweep == "density" else replace(base, theta=value)
    r = run(p, seed, T_MAX)
    return (sweep, value, int(chem), seed, r["t_agg"], r["aggregated"], r["share"], r["clusters"], r["fired"])


def do_run(seeds: int) -> None:
    DATA.mkdir(exist_ok=True)
    jobs = [("density", d, c, s) for d in DENSITIES for c in (True, False) for s in range(seeds)]
    jobs += [("theta", th, True, s) for th in THETAS for s in range(seeds)]
    with Pool(2) as pool, open(DATA / "dicty_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sweep", "value", "chemotaxis", "seed", "t_agg", "aggregated", "share", "clusters", "fired"])
        for i, row in enumerate(pool.imap(_job, jobs, chunksize=2)):
            w.writerow([row[0], row[1], row[2], row[3]] + [f"{v:.6g}" for v in row[4:]])
            if i % 20 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
                f.flush()
    with open(DATA / "dicty_front_speed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["D", "k", "theta", "c_measured", "c_exact"])
        for D in D_VALUES:
            w.writerow([D, 1.0, 0.15, f"{front_speed(D):.5f}", f"{nagumo_speed(D, 1.0, 0.15):.5f}"])


def load_runs():
    with open(DATA / "dicty_runs.csv") as f:
        return list(csv.DictReader(f))


def mean_ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def do_plot() -> None:
    rows = load_runs()
    P = apply_theme("light")
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.7), constrained_layout=True)
    with open(DATA / "dicty_front_speed.csv") as f:
        fs = list(csv.DictReader(f))
    D = np.array([float(r["D"]) for r in fs])
    ax = axs[0]
    ax.plot(np.sqrt(D), [float(r["c_exact"]) for r in fs], "-", color=P["muted"], lw=2, label="exact $\\sqrt{Dk/2}\\,(1-2\\theta)$")
    ax.plot(np.sqrt(D), [float(r["c_measured"]) for r in fs], "o", color=P["prey"], ms=7, label="simulation")
    ax.set_xlabel("$\\sqrt{D}$")
    ax.set_ylabel("front speed $c$")
    ax.set_title("Front speed", loc="left")
    ax.legend(fontsize=8)
    for ax, key, yl, title in [(axs[1], "share", "share of cells in crowded blocks", "Aggregation"),
                               (axs[2], "clusters", "number of clusters at t = 1500", "Clusters")]:
        for chem, col, mk, lab in [(1, P["prey"], "o-", "chemotaxis on"), (0, P["muted"], "s--", "chemotaxis off (control)")]:
            xs, ms, cs = [], [], []
            for d in DENSITIES:
                v = [float(r[key]) for r in rows if r["sweep"] == "density" and int(r["chemotaxis"]) == chem and float(r["value"]) == d]
                m, c = mean_ci(v)
                xs.append(d), ms.append(m), cs.append(c)
            ax.errorbar(xs, ms, yerr=cs, fmt=mk, color=col, lw=2, ms=6, capsize=3, label=lab)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(DENSITIES, [str(d) for d in DENSITIES], fontsize=8)
        ax.set_xlabel("cell density (cells per grid site)")
        ax.set_ylabel(yl)
        ax.set_title(title, loc="left")
    axs[1].legend(fontsize=8)
    ax = axs[3]
    ms, cs = [], []
    for th in THETAS:
        m, c = mean_ci([float(r["t_agg"]) for r in rows if r["sweep"] == "theta" and float(r["value"]) == th])
        ms.append(m), cs.append(c)
    ax.errorbar(THETAS, ms, yerr=cs, fmt="o-", color=P["prey"], lw=2, ms=6, capsize=3)
    ax.set_xlabel("excitation threshold $\\theta$ (higher = less excitable)")
    ax.set_ylabel("time to 50 % aggregated")
    ax.set_title("Excitability", loc="left")
    IMG.mkdir(exist_ok=True)
    fig.savefig(IMG / "dicty_experiment.png", dpi=160)
    print("wrote", IMG / "dicty_experiment.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["run", "plot"])
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    if a.what == "run":
        do_run(a.seeds)
    do_plot()
