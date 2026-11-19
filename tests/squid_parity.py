"""Parity test, Python side: the same four settings as ``tests/squid_parity.js``.

Run from the repository root::

    python tests/squid_parity.py [seeds]        # default 40

The two programs use different random-number generators, so means agree only
statistically (tolerance about 5 t.u. with 40 seeds or more).
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from amlab.collective_motion.squid_ink import InkParams, hunt  # noqa: E402

T_MAX = 250.0
base = InkParams()
# keep in sync with SETTINGS in squid_parity.js
SETTINGS = [
    ("no ink", replace(base, ink=False)),
    ("ink, searching fish", base),
    ("ink, searching fish, jet x2", replace(base, jet_boost=2.0)),
    ("ink, searching fish, jet x2, turn 90 deg", replace(base, jet_boost=2.0, escape_angle=90.0)),
]

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print(f"Python model, {n_seeds} seeds, t_max = {T_MAX:.0f}")
    for label, par in SETTINGS:
        res = [hunt(par, seed, T_MAX) for seed in range(n_seeds)]
        t = np.array([r["time"] for r in res])
        unc = int(sum(1 - r["captured"] for r in res))
        print(f"{label:44s} mean {t.mean():6.1f} +- {1.96 * t.std(ddof=1) / np.sqrt(len(t)):4.1f}  uncaught {unc}/{n_seeds}")
