"""Compartmental epidemic models used in Module 1 (Sessions 5 and 6).

Submodules
----------
models
    SIR and SIS right-hand sides, final size relation, peak prevalence.
data
    Loader for ``data/epidemic/cases.csv`` with a synthetic fallback.
fitting
    Least-squares fit of the SIR model to daily incidence.
"""

from amlab.epidemics.data import load_cases, synthetic_sir_cases
from amlab.epidemics.fitting import fit_sir_incidence, sir_incidence
from amlab.epidemics.models import (
    final_size,
    peak_prevalence,
    sir_rhs,
    sis_endemic_level,
    sis_rhs,
)

__all__ = [
    "sir_rhs",
    "sis_rhs",
    "final_size",
    "peak_prevalence",
    "sis_endemic_level",
    "load_cases",
    "synthetic_sir_cases",
    "sir_incidence",
    "fit_sir_incidence",
]
