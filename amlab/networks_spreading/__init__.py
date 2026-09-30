"""Vectorized SIS/SIR simulations and immunization strategies on networks."""

from amlab.networks_spreading.epidemics import (
    acquaintance_immunization,
    random_immunization,
    run_sir_fast,
    run_sis_fast,
    targeted_immunization,
)

__all__ = [
    "acquaintance_immunization",
    "random_immunization",
    "run_sir_fast",
    "run_sis_fast",
    "targeted_immunization",
]
