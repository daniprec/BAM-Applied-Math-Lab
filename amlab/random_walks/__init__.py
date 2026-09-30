"""Random walks, diffusion and heavy-tailed sampling (Module 3).

Main functions
--------------
simulate_lattice_walk      1D lattice walk with steps of +dx or -dx.
simulate_gaussian_walk     Walk in d dimensions with Gaussian steps.
mean_squared_displacement  Ensemble MSD at every time step.
heat_kernel                Solution of the diffusion equation for a point source.
sample_pareto              Samples from a continuous power law.
sample_symmetric_stable    Symmetric alpha-stable samples (Chambers-Mallows-Stuck).
mle_power_law_exponent     Maximum likelihood estimate of a power-law exponent.
levy_flight_2d             Planar walk with power-law step lengths.
"""

from amlab.random_walks.walks import (
    heat_kernel,
    levy_flight_2d,
    mean_squared_displacement,
    mle_power_law_exponent,
    sample_pareto,
    sample_symmetric_stable,
    simulate_gaussian_walk,
    simulate_lattice_walk,
)

__all__ = [
    "heat_kernel",
    "levy_flight_2d",
    "mean_squared_displacement",
    "mle_power_law_exponent",
    "sample_pareto",
    "sample_symmetric_stable",
    "simulate_gaussian_walk",
    "simulate_lattice_walk",
]
