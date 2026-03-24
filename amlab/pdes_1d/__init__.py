"""1D PDE models for the Applied Math Lab package."""

from .gierer_meinhardt_1d import (
    find_unstable_spatial_modes,
    gierer_meinhardt_pde,
    is_turing_instability,
)

__all__ = [
    "find_unstable_spatial_modes",
    "gierer_meinhardt_pde",
    "is_turing_instability",
]
