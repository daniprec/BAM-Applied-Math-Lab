"""2D PDE models and interactive demos for the Applied Math Lab package."""

from .gierer_meinhardt_2d import find_unstable_spatial_modes, gierer_meinhardt_pde
from .gray_scott import gray_scott_pde

__all__ = [
    "find_unstable_spatial_modes",
    "gierer_meinhardt_pde",
    "gray_scott_pde",
]
