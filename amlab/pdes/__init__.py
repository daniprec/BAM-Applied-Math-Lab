"""Backward-compatible re-exports from amlab.pdes_1d and amlab.pdes_2d."""

from amlab.pdes_1d.gierer_meinhardt_1d import (
    find_unstable_spatial_modes as find_unstable_spatial_modes_1d,
)
from amlab.pdes_1d.gierer_meinhardt_1d import (
    gierer_meinhardt_pde as gierer_meinhardt_pde_1d,
)
from amlab.pdes_1d.gierer_meinhardt_1d import is_turing_instability
from amlab.pdes_2d.gierer_meinhardt_2d import (
    find_unstable_spatial_modes as find_unstable_spatial_modes_2d,
)
from amlab.pdes_2d.gierer_meinhardt_2d import (
    gierer_meinhardt_pde as gierer_meinhardt_pde_2d,
)
from amlab.pdes_2d.gray_scott import gray_scott_pde

__all__ = [
    "find_unstable_spatial_modes_1d",
    "find_unstable_spatial_modes_2d",
    "gierer_meinhardt_pde_1d",
    "gierer_meinhardt_pde_2d",
    "gray_scott_pde",
    "is_turing_instability",
]
