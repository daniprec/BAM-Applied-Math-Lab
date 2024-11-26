from typing import List

import numpy as np
from visualization import run_interactive_plot


def fitzhugh_nagumo(
    t: float,
    y: np.ndarray,
    i_ext: float,
    a: float = 0.7,
    b: float = 0.8,
    tau: float = 12.5,
    r: float = 0.1,
) -> List[float]:
    """
    Defines the FitzHugh-Nagumo model equations.

    Parameters
    ----------
    t : float
        Time variable.
    y : ndarray
        Array containing the variables [v, w] at time t.
    i_ext : float
        External stimulus current.
    a : float
        Recovery time constant.
    b : float
        Recovery time constant.
    tau : float
        Recovery time scale.
    r : float
        Recovery time scale.

    Returns
    -------
    dydt : list of float
        Derivatives [dv/dt, dw/dt] at time t.
    """
    v, w = y
    dvdt = v - (v**3) / 3 - w + r * i_ext
    dwdt = (v + a - b * w) / tau
    return [dvdt, dwdt]


def main(
    i_ext: float = 0.5,
    t_span: float = 100,
    t_eval: float = 1000,
):
    """
    Main function to run the interactive FitzHugh-Nagumo model simulation.
    """
    run_interactive_plot(
        fitzhugh_nagumo,
        i_ext=i_ext,
        t_span=t_span,
        t_eval=t_eval,
        v0=0,
        w0=0,
        limits=(-3, -3, 3, 3),
    )


if __name__ == "__main__":
    main()
