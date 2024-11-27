from typing import List

import numpy as np
from visualization import run_interactive_plot


def morris_lecar(
    t: float,
    y: np.ndarray,
    i_ext: float,
    c: float = 20.0,
    g_ca: float = 4.0,
    g_k: float = 8.0,
    g_l: float = 2.0,
    v_ca: float = 120.0,
    v_k: float = -84.0,
    v_l: float = -60.0,
    v1: float = -1.2,
    v2: float = 18.0,
    v3: float = 2.0,
    v4: float = 30.0,
    phi: float = 0.04,
) -> List[float]:
    """
    Defines the Morris-Lecar model equations.

    Parameters
    ----------
    t : float
        Time variable.
    y : ndarray
        Array containing the variables [v, w] at time t.
    i_ext : float
        External current.
    c : float, optional
        Membrane capacitance (uF/cm^2).
    g_ca : float, optional
        Maximal Ca2+ conductance (mS/cm^2).
    g_k : float, optional
        Maximal K+ conductance (mS/cm^2).
    g_l : float, optional
        Leak conductance (mS/cm^2).
    v_ca : float, optional
        Ca2+ reversal potential (mV).
    v_k : float, optional
        K+ reversal potential (mV).
    v_l : float, optional
        Leak reversal potential (mV).
    v1 : float, optional
        Parameter for steady-state function.
    v2 : float, optional
        Parameter for steady-state function.
    v3 : float, optional
        Parameter for steady-state function.
    v4 : float, optional
        Parameter for steady-state function.
    phi : float, optional
        Temperature-like parameter.

    Returns
    -------
    dydt : list of float
        Derivatives [dv/dt, dw/dt] at time t.
    """
    v, w = y

    # Steady-state functions
    m_inf = 0.5 * (1 + np.tanh((v - v1) / v2))
    w_inf = 0.5 * (1 + np.tanh((v - v3) / v4))
    tau_w = 1 / (phi * np.cosh((v - v3) / (2 * v4)))

    # Differential equations
    dvdt = (
        i_ext - g_l * (v - v_l) - g_ca * m_inf * (v - v_ca) - g_k * w * (v - v_k)
    ) / c
    dwdt = (w_inf - w) / tau_w

    return [dvdt, dwdt]


def main(
    i_ext: float = 40.0,
    t_span: float = 200,
    t_eval: float = 2000,
):
    """
    Main function to run the interactive Morris-Lecar model simulation.
    """
    run_interactive_plot(
        morris_lecar,
        i_ext=i_ext,
        t_span=t_span,
        t_eval=t_eval,
        v0=-60.0,
        w0=0.0,
        limits=(-60.0, 0.0, 80.0, 1.0),
    )


if __name__ == "__main__":
    main()
