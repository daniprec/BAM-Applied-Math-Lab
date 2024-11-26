from typing import List

import numpy as np
from visualization import run_interactive_plot


def morris_lecar(t: float, y: np.ndarray, i_ext: float) -> List[float]:
    """
    Defines the Morris-Lecar model equations.

    Parameters
    ----------
    t : float
        Time variable.
    y : ndarray
        Array containing the variables [V, w] at time t.
    i_ext : float
        External current.

    Returns
    -------
    dydt : list of float
        Derivatives [dV/dt, dw/dt] at time t.
    """
    V, w = y
    # Parameters for the Morris-Lecar model
    C = 20.0  # Membrane capacitance (μF/cm²)
    g_Ca = 4.0  # Maximal Ca²⁺ conductance (mS/cm²)
    g_K = 8.0  # Maximal K⁺ conductance (mS/cm²)
    g_L = 2.0  # Leak conductance (mS/cm²)
    V_Ca = 120.0  # Ca²⁺ reversal potential (mV)
    V_K = -84.0  # K⁺ reversal potential (mV)
    V_L = -60.0  # Leak reversal potential (mV)
    V1 = -1.2  # Parameters for steady-state functions
    V2 = 18.0
    V3 = 2.0
    V4 = 30.0
    phi = 0.04  # Temperature-like parameter

    # Steady-state functions
    m_inf = 0.5 * (1 + np.tanh((V - V1) / V2))
    w_inf = 0.5 * (1 + np.tanh((V - V3) / V4))
    tau_w = 1 / (phi * np.cosh((V - V3) / (2 * V4)))

    # Differential equations
    dVdt = (
        i_ext - g_L * (V - V_L) - g_Ca * m_inf * (V - V_Ca) - g_K * w * (V - V_K)
    ) / C
    dwdt = (w_inf - w) / tau_w

    return [dVdt, dwdt]


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
        v0=-60,
        w0=0,
        limits=(-60, 0, 80, 1),
    )


if __name__ == "__main__":
    main()
