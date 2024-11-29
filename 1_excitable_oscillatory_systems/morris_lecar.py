from typing import List

import numpy as np
import toml
from visualization import run_interactive_plot


def morris_lecar(
    t: float,
    y: np.ndarray,
    i_ext: float = 40.0,
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
    The Morris-Lecar model is a biological neuron model developed by
    Catherine Morris and Harold Lecar to reproduce the variety of oscillatory
    behavior in relation to Ca++ and K+ conductance in the muscle fiber of the
    giant barnacle. Morris-Lecar neurons exhibit both class I and class II
    neuron excitability.
    https://en.wikipedia.org/wiki/Morris%E2%80%93Lecar_model

    Parameters
    ----------
    t : float
        Time variable.
    y : ndarray
        Array containing the variables [v, n] at time t.
        v = membrane potential
        n = recovery variable: the probability that the K+ channel is conducting
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
    v, n = y

    # Steady-state functions
    m_inf = 0.5 * (1 + np.tanh((v - v1) / v2))
    n_inf = 0.5 * (1 + np.tanh((v - v3) / v4))
    tau_w = 1 / (phi * np.cosh((v - v3) / (2 * v4)))

    # Differential equations
    dvdt = (
        i_ext - g_l * (v - v_l) - g_ca * m_inf * (v - v_ca) - g_k * n * (v - v_k)
    ) / c
    dndt = (n_inf - n) / tau_w

    return [dvdt, dndt]


def main(config: str = "config.toml", key: str = "morris-lecar"):
    """
    Main function to run the interactive FitzHugh-Nagumo model simulation.
    """
    # Load config file
    dict_config: dict = toml.load(config)[key]

    # Run interactive plot
    run_interactive_plot(morris_lecar, **dict_config)


if __name__ == "__main__":
    main()
