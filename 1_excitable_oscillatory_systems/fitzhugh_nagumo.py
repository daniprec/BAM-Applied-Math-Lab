import numpy as np
import toml
from visualization import run_interactive_plot


def fitzhugh_nagumo(
    t: float,
    y: np.ndarray,
    i_ext: float = 0.5,
    a: float = 0.7,
    b: float = 0.8,
    tau: float = 12.5,
    r: float = 0.1,
) -> np.ndarray:
    """
    Defines the FitzHugh-Nagumo (FHN) model equations.
    The FHN model describes a prototype of an excitable system (e.g., a neuron).
    It is an example of a relaxation oscillator because, if the external
    stimulus i_ext exceeds a certain threshold value, the system will exhibit a
    characteristic excursion in phase space, before the variables v and w
    relax back to their rest values.
    https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model

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
    np.ndarray
        Derivatives [dv/dt, dw/dt] at time t.
    """
    v, w = y
    dvdt = v - (v**3) / 3 - w + r * i_ext
    dwdt = (v + a - b * w) / tau
    return np.array([dvdt, dwdt])


def main(config: str = "config.toml", key: str = "fitzhugh-nagumo"):
    """
    Main function to run the interactive FitzHugh-Nagumo model simulation.
    """
    # Load config file
    dict_config: dict = toml.load(config)[key]

    # Run interactive plot
    run_interactive_plot(fitzhugh_nagumo, **dict_config)


if __name__ == "__main__":
    main()
