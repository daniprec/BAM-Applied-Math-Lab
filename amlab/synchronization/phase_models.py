"""Phase models on the circle (Strogatz, Nonlinear Dynamics and Chaos, Ch. 4).

The functions here support the pages ``flows-on-the-circle.qmd`` and
``phase-locking.qmd`` in ``modules/collective-behaviour``.
"""

import numpy as np


def nonuniform_oscillator_period(omega: float, a: np.ndarray) -> np.ndarray:
    """Period of the nonuniform oscillator d(theta)/dt = omega - a sin(theta).

    For |a| < |omega| the flow has no fixed points and every trajectory
    circulates with period T = 2 pi / sqrt(omega^2 - a^2) (Strogatz 4.3).
    For |a| >= |omega| there are fixed points and the period is infinite.

    Parameters
    ----------
    omega : float
        Frequency of the uniform part of the flow.
    a : np.ndarray
        Amplitude of the nonuniform term.

    Returns
    -------
    np.ndarray
        Period for each value of ``a`` (``np.inf`` where fixed points exist).
    """
    a = np.asarray(a, dtype=float)
    disc = omega**2 - a**2
    period = np.full_like(a, np.inf)
    mask = disc > 0
    period[mask] = 2 * np.pi / np.sqrt(disc[mask])
    return period


def firefly_phase_difference(
    phi0: float,
    omega: float,
    big_omega: float,
    amplitude: float,
    t_end: float = 50.0,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the firefly entrainment model for the phase difference.

    The stimulus has phase Theta with dTheta/dt = big_omega. The firefly has
    phase theta with d(theta)/dt = omega + A sin(Theta - theta). The phase
    difference phi = Theta - theta obeys

        d(phi)/dt = big_omega - omega - A sin(phi)

    (Strogatz 4.5; Ermentrout and Rinzel 1984). The equation is integrated
    with the classical fourth-order Runge-Kutta scheme.

    Parameters
    ----------
    phi0 : float
        Initial phase difference, in radians.
    omega : float
        Natural flashing frequency of the firefly (rad per unit time).
    big_omega : float
        Frequency of the periodic stimulus (rad per unit time).
    amplitude : float
        Resetting strength A > 0.
    t_end : float, optional
        Final time, by default 50.
    dt : float, optional
        Time step, by default 0.01.

    Returns
    -------
    t : np.ndarray
        Time points.
    phi : np.ndarray
        Phase difference (not wrapped), same length as ``t``.
    """

    def rhs(p):
        return big_omega - omega - amplitude * np.sin(p)

    num_steps = int(round(t_end / dt))
    t = np.linspace(0.0, num_steps * dt, num_steps + 1)
    phi = np.empty(num_steps + 1)
    phi[0] = phi0
    for n in range(num_steps):
        p = phi[n]
        k1 = rhs(p)
        k2 = rhs(p + 0.5 * dt * k1)
        k3 = rhs(p + 0.5 * dt * k2)
        k4 = rhs(p + dt * k3)
        phi[n + 1] = p + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t, phi


def firefly_drift_period(omega: float, big_omega: float, amplitude: float) -> float:
    """Time for the phase difference to drift by 2 pi outside the locking range.

    T_drift = 2 pi / sqrt((omega - big_omega)^2 - A^2) when
    |omega - big_omega| > A (Strogatz 4.5). Inside the locking range the
    phase difference settles to a constant and the drift period is infinite.

    Parameters
    ----------
    omega : float
        Natural frequency of the firefly.
    big_omega : float
        Stimulus frequency.
    amplitude : float
        Resetting strength A.

    Returns
    -------
    float
        Drift period, or ``np.inf`` inside the locking range.
    """
    detuning = omega - big_omega
    if abs(detuning) <= amplitude:
        return float(np.inf)
    return float(2 * np.pi / np.sqrt(detuning**2 - amplitude**2))
