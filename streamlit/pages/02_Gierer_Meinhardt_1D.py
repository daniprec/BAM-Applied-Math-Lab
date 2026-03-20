import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import streamlit as st

sys.path.append(".")

from amlab.pdes.gierer_meinhardt_1d import (
    find_unstable_spatial_modes,
    gierer_meinhardt_pde,
)

MAX_TERMS = 4
MAX_MODE_NUMBER = 12
NUM_POINTS = 200
NUM_STORED_STEPS = 1000
B_PARAMETER = 1.0
GAMMA_PARAMETER = 1.0
PERTURBATION_AMPLITUDE = 0.01
FRAME_TIME_STEP = 0.01


def homogeneous_steady_state(a: float, b: float = B_PARAMETER) -> tuple[float, float]:
    """Return the homogeneous steady state of the kinetic system."""
    u_star = (a + 1.0) / b
    v_star = u_star**2
    return u_star, v_star


def evaluate_term(x: np.ndarray, length: float, family: str, mode: int) -> np.ndarray:
    """Evaluate a sine or cosine mode on the spatial grid."""
    argument = mode * np.pi * x / length
    if family == "sin":
        return np.sin(argument)
    if family == "cos":
        return np.cos(argument)
    raise ValueError("family must be either 'sin' or 'cos'.")


def build_noise_profile(
    x: np.ndarray, length: float, terms: tuple[tuple[str, int], ...]
) -> np.ndarray:
    """Average the selected trigonometric terms into one perturbation profile."""
    profiles = [evaluate_term(x, length, family, mode) for family, mode in terms]
    return np.mean(profiles, axis=0)


def format_term_latex(family: str, mode: int) -> str:
    """Return the LaTeX string for one trigonometric term."""
    return rf"\{family}\left({mode}\pi x / L\right)"


def format_noise_equation(terms: tuple[tuple[str, int], ...]) -> str:
    """Return the displayed perturbation equation."""
    pieces = [format_term_latex(family, mode) for family, mode in terms]
    if len(pieces) == 1:
        return rf"\eta(x) = {pieces[0]}"
    joined = " + ".join(pieces)
    return rf"\eta(x) = \frac{{1}}{{{len(pieces)}}}\left({joined}\right)"


def initialize_state(
    x: np.ndarray,
    a: float,
    terms: tuple[tuple[str, int], ...],
    b: float = B_PARAMETER,
    amplitude: float = PERTURBATION_AMPLITUDE,
) -> np.ndarray:
    """Build the initial condition around the homogeneous steady state."""
    u_star, v_star = homogeneous_steady_state(a, b)
    noise_profile = build_noise_profile(x, x[-1], terms)
    baseline = np.array([[u_star], [v_star]], dtype=float)
    state = np.repeat(baseline, x.size, axis=1)
    state *= 1.0 + amplitude * noise_profile[np.newaxis, :]
    return state


def stable_internal_time_step(dx: float, d: float) -> float:
    """Return a conservative explicit Euler time step for the diffusion part."""
    diffusion_scale = max(1.0, d)
    return min(0.002, 0.2 * dx**2 / diffusion_scale)


@st.cache_data(show_spinner=False)
def simulate_gierer_meinhardt(
    length: float,
    a: float,
    d: float,
    terms: tuple[tuple[str, int], ...],
    num_points: int = NUM_POINTS,
    num_stored_steps: int = NUM_STORED_STEPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the 1D Gierer-Meinhardt PDE and store the first time slices."""
    x = np.linspace(0.0, length, num_points)
    dx = float(x[1] - x[0])
    internal_dt = stable_internal_time_step(dx, d)
    substeps = max(1, int(np.ceil(FRAME_TIME_STEP / internal_dt)))
    dt = FRAME_TIME_STEP / substeps

    uv = initialize_state(x, a, terms)
    v_history = np.zeros((num_stored_steps, num_points), dtype=float)
    times = np.arange(num_stored_steps, dtype=float) * FRAME_TIME_STEP
    v_history[0] = uv[1]

    for step in range(1, num_stored_steps):
        for _ in range(substeps):
            dudt = gierer_meinhardt_pde(
                0.0,
                uv,
                gamma=GAMMA_PARAMETER,
                a=a,
                b=B_PARAMETER,
                d=d,
                dx=dx,
            )
            uv = uv + dt * dudt
            uv[:, 0] = uv[:, 1]
            uv[:, -1] = uv[:, -2]
            if not np.isfinite(uv).all():
                raise ValueError(
                    "The simulation became unstable for these parameters. "
                    "Try a larger L or a smaller d."
                )
        v_history[step] = uv[1]

    return x, times, v_history


def plot_profile(x: np.ndarray, values: np.ndarray, time_value: float) -> plt.Figure:
    """Plot the inhibitor profile at one stored time."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, values, color="#1f77b4", linewidth=2)
    ax.set_xlabel("x")
    ax.set_ylabel("v(x, t)")
    ax.set_title(f"Gierer-Meinhardt 1D, t = {time_value:.2f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def ensure_term_state() -> None:
    """Initialize the session state used by the equation builder."""
    if "gm_terms" not in st.session_state:
        st.session_state["gm_terms"] = [("cos", 1)]
    if "gm_time_index" not in st.session_state:
        st.session_state["gm_time_index"] = 0


def render_term_controls() -> tuple[tuple[str, int], ...]:
    """Render the mode builder and return the selected terms."""
    terms = list(st.session_state["gm_terms"])

    if st.button("Add another equation", disabled=len(terms) >= MAX_TERMS):
        if len(terms) < MAX_TERMS:
            terms.append(("sin", 1))
            st.session_state["gm_terms"] = terms
            st.rerun()

    updated_terms: list[tuple[str, int]] = []
    for index, (family_default, mode_default) in enumerate(terms):
        family_col, mode_col = st.columns([3, 2])
        with family_col:
            family = st.selectbox(
                f"Function {index + 1}",
                options=["sin", "cos"],
                index=0 if family_default == "sin" else 1,
                key=f"gm_family_{index}",
            )
        with mode_col:
            mode = int(
                st.number_input(
                    f"n {index + 1}",
                    min_value=0,
                    max_value=MAX_MODE_NUMBER,
                    value=int(mode_default),
                    step=1,
                    key=f"gm_mode_{index}",
                )
            )
        updated_terms.append((family, mode))

    st.session_state["gm_terms"] = updated_terms
    return tuple(updated_terms)


def render_expected_modes(length: float, a: float, d: float) -> None:
    """Display the unstable modes predicted by linear theory."""
    unstable_modes = find_unstable_spatial_modes(
        a=a,
        b=B_PARAMETER,
        d=d,
        length=length,
        num_modes=MAX_MODE_NUMBER + 1,
        boundary_conditions="neumann",
    )
    nonzero_modes = [mode for mode in unstable_modes if mode > 0]

    st.markdown("### Expected Modes")
    if len(nonzero_modes) == 0:
        st.info("No unstable nonzero modes are predicted for these parameters.")
        return

    modes_text = ", ".join(str(mode) for mode in nonzero_modes[:8])
    st.success(
        "Linear stability predicts the following dominant Neumann modes: "
        f"m = {modes_text}."
    )


def render_page() -> None:
    """Render the Streamlit page."""
    ensure_term_state()

    st.title("Gierer-Meinhardt in 1D")
    st.markdown(
        "Set the domain length and reaction parameters, then build a custom "
        "perturbation from up to four sine and cosine terms. The PDE uses zero-flux "
        "boundary conditions and starts from the homogeneous steady state with a small "
        "averaged perturbation."
    )

    parameter_col1, parameter_col2, parameter_col3 = st.columns(3)
    with parameter_col1:
        length = st.slider(
            "Domain length L", min_value=20.0, max_value=80.0, value=40.0
        )
    with parameter_col2:
        a = st.slider("Reaction parameter a", min_value=0.0, max_value=1.0, value=0.4)
    with parameter_col3:
        d = st.slider(
            "Diffusion parameter d", min_value=1.0, max_value=40.0, value=20.0
        )

    st.caption("The parameters b = 1 and gamma = 1 are fixed.")
    st.markdown("### Custom Perturbation")
    terms = render_term_controls()
    st.latex(format_noise_equation(terms))
    st.caption(
        "The initial condition is u(x, 0) = u_* [1 + epsilon eta(x)] and "
        "v(x, 0) = v_* [1 + epsilon eta(x)] with epsilon = 0.01."
    )

    if st.button("Start", type="primary"):
        with st.spinner("Running 1000 stored time steps..."):
            try:
                x, times, v_history = simulate_gierer_meinhardt(length, a, d, terms)
            except ValueError as error:
                st.session_state["gm_simulation"] = None
                st.error(str(error))
            else:
                st.session_state["gm_simulation"] = {
                    "x": x,
                    "times": times,
                    "v_history": v_history,
                    "length": length,
                    "a": a,
                    "d": d,
                    "terms": terms,
                }
                st.session_state["gm_time_index"] = 0

    simulation: dict[str, Any] | None = st.session_state.get("gm_simulation")
    if simulation is not None:
        st.markdown("### Stored Evolution")
        time_index = st.slider(
            "Stored time step",
            min_value=0,
            max_value=NUM_STORED_STEPS - 1,
            value=int(st.session_state.get("gm_time_index", 0)),
            key="gm_time_index",
        )
        figure = plot_profile(
            simulation["x"],
            simulation["v_history"][time_index],
            float(simulation["times"][time_index]),
        )
        st.pyplot(figure)
        plt.close(figure)
        render_expected_modes(
            float(simulation["length"]),
            float(simulation["a"]),
            float(simulation["d"]),
        )
    else:
        st.info("Press Start to compute the first 1000 stored time steps.")
        render_expected_modes(length, a, d)


if __name__ == "__main__":
    render_page()
