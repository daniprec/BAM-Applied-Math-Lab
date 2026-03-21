from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import streamlit as st

MAX_TERMS = 4
MAX_MODE_NUMBER = 12
NUM_POINTS = 200
B_PARAMETER = 1.0
GAMMA_PARAMETER = 1.0
PERTURBATION_AMPLITUDE = 0.01
MODEL_TIME_STEP = 0.01
STORE_FINE_STEP = 0.1
STORE_COARSE_STEP = 1.0
STORE_FINE_END = 10.0
FINAL_TIME = 50.0

BOUNDARY_CONDITION_DETAILS: dict[str, dict[str, Any]] = {
    "neumann": {
        "label": "Neumann",
        "family": "cos",
        "formula": (
            r"\partial_x u(0,t)=\partial_x u(L,t)=0, "
            r"\quad \partial_x v(0,t)=\partial_x v(L,t)=0"
        ),
        "mode_text": r"\cos\left(n\pi x / L\right), \quad n=0,1,2,\ldots",
        "summary": (
            "Zero-flux boundaries mirror the field at the endpoints, so peaks can sit "
            "against the walls. The admissible spatial basis is purely cosine."
        ),
    },
    "dirichlet": {
        "label": "Dirichlet",
        "family": "sin",
        "formula": r"u(0,t)=u(L,t)=u_*, \quad v(0,t)=v(L,t)=v_*",
        "mode_text": r"\sin\left(n\pi x / L\right), \quad n=1,2,3,\ldots",
        "summary": (
            "Fixed boundaries pin both fields to the homogeneous steady state at x = 0 "
            "and x = L. The admissible spatial basis is purely sine."
        ),
    },
    "periodic": {
        "label": "Periodic",
        "family": None,
        "formula": (
            r"u(0,t)=u(L,t), \quad v(0,t)=v(L,t), "
            r"\quad \partial_x u(0,t)=\partial_x u(L,t), "
            r"\quad \partial_x v(0,t)=\partial_x v(L,t)"
        ),
        "mode_text": (
            r"\sin\left(2n\pi x / L\right), \ \cos\left(2n\pi x / L\right), "
            r"\quad n=1,2,3,\ldots"
        ),
        "summary": (
            "Periodic boundaries wrap the left and right endpoints together. Each unstable "
            "mode number can appear as either a sine or a cosine profile."
        ),
    },
}


def homogeneous_steady_state(a: float, b: float = B_PARAMETER) -> tuple[float, float]:
    """Return the homogeneous steady state of the kinetic system."""
    u_star = (a + 1.0) / b
    v_star = u_star**2
    return u_star, v_star


def kinetic_jacobian(
    a: float, b: float = B_PARAMETER
) -> tuple[float, float, float, float]:
    """Return the Jacobian of the reaction terms at the homogeneous steady state."""
    fu = 2.0 * b / (a + 1.0) - b
    fv = -((b / (a + 1.0)) ** 2)
    gu = 2.0 * (a + 1.0) / b
    gv = -1.0
    return fu, fv, gu, gv


def boundary_label(boundary_condition: str) -> str:
    """Return the user-facing label for the selected boundary condition."""
    return str(BOUNDARY_CONDITION_DETAILS[boundary_condition]["label"])


def default_term(boundary_condition: str) -> tuple[str, int]:
    """Return the default perturbation term for the selected boundary condition."""
    forced_family = BOUNDARY_CONDITION_DETAILS[boundary_condition]["family"]
    if forced_family is None:
        return "cos", 1
    return str(forced_family), 1


def sanitize_terms(
    terms: tuple[tuple[str, int], ...], boundary_condition: str
) -> tuple[tuple[str, int], ...]:
    """Coerce user-selected terms to the families allowed by the boundary condition."""
    forced_family = BOUNDARY_CONDITION_DETAILS[boundary_condition]["family"]
    sanitized_terms: list[tuple[str, int]] = []

    for family, mode in terms:
        clean_family = str(forced_family) if forced_family is not None else str(family)
        clean_mode = min(MAX_MODE_NUMBER, max(1, int(mode)))
        sanitized_terms.append((clean_family, clean_mode))

    if len(sanitized_terms) == 0:
        sanitized_terms.append(default_term(boundary_condition))

    return tuple(sanitized_terms)


def mode_multiplier(boundary_condition: str) -> int:
    """Return the spatial frequency multiplier induced by the boundary condition."""
    if boundary_condition == "periodic":
        return 2
    return 1


def evaluate_term(
    x: np.ndarray,
    length: float,
    family: str,
    mode: int,
    boundary_condition: str,
) -> np.ndarray:
    """Evaluate an admissible sine or cosine mode on the spatial grid."""
    multiplier = mode_multiplier(boundary_condition)
    argument = multiplier * mode * np.pi * x / length
    if family == "sin":
        return np.sin(argument)
    if family == "cos":
        return np.cos(argument)
    raise ValueError("family must be either 'sin' or 'cos'.")


def build_noise_profile(
    x: np.ndarray,
    length: float,
    terms: tuple[tuple[str, int], ...],
    boundary_condition: str,
) -> np.ndarray:
    """Build a mean-zero, unit-amplitude perturbation compatible with the boundary data."""
    profiles = [
        evaluate_term(x, length, family, mode, boundary_condition)
        for family, mode in sanitize_terms(terms, boundary_condition)
    ]
    raw_profile = np.sum(profiles, axis=0)
    centered_profile = raw_profile - float(np.mean(raw_profile))
    scale = float(np.max(np.abs(centered_profile)))
    if np.isclose(scale, 0.0):
        scale = float(np.max(np.abs(raw_profile)))
        if np.isclose(scale, 0.0):
            return np.zeros_like(x)
        centered_profile = raw_profile

    normalized_profile = centered_profile / scale
    if boundary_condition == "dirichlet":
        normalized_profile[0] = 0.0
        normalized_profile[-1] = 0.0
    return normalized_profile


def format_term_latex(family: str, mode: int, boundary_condition: str) -> str:
    """Return the LaTeX string for one trigonometric term."""
    multiplier = mode_multiplier(boundary_condition)
    if multiplier == 1:
        return rf"\{family}\left({mode}\pi x / L\right)"
    return rf"\{family}\left(2 \cdot {mode}\pi x / L\right)"


def format_noise_equation(
    terms: tuple[tuple[str, int], ...], boundary_condition: str
) -> str:
    """Return the displayed perturbation equation."""
    pieces = [
        format_term_latex(family, mode, boundary_condition)
        for family, mode in sanitize_terms(terms, boundary_condition)
    ]
    raw_expression = pieces[0] if len(pieces) == 1 else " + ".join(pieces)
    return rf"\tilde{{\eta}}(x) = {raw_expression}"


def format_initial_conditions_latex(
    terms: tuple[tuple[str, int], ...], boundary_condition: str
) -> str:
    """Return the perturbation and initial conditions as one LaTeX block."""
    noise_equation = format_noise_equation(terms, boundary_condition)
    return rf"""
\begin{{aligned}}
{noise_equation} \\
\eta(x) &= \frac{{\tilde{{\eta}}(x) - \langle \tilde{{\eta}} \rangle}}{{\max_x \left|\tilde{{\eta}}(x) - \langle \tilde{{\eta}} \rangle\right|}} \\
u(x, 0) &= u_* \left(1 + \varepsilon \eta(x)\right) \\
v(x, 0) &= v_* \left(1 + \varepsilon \eta(x)\right) \\
\varepsilon &= {PERTURBATION_AMPLITUDE:.2f}
\end{{aligned}}
"""


def make_spatial_grid(
    length: float, num_points: int, boundary_condition: str
) -> tuple[np.ndarray, float]:
    """Return the spatial grid and spacing for the chosen boundary condition."""
    if boundary_condition == "periodic":
        x = np.linspace(0.0, length, num_points, endpoint=False)
        dx = float(length / num_points)
        return x, dx

    x = np.linspace(0.0, length, num_points)
    dx = float(x[1] - x[0])
    return x, dx


def apply_boundary_conditions(
    uv: np.ndarray, boundary_condition: str, boundary_values: np.ndarray
) -> None:
    """Project the state onto the selected boundary condition."""
    if boundary_condition == "neumann":
        uv[:, 0] = uv[:, 1]
        uv[:, -1] = uv[:, -2]
        return

    if boundary_condition == "dirichlet":
        uv[:, 0] = boundary_values
        uv[:, -1] = boundary_values
        return

    if boundary_condition == "periodic":
        return

    raise ValueError(
        "Invalid boundary_condition value. Use 'neumann', 'dirichlet', or 'periodic'."
    )


def initialize_state(
    x: np.ndarray,
    length: float,
    a: float,
    terms: tuple[tuple[str, int], ...],
    boundary_condition: str,
    b: float = B_PARAMETER,
    amplitude: float = PERTURBATION_AMPLITUDE,
) -> np.ndarray:
    """Build the initial condition around the homogeneous steady state."""
    u_star, v_star = homogeneous_steady_state(a, b)
    noise_profile = build_noise_profile(x, length, terms, boundary_condition)
    baseline = np.array([[u_star], [v_star]], dtype=float)
    state = np.repeat(baseline, x.size, axis=1)
    state *= 1.0 + amplitude * noise_profile[np.newaxis, :]
    apply_boundary_conditions(
        state, boundary_condition, np.array([u_star, v_star], dtype=float)
    )
    return state


def stable_internal_time_step(dx: float, d: float) -> float:
    """Return a conservative explicit Euler time step for the diffusion part."""
    diffusion_scale = max(1.0, d)
    return min(0.002, 0.2 * dx**2 / diffusion_scale)


def build_storage_steps() -> np.ndarray:
    """Return the step indices saved during the simulation."""
    fine_stride = int(round(STORE_FINE_STEP / MODEL_TIME_STEP))
    coarse_stride = int(round(STORE_COARSE_STEP / MODEL_TIME_STEP))
    fine_end_step = int(round(STORE_FINE_END / MODEL_TIME_STEP))
    final_step = int(round(FINAL_TIME / MODEL_TIME_STEP))
    fine_steps = np.arange(0, fine_end_step + fine_stride, fine_stride, dtype=int)
    coarse_steps = np.arange(
        fine_end_step + coarse_stride,
        final_step + coarse_stride,
        coarse_stride,
        dtype=int,
    )
    return np.concatenate((fine_steps, coarse_steps))


def laplacian_1d(uv: np.ndarray, dx: float, boundary_condition: str) -> np.ndarray:
    """Return the discrete Laplacian with the requested boundary condition."""
    if boundary_condition == "periodic":
        laplacian = -2.0 * uv
        laplacian += np.roll(uv, shift=1, axis=1)
        laplacian += np.roll(uv, shift=-1, axis=1)
        return laplacian / dx**2

    laplacian = np.zeros_like(uv)
    laplacian[:, 1:-1] = (uv[:, :-2] - 2.0 * uv[:, 1:-1] + uv[:, 2:]) / dx**2

    if boundary_condition == "neumann":
        laplacian[:, 0] = 2.0 * (uv[:, 1] - uv[:, 0]) / dx**2
        laplacian[:, -1] = 2.0 * (uv[:, -2] - uv[:, -1]) / dx**2
        return laplacian

    if boundary_condition == "dirichlet":
        return laplacian

    raise ValueError(
        "Invalid boundary_condition value. Use 'neumann', 'dirichlet', or 'periodic'."
    )


def gierer_meinhardt_rhs(
    uv: np.ndarray,
    a: float,
    d: float,
    dx: float,
    boundary_condition: str,
) -> np.ndarray:
    """Return the PDE right-hand side for one explicit Euler update."""
    laplacian = laplacian_1d(uv, dx, boundary_condition)
    u, v = uv
    lu, lv = laplacian

    reaction_u = a - B_PARAMETER * u + u**2 / v
    reaction_v = u**2 - v
    du_dt = lu + GAMMA_PARAMETER * reaction_u
    dv_dt = d * lv + GAMMA_PARAMETER * reaction_v

    if boundary_condition == "dirichlet":
        du_dt[0] = 0.0
        du_dt[-1] = 0.0
        dv_dt[0] = 0.0
        dv_dt[-1] = 0.0

    return np.array([du_dt, dv_dt])


@st.cache_data(show_spinner=False, max_entries=16)
def compute_unstable_modes(
    length: float,
    a: float,
    d: float,
    boundary_condition: str,
    num_modes: int = MAX_MODE_NUMBER,
) -> list[int]:
    """Return unstable mode numbers predicted by linear stability analysis."""
    fu, fv, gu, gv = kinetic_jacobian(a, B_PARAMETER)
    jacobian = np.array([[fu, fv], [gu, gv]], dtype=float)
    diffusion_matrix = np.diag([1.0, d])

    if boundary_condition == "periodic":
        mode_numbers = np.arange(1, num_modes + 1)
        lambda_values = ((2.0 * np.pi * mode_numbers) / length) ** 2
    elif boundary_condition == "dirichlet":
        mode_numbers = np.arange(1, num_modes + 1)
        lambda_values = ((np.pi * mode_numbers) / length) ** 2
    elif boundary_condition == "neumann":
        mode_numbers = np.arange(0, num_modes + 1)
        lambda_values = ((np.pi * mode_numbers) / length) ** 2
    else:
        raise ValueError(
            "Invalid boundary_condition value. Use 'neumann', 'dirichlet', or 'periodic'."
        )

    unstable_pairs: list[tuple[int, float]] = []
    for mode, lambda_value in zip(mode_numbers, lambda_values):
        spectrum = np.linalg.eigvals(jacobian - lambda_value * diffusion_matrix)
        growth_rate = float(np.max(spectrum.real))
        if mode > 0 and growth_rate > 0.0:
            unstable_pairs.append((int(mode), growth_rate))

    unstable_pairs.sort(key=lambda item: item[1], reverse=True)
    return [mode for mode, _ in unstable_pairs]


@st.cache_data(show_spinner=False, max_entries=12)
def simulate_gierer_meinhardt(
    length: float,
    a: float,
    d: float,
    boundary_condition: str,
    terms: tuple[tuple[str, int], ...],
    num_points: int = NUM_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the 1D Gierer-Meinhardt PDE with a nonuniform storage schedule."""
    x, dx = make_spatial_grid(length, num_points, boundary_condition)
    save_steps = build_storage_steps()
    total_steps = int(round(FINAL_TIME / MODEL_TIME_STEP))
    times = save_steps.astype(float) * MODEL_TIME_STEP

    boundary_values = np.array(homogeneous_steady_state(a, B_PARAMETER), dtype=float)
    uv = initialize_state(x, length, a, terms, boundary_condition)
    apply_boundary_conditions(uv, boundary_condition, boundary_values)

    internal_dt = stable_internal_time_step(dx, d)
    substeps = max(1, int(np.ceil(MODEL_TIME_STEP / internal_dt)))
    dt = MODEL_TIME_STEP / substeps

    v_history = np.zeros((save_steps.size, num_points), dtype=float)
    v_history[0] = uv[1]
    save_position = 0

    for step in range(1, total_steps + 1):
        for _ in range(substeps):
            if np.any(uv[1] <= 0.0):
                raise ValueError(
                    "The inhibitor became non-positive during the simulation. "
                    "Try a larger L, a smaller d, or a simpler perturbation."
                )

            dudt = gierer_meinhardt_rhs(uv, a, d, dx, boundary_condition)
            uv = uv + dt * dudt
            apply_boundary_conditions(uv, boundary_condition, boundary_values)

            if not np.isfinite(uv).all() or np.any(uv[1] <= 0.0):
                raise ValueError(
                    "The simulation became unstable for these parameters. "
                    "Try a larger L, a smaller d, or fewer perturbation modes."
                )

        if (
            save_position + 1 < save_steps.size
            and step == save_steps[save_position + 1]
        ):
            save_position += 1
            v_history[save_position] = uv[1]

    return x, times, v_history


def simulation_ylim(v_history: np.ndarray) -> tuple[float, float]:
    """Return fixed y-limits based on the full stored simulation."""
    y_min = float(np.min(v_history))
    y_max = float(np.max(v_history))
    span = y_max - y_min
    if np.isclose(span, 0.0):
        margin = max(0.05 * max(abs(y_max), 1.0), 1e-3)
    else:
        margin = 0.05 * span
    return y_min - margin, y_max + margin


def plot_profile(
    x: np.ndarray,
    values: np.ndarray,
    time_value: float,
    y_limits: tuple[float, float],
    boundary_condition: str,
) -> plt.Figure:
    """Plot the inhibitor profile at one stored time."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, values, color="#1f77b4", linewidth=2)
    ax.set_ylim(*y_limits)
    ax.set_xlabel("x")
    ax.set_ylabel("v(x, t)")
    ax.set_title(
        f"Gierer-Meinhardt 1D ({boundary_label(boundary_condition)}), t = {time_value:.2f}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def ensure_term_state() -> None:
    """Initialize the session state used by the equation builder."""
    if "gm_boundary_condition" not in st.session_state:
        st.session_state["gm_boundary_condition"] = "neumann"
    if "gm_terms" not in st.session_state:
        st.session_state["gm_terms"] = [default_term("neumann")]
    if "gm_time_index" not in st.session_state:
        st.session_state["gm_time_index"] = 0


def sync_simulation_state(current_signature: tuple[Any, ...]) -> None:
    """Clear stale simulation results when the user changes the controls."""
    previous_signature = st.session_state.get("gm_last_controls_signature")
    if previous_signature == current_signature:
        return

    st.session_state["gm_last_controls_signature"] = current_signature
    simulation = st.session_state.get("gm_simulation")
    if simulation is not None and simulation.get("signature") != current_signature:
        st.session_state["gm_simulation"] = None
        st.session_state["gm_time_index"] = 0


def render_term_controls(boundary_condition: str) -> tuple[tuple[str, int], ...]:
    """Render the mode builder and return the selected terms."""
    terms = list(
        sanitize_terms(tuple(st.session_state["gm_terms"]), boundary_condition)
    )
    st.session_state["gm_terms"] = terms

    add_col, remove_col = st.columns(2)
    with add_col:
        if st.button("Add another mode", disabled=len(terms) >= MAX_TERMS):
            if len(terms) < MAX_TERMS:
                terms.append(default_term(boundary_condition))
                st.session_state["gm_terms"] = terms
                st.rerun()
    with remove_col:
        if st.button("Remove last mode", disabled=len(terms) <= 1):
            if len(terms) > 1:
                terms.pop()
                st.session_state["gm_terms"] = terms
                st.rerun()

    updated_terms: list[tuple[str, int]] = []
    forced_family = BOUNDARY_CONDITION_DETAILS[boundary_condition]["family"]
    for index, (family_default, mode_default) in enumerate(terms):
        family_col, mode_col = st.columns([3, 2])
        if forced_family is None:
            with family_col:
                family = st.selectbox(
                    f"Function {index + 1}",
                    options=["sin", "cos"],
                    index=0 if family_default == "sin" else 1,
                    key=f"gm_family_{index}",
                )
        else:
            family = str(forced_family)
            with family_col:
                st.markdown(
                    f"Function {index + 1}: {family} is enforced by the selected boundary condition."
                )
        with mode_col:
            mode = int(
                st.number_input(
                    f"Mode number n {index + 1}",
                    min_value=1,
                    max_value=MAX_MODE_NUMBER,
                    value=int(mode_default),
                    step=1,
                    key=f"gm_mode_{index}",
                )
            )
        updated_terms.append((family, mode))

    clean_terms = sanitize_terms(tuple(updated_terms), boundary_condition)
    st.session_state["gm_terms"] = list(clean_terms)
    return clean_terms


def render_boundary_condition_help(boundary_condition: str) -> None:
    """Display the meaning of the selected boundary condition."""
    details = BOUNDARY_CONDITION_DETAILS[boundary_condition]
    st.markdown(f"### {details['label']} boundary condition")
    st.markdown(str(details["summary"]))
    st.latex(str(details["formula"]))
    st.markdown("Expected basis functions:")
    st.latex(str(details["mode_text"]))


def render_expected_modes(
    length: float, a: float, d: float, boundary_condition: str
) -> None:
    """Display the unstable modes predicted by linear theory."""
    unstable_modes = compute_unstable_modes(length, a, d, boundary_condition)

    st.markdown("### Expected Modes")
    if len(unstable_modes) == 0:
        st.info("No unstable nonzero modes are predicted for these parameters.")
        return

    modes_text = ", ".join(str(mode) for mode in unstable_modes[:8])
    if boundary_condition == "periodic":
        family_text = (
            "Each mode number can appear as either a sine or a cosine profile."
        )
    else:
        family = str(BOUNDARY_CONDITION_DETAILS[boundary_condition]["family"])
        family_text = f"The expected shapes lie in the {family} family."
    st.success(
        f"Linear stability predicts the following dominant {boundary_label(boundary_condition)} "
        f"modes: n = {modes_text}. {family_text}"
    )


def format_time_label(time_value: float) -> str:
    """Return a compact label for the stored time slider."""
    if time_value <= STORE_FINE_END:
        return f"t = {time_value:.1f}"
    return f"t = {time_value:.0f}"


def render_page() -> None:
    """Render the Streamlit page."""
    ensure_term_state()

    st.title("Gierer-Meinhardt in 1D")
    st.markdown(
        "Choose a boundary condition first, because that determines which spatial modes are "
        "admissible. Then pick the domain length and reaction parameters, build a perturbation, "
        "and compare the observed pattern with the unstable modes predicted by linear theory."
    )
    st.markdown(
        "The simulation advances with a requested output step of dt = 0.01, stores every 0.1 "
        "from t = 0 to t = 10, then stores every 1.0 until t = 50. Extra internal substeps are "
        "used automatically when explicit Euler needs more stability."
    )

    selector_col, explanation_col = st.columns([1.2, 1.8], gap="large")
    with selector_col:
        boundary_condition = st.radio(
            "Boundary condition",
            options=["neumann", "dirichlet", "periodic"],
            format_func=boundary_label,
            key="gm_boundary_condition",
        )
    with explanation_col:
        render_boundary_condition_help(boundary_condition)

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

    st.caption(
        "The parameters b = 1 and gamma = 1 are fixed. Dirichlet boundaries pin both fields "
        "to the homogeneous steady state at the endpoints."
    )
    st.markdown("### Custom Perturbation")
    st.markdown(
        "The selected basis sum is recentered to zero mean and rescaled so its maximum absolute "
        "value is 1 before the amplitude epsilon is applied. This keeps the perturbation size "
        "consistent when you add more modes."
    )
    terms = render_term_controls(boundary_condition)
    st.latex(format_initial_conditions_latex(terms, boundary_condition))

    current_signature = (length, a, d, boundary_condition, terms, NUM_POINTS)
    sync_simulation_state(current_signature)

    if st.button("Start", type="primary"):
        with st.spinner("Running the stored evolution up to t = 50..."):
            try:
                x, times, v_history = simulate_gierer_meinhardt(
                    length, a, d, boundary_condition, terms
                )
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
                    "boundary_condition": boundary_condition,
                    "terms": terms,
                    "signature": current_signature,
                }
                st.session_state["gm_time_index"] = 0

    simulation: dict[str, Any] | None = st.session_state.get("gm_simulation")
    if simulation is not None:
        st.markdown("### Stored Evolution")
        time_index = st.select_slider(
            "Stored time",
            options=list(range(len(simulation["times"]))),
            value=int(st.session_state.get("gm_time_index", 0)),
            format_func=lambda index: format_time_label(
                float(simulation["times"][index])
            ),
            key="gm_time_index",
        )
        figure = plot_profile(
            simulation["x"],
            simulation["v_history"][time_index],
            float(simulation["times"][time_index]),
            simulation_ylim(simulation["v_history"]),
            str(simulation["boundary_condition"]),
        )
        st.pyplot(figure)
        plt.close(figure)
        render_expected_modes(
            float(simulation["length"]),
            float(simulation["a"]),
            float(simulation["d"]),
            str(simulation["boundary_condition"]),
        )
    else:
        st.info(
            "Press Start to compute the stored trajectory. If you change the boundary condition, "
            "parameters, or perturbation, the previous graph is cleared so you do not compare it "
            "with the wrong setup."
        )
        render_expected_modes(length, a, d, boundary_condition)


if __name__ == "__main__":
    render_page()
