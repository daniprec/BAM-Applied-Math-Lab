import sys

import numpy as np
from scipy.integrate import solve_ivp

import streamlit as st

# Add the path to the sys module
# (allowing the import of the utils module)
sys.path.append(".")

from sessions.s01_odes_1d.sir_model import plot_sir_model, sir_model
from sessions.s01_odes_1d.spruce_budworm import plot_spruce_budworm, spruce_budworm

# Open a tab
ls_tabs = st.tabs(["SIR Model", "Spruce Budworm"])

with ls_tabs[0]:
    col1, col2 = st.columns(2)

    with col1:
        beta = st.slider("Transmission rate (beta)", 0.0, 1.0, 0.3)
        gamma = st.slider("Recovery rate (gamma)", 0.0, 1.0, 0.1)

    with col2:
        i0 = st.slider("Initial infected population (i0)", 0.0, 1.0, 0.01)
        r0 = st.slider("Initial recovered population (r0)", 0.0, 1.0, 0.0)
        s0 = max(1 - i0 - r0, 0)

    # Time span
    t_span = (0, 160)
    t_eval = np.linspace(0, 160, 1000)

    # Solve the ODE
    solution = solve_ivp(
        sir_model, t_span, [s0, i0, r0], args=(beta, gamma), t_eval=t_eval
    )

    # Plot the results in Streamlit
    fig, ax = plot_sir_model(solution)
    st.pyplot(fig)

with ls_tabs[1]:
    # Make two columns
    col1, col2 = st.columns(2)

    with col1:
        n0 = st.slider("Initial budworm population (n0)", 0, 100, 10)
        r = st.slider("Intrinsic growth rate (r)", 0.0, 1.0, 0.1)

    with col2:
        k = st.slider("Carrying capacity (k)", 0.0, 200.0, 100.0)
        b = st.slider("Predation rate (b)", 0.0, 1.0, 0.1)

    # Time span
    t_span = (0, 200)
    t_eval = np.linspace(0, 200, 1000)

    # Solve the ODE
    solution = solve_ivp(spruce_budworm, t_span, [n0], args=(r, k, b), t_eval=t_eval)

    # Plot the results in Streamlit
    fig, ax = plot_spruce_budworm(solution)
    st.pyplot(fig)
