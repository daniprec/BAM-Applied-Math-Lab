import sys

import streamlit as st

# Add the path to the sys module
# (allowing the import of the utils module)
sys.path.append(".")

from sessions.s01_odes_1d.michaelis_menten import plot_michaelis_menten
from sessions.s01_odes_1d.sir_model import plot_sir_model
from sessions.s01_odes_1d.spruce_budworm import plot_spruce_budworm

# Open one tab per model
ls_tabs = st.tabs(["SIR Model", "Spruce Budworm", "Michaelis-Menten"])

# ---------------------------------#

# SIR Model

with ls_tabs[0]:
    col1, col2 = st.columns(2)

    with col1:
        beta = st.slider("Transmission rate (beta)", 0.0, 1.0, 0.3)
        gamma = st.slider("Recovery rate (gamma)", 0.0, 1.0, 0.1)

    with col2:
        i0 = st.slider("Initial infected population (i0)", 0.0, 1.0, 0.01)
        r0 = st.slider("Initial recovered population (r0)", 0.0, 1.0, 0.0)
        s0 = max(1 - i0 - r0, 0)  # Initial susceptible population
        t_show = st.slider("Time (days)", 1, 200, 160)

    # Plot the results in Streamlit
    fig, ax = plot_sir_model(beta=beta, gamma=gamma, s0=s0, i0=i0, r0=r0, t_show=t_show)
    st.pyplot(fig)

# ---------------------------------#

# Spruce Budworm

with ls_tabs[1]:
    # Make two columns
    col1, col2 = st.columns(2)

    with col1:
        n0 = st.slider("Initial budworm population (n0)", 0, 100, 10)
        r = st.slider("Intrinsic growth rate (r)", 0.0, 1.0, 0.1)

    with col2:
        k = st.slider("Carrying capacity (k)", 0.0, 200.0, 100.0)
        b = st.slider("Predation rate (b)", 0.0, 1.0, 0.1)
        t_show = st.slider("Time (years)", 1, 100, 50)

    # Plot the results in Streamlit
    fig, ax = plot_spruce_budworm(r=r, k=k, b=b, n0=n0, t_show=t_show)
    st.pyplot(fig)

# ---------------------------------#

# Michaelis-Menten

with ls_tabs[2]:
    # Make two columns
    col1, col2 = st.columns(2)

    with col1:
        s0 = st.slider("Initial substrate concentration (s0)", 0, 100, 10)
        vmax = st.slider("Maximum reaction rate (vmax)", 0.0, 1.0, 1.0)

    with col2:
        km = st.slider("Michaelis constant (km)", 0.0, 1.0, 0.5)
        t_show = st.slider("Time (seconds)", 1, 10, 2)
    # Plot the results in Streamlit
    fig, ax = plot_michaelis_menten(s0=s0, vmax=vmax, km=km, t_show=t_show)
    st.pyplot(fig)
