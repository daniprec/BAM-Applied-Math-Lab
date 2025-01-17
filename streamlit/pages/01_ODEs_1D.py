import sys

import numpy as np

import streamlit as st

# Add the path to the sys module
# (allowing the import of the utils module)
sys.path.append(".")

from sessions.s01_odes_1d.michaelis_menten import plot_michaelis_menten
from sessions.s01_odes_1d.sir_model import plot_sir_model
from sessions.s01_odes_1d.spruce_budworm import (
    evolve_spruce_budworm,
    plot_spruce_budworm,
    plot_spruce_budworm_rate,
)

# Create tabs for each model
ls_tabs = st.tabs(["SIR Model", "Spruce Budworm", "Michaelis-Menten"])

# ---------------------------------#

# SIR Model
with ls_tabs[0]:
    # Create two columns for input sliders
    col1, col2 = st.columns(2)

    # Column 1: Transmission and recovery rates
    with col1:
        beta = st.slider("Transmission rate (beta)", 0.0, 1.0, 0.3)
        gamma = st.slider("Recovery rate (gamma)", 0.0, 1.0, 0.1)

    # Column 2: Initial populations and time
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
    # Create two columns for input sliders
    col1, col2 = st.columns(2)

    # Column 1 (left): Intrinsic growth rate
    with col1:
        r = st.slider("Intrinsic growth rate (r)", 0.0, 1.0, 0.1)

    # Column 2 (right): Carrying capacity
    with col2:
        k = st.slider("Carrying capacity (k)", 0, 200, 100)

    # Plot the growth rate
    fig, ax = plot_spruce_budworm_rate(r=r, k=k)
    st.pyplot(fig)

    # Create three columns for additional controls
    col3, col4, col5 = st.columns(3)

    # Column 3 (left): Time evaluation
    with col3:
        t_eval = st.slider("Time (years)", 1, 100, 50)

    # Column 4 (center): Button to evolve the population
    with col4:
        button = st.button("Go!")

    # Column 5 (right): Button to reset history
    with col5:
        button_reset = st.button("Reset")

    # Use Streamlit session state to store the data
    if ("sbw_x" not in st.session_state) or (button_reset):
        st.session_state["sbw_x"] = np.array([10])
        st.session_state["sbw_t"] = np.array([0])
        st.session_state["sbw_fig"] = None

    # Initialize empty Streamlit figure
    stfig = st.empty()

    # If Go button is pressed, evolve the model and plot the results
    if button:
        t = st.session_state["sbw_t"]
        x = st.session_state["sbw_x"]
        t, x = evolve_spruce_budworm(t, x, r=r, k=k, t_eval=t_eval)
        fig, ax = plot_spruce_budworm(t, x)
        st.session_state["sbw_fig"] = fig
        st.session_state["sbw_t"] = t
        st.session_state["sbw_x"] = x

    # Display the last figure, stored in the session state
    if st.session_state["sbw_fig"] is not None:
        stfig.pyplot(st.session_state["sbw_fig"])

# ---------------------------------#

# Michaelis-Menten
with ls_tabs[2]:
    # Create two columns for input sliders
    col1, col2 = st.columns(2)

    # Column 1: Initial substrate concentration and maximum reaction rate
    with col1:
        s0 = st.slider("Initial substrate concentration (s0)", 0, 100, 10)
        vmax = st.slider("Maximum reaction rate (vmax)", 0.0, 1.0, 1.0)

    # Column 2: Michaelis constant and time
    with col2:
        km = st.slider("Michaelis constant (km)", 0.0, 1.0, 0.5)
        t_show = st.slider("Time (seconds)", 1, 10, 2)

    # Plot the results in Streamlit
    fig, ax = plot_michaelis_menten(s0=s0, vmax=vmax, km=km, t_show=t_show)
    st.pyplot(fig)
