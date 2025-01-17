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
    # Text
    st.markdown(
        "The SIR model is a simple mathematical model to describe the spread "
        "of a disease in a population. It follows the following system of ODEs:"
    )
    st.markdown("$$\\frac{dS}{dt} = - \\beta \\cdot S \\cdot I$$")
    st.markdown("$$\\frac{dI}{dt} = \\beta \\cdot S \\cdot I - \\gamma \\cdot I$$")
    st.markdown("$$\\frac{dR}{dt} = \\gamma \\cdot I$$")
    st.markdown(
        "where $S$ is the susceptible population, $I$ is the infected population, "
        "$R$ is the recovered population, $\\beta$ is the transmission rate, and $\\gamma$ "
        "is the recovery rate. All quantities are adimensional."
    )
    st.markdown(
        "Use the sliders below to change the "
        "transmission and recovery rates, as well as the initial populations "
        "and the time interval to show the results."
    )
    st.markdown("---")

    # Create two columns for input sliders
    col1, col2 = st.columns(2, gap="medium")

    # Column 1 (left): Transmission and recovery rates
    with col1:
        beta = st.slider("Transmission rate (beta)", 0.0, 1.0, 0.3)
        gamma = st.slider("Recovery rate (gamma)", 0.0, 1.0, 0.1)

    # Column 2 (right): Initial populations and time
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
    # FIRST - STRUCTURE THE STREAMLIT PAGE
    # ANY ELEMENT BOUND TO CHANGE IS DEFAULTED AS EMPTY

    # Text
    st.markdown("The spruce budworm change rate has been modeled by the following ODE:")
    st.markdown(
        " $$\\frac{dx}{dt} = r \\cdot x \\cdot (1 - \\frac{x}{k}) - \\frac{x^2}{1 + x^2}$$"
    )
    st.markdown(
        "where $x$ is the current budworm population, $r$ is the intrinsic "
        "growth rate and $k$ is the carrying capacity of the forest. "
        "All quantities are adimensional."
    )
    st.markdown(
        "The graph below shows the rate of change of the budworm population. "
        "Red points indicate equilibrium points, while the green dashed line "
        "indicates the current population."
    )
    st.markdown("---")

    # Create two columns for input sliders
    col1, col2 = st.columns(2)

    # Column 1 (left): Intrinsic growth rate
    with col1:
        r = st.slider("Intrinsic growth rate (r)", 0.0, 1.0, 0.5)

    # Column 2 (right): Carrying capacity
    with col2:
        k = st.slider("Carrying capacity (k)", 0.1, 10.0, 10.0)

    # This placeholder will be used to display the growth rate plot
    st_gr = st.empty()

    # Text
    st.markdown("---")
    st.markdown(
        "The graph below shows the evolution of the budworm population over time. "
        "The initial population is always $k / 10$. "
        "Use the slider to set the time interval you allow the population to evolve. "
        "Hit 'Evolve' to run the simulation, or 'Reset' to start over."
    )
    st.markdown("---")

    # Create three columns for additional controls
    col3, col4, col5 = st.columns([2, 1, 1], gap="large", vertical_alignment="bottom")

    # Column 3 (left): Time evaluation
    with col3:
        t_eval = st.slider("Time", 1, 100, 10)

    # Column 4 (center): Button to evolve the population
    with col4:
        button = st.button("Evolve")

    # Column 5 (right): Button to reset history
    with col5:
        button_reset = st.button("Reset")

    # This placeholder will be used to display the population plot
    st_pop = st.empty()

    st.markdown("---")
    st.markdown("## References")
    st.markdown(
        "Chapter 3.7 from Strogatz, S. H. (2018). "
        "Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, "
        "Chemistry, and Engineering. CRC Press."
    )

    # SECOND - RUN THE CHANGES
    # BUTTON PRESSES, PLOTS, AND DATA STORAGE

    # Use Streamlit session state to store the data (is like a dictionary)
    if ("sbw_x" not in st.session_state) or (button_reset):
        st.session_state["sbw_x"] = np.array([k / 10])
        st.session_state["sbw_t"] = np.array([0])

    # Retrieve the data from the session state
    t = st.session_state["sbw_t"]
    x = st.session_state["sbw_x"]

    # If Go button is pressed, evolve the model and plot the results
    if button:
        t, x = evolve_spruce_budworm(t, x, r=r, k=k, t_eval=t_eval)
        st.session_state["sbw_t"] = t
        st.session_state["sbw_x"] = x

    # Plot the growth rate
    fig_gr, ax_gr = plot_spruce_budworm_rate(x[-1], r=r, k=k)
    st_gr.pyplot(fig_gr)

    # Plot the population
    fig_pop, ax_pop = plot_spruce_budworm(t, x)
    st_pop.pyplot(fig_pop)

# ---------------------------------#

# Michaelis-Menten
with ls_tabs[2]:
    # Text
    st.markdown(
        "The Michaelis-Menten equation describes the rate of an enzymatic reaction "
        "as a function of the substrate concentration. It follows the equation:"
    )
    st.markdown("$$v = \\frac{v_{max} \\cdot s}{K_m + s}$$")
    st.markdown(
        "where $v$ is the reaction rate, $v_{max}$ is the maximum reaction rate, "
        "$s$ is the substrate concentration, and $K_m$ is the Michaelis constant. "
        "All quantities are adimensional."
    )
    st.markdown(
        "Use the sliders below to change the initial substrate concentration, "
        "maximum reaction rate, Michaelis constant, and the time interval to show the results."
    )
    st.markdown("---")

    # Create two columns for input sliders
    col1, col2 = st.columns(2, gap="medium")

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
