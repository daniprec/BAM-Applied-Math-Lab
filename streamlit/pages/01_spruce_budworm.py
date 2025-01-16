import sys

import numpy as np
from scipy.integrate import solve_ivp

import streamlit as st

# Add the path to the sys module
# (allowing the import of the utils module)
sys.path.append(".")

from sessions.s01_odes_1d.spruce_budworm import plot_solution, spruce_budworm

st.title("Spruce Budworm Population Dynamics")

# Make two columns
col1, col2 = st.columns(2)

with col1:
    n0 = st.slider("Initial budworm population (n0)", 0, 100, 10)
    r = st.slider("Intrinsic growth rate (r)", 0.0, 1.0, 0.1)

with col2:
    k = st.slider("Carrying capacity (k)", 0.0, 200.0, 100.0)
    b = st.slider("Predation rate (b)", 0.0, 1.0, 0.1)

# Solve the ODE
# Time span
t_span = (0, 200)
t_eval = np.linspace(0, 200, 1000)

# Solve the ODE
solution = solve_ivp(spruce_budworm, t_span, [n0], args=(r, k, b), t_eval=t_eval)

# Plot the results in Streamlit
fig, ax = plot_solution(solution)
st.pyplot(fig)
