# BAM: APPLIED MATH LAB

Welcome to the Applied Math Lab! This repository contains the materials for the Applied Math Lab, a course offered by the [BAM](https://www.ie.edu/university/studies/academic-programs/bachelor-applied-mathematics/) program at IE University. The course is designed to introduce students to computational methods for solving problems in applied mathematics, with a focus on modeling and simulation. The course covers a range of topics, including ordinary differential equations, partial differential equations, agent-based modeling, network theory, and cellular automata.

## How to Use This Repository

To get started with this repository, follow these steps:

**Clone or Download**: Clone the repository using the command below, or download it as a ZIP file and extract it.

```bash
git clone https://github.com/daniprec/BAM-Applied-Math-Lab.git
```

**Create a Conda Environment**: It is recommended to create a new conda environment with the latest version of Python (as of now, Python 3.13).

```bash
conda create --name amlab python=3.13
conda activate amlab
```

**Install Required Packages**: Install the necessary packages listed in `requirements.txt`.

```bash
conda install --yes --file requirements.txt
```

Now you are all set up and ready to use the repository!

## Running Streamlit

Some scripts in this repository use Streamlit, a library included in the `requirements.txt` file. When you run a script with Streamlit, you might see the following warning:

```bash
Warning: to view this Streamlit app on a browser, run it with the following command:
```

To run Streamlit, follow these steps:

**Open Terminal**: Navigate to the directory containing your script.

**Run Streamlit**: Use the following command, replacing `<path-to-your-script>` with the path to your Streamlit script.

```bash
streamlit run <path-to-your-script>
```

This command will automatically open a new tab in your default web browser, displaying the Streamlit app.

If the browser does not open automatically, you can manually open it by navigating to the URL provided in the terminal output.

## SESSION 1 (LIVE IN-PERSON)

- Excitable and oscillatory systems.
- Using Python for numerical integration to solve 1D systems of nonlinear ordinary differential equations (ODEs).
- Creating animations in Python to visualize system evolution.
- Building interactive Python programs with on-click events to set initial conditions.

## SESSION 2 (LIVE IN-PERSON)

- Using Python for numerical integration to solve 2D systems of nonlinear ordinary differential equations (ODEs).
- Creating animations in Python to visualize system evolution.
- Building interactive Python programs with on-click events to set initial conditions.

## SESSION 3 (LIVE IN-PERSON)

- Reaction-diffusion equations in 1D.
- Using Python for numerical integration of reaction-diffusion equations, based on a finite differences scheme.
- Creating animations to visualize pattern formation.
- Exploring parameter space to observe different patterns.

## SESSION 4 (LIVE IN-PERSON)

- Reaction-diffusion equations and pattern formation in 2D
- Using Python for numerical integration of reaction-diffusion equations, based on a finite differences scheme.
- Creating animations to visualize pattern formation.
- Exploring parameter space to observe different patterns.

## SESSION 5 (LIVE IN-PERSON)

- Exploring synchronization with multiple coupled ODEs.
- Implementing the Kuramoto model in Python.
- Visualizing synchronization and analyzing the effects of coupling strength and initial conditions. This might include visual animations.

## SESSION 6 (LIVE IN-PERSON)

- Studying flocking and collective motion.
- Implementing the Vicsek model with Python.
- Analyzing the impact of noise and agent density on collective motion.
- More user interaction in Python: using the mouse as a predator to influence flock movements.

## SESSION 7 (LIVE IN-PERSON)

- Introduction to the NetworkX library.
- Creating and analyzing different network topologies (random, small-world, scale-free).
- Visualizing network structures and properties.
- Computing network metrics: degree distributions, node distance, centrality measures, clustering coefficient.

## SESSION 8 (LIVE IN-PERSON)

- Modeling epidemic spreading on complex networks.
- Introduction to compartmental models (SIS, SIR) on networks.
- Implementing epidemic models on various network topologies in Python.
- Simulating disease spread and visualizing the results.
- Analyzing the influence of network structure on epidemic dynamics.

## SESSION 9 (LIVE IN-PERSON)

- Use Facebook data to create and explore networks of social interactions.
- Simulate the dynamics of disinformation spreading on social networks, using the Daley-Kendall model.

## SESSION 10 (LIVE IN-PERSON)

- Introduction to cellular automata and Wolfram’s classification.
- Implementing 1-D cellular automata rules in Python.
- Visualizing the evolution of cellular automata patterns.
- Exploring the effects of initial and boundary conditions.
- Modeling stochastic cellular automata applied to traffic flow models.
- Implementing the Nagel-Schreckenberg model in Python.
- Simulating traffic flow and visualizing different traffic phases.
- Analyzing factors affecting traffic congestion and flow.
