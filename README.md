# BAM: APPLIED MATH LAB

Welcome to the Applied Math Lab! This repository contains the materials for the Applied Math Lab, a course offered by the [BAM](https://www.ie.edu/university/studies/academic-programs/bachelor-applied-mathematics/) program at IE University. The course is designed to introduce students to computational methods for solving problems in applied mathematics, with a focus on modeling and simulation. The course covers a range of topics, including ordinary differential equations, partial differential equations, agent-based modeling, network theory, and cellular automata.

You can see the [online version of the course materials here](https://daniprec.github.io/BAM-Applied-Math-Lab/).

## Repository Structure

At a high level:

```text
.
├─ _quarto.yml
├─ index.qmd
├─ syllabus.qmd
├─ requirements.txt
├─ amlab/
├─ data/
├─ img/
├─ modules/
├─ references/
├─ streamlit/
└─ tex/
```

Brief explanation:

- **\_quarto.yml**: Quarto website configuration (navbar/sidebar/theme, site structure).
- **index.qmd**: Website home page.
- **requirements.txt**: Python dependencies used across sessions and apps.
- **syllabus.qmd**: Course syllabus / program overview.

- **amlab/**: Code used during live sessions (scripts and examples by topic/session).
- **amlab/utils/**: Reusable helpers and utilities (e.g., event handlers, solvers) used by scripts.
- **data/**: Datasets used in class (e.g., network edge lists, example graphs).
- **img/**: Images used in the Quarto site and teaching materials.
- **modules/**: Guided learning modules written in Quarto (notes, explanations, exercises).
- **references/**: External reference materials (e.g., NetworkX tutorial notebooks, datasets).
- **streamlit/**: Streamlit app(s) used for interactive demonstrations and student exploration.
- **tex/**: LaTeX assets and builds (style files, bibliography, generated PDFs/aux files).

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

Python 3.13 is the safest choice for the current pinned scientific stack. Python 3.14 is not yet recommended here because some pinned wheels are not available and pip may fall back to a local source build.

**Install Required Packages**: Install the full course and site dependencies from `requirements.txt`, then install the local package in editable mode.

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Now you are all set up and ready to use the repository!

## Package Installation

If you only want the reusable Python package and not the full Quarto or Streamlit authoring environment, install the package directly:

```bash
python -m pip install .
```

The full pinned course environment is currently validated on Python 3.11 through 3.13. The package metadata itself stays unpinned above that range so newer Python versions can still work when compatible wheels are available.

## Rendering Quarto Modules

The Quarto lessons execute Python code cells during rendering. After installing `requirements.txt`, you can render the full site or a single lesson page.

```bash
quarto render
```

or

```bash
quarto render modules/cellular-automata/cellular-1d.qmd
```

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
