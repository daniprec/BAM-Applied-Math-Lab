# BAM: APPLIED MATH LAB

Welcome to the Applied Math Lab! This repository contains the materials for the Applied Math Lab, a course offered by the [BAM](https://www.ie.edu/university/studies/academic-programs/bachelor-applied-mathematics/) program at IE University. The course is designed to introduce students to computational methods for solving problems in applied mathematics, with a focus on modeling and simulation. The course covers a range of topics, including ordinary differential equations, partial differential equations, agent-based modeling, network theory, and cellular automata.

You can see the [online version of the course materials here](https://daniprec.github.io/BAM-Applied-Math-Lab/).

You can explore the hosted Streamlit app here: [bam-applied-math-lab.streamlit.app](https://bam-applied-math-lab.streamlit.app/).

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
- **Hosted Streamlit app**: [bam-applied-math-lab.streamlit.app](https://bam-applied-math-lab.streamlit.app/).
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
quarto render modules/discrete-models/cellular-1d.qmd
```

## Collective Motion: Confusion Effect, Crowd Evacuation and Squid Ink

Three extensions of the Module 2 labs (the Couzin model, the social force model and the Vicsek-type school). Each one has a **model** (Python), a **local animation**, an **interactive web figure** and a **lesson page**.

|                 | Confusion effect (predator and prey)                               | Crowd evacuation (faster is slower)                                                            |
| --------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Question        | Does blending into a school protect you from a predator?           | Does pushing harder get a crowd out of a door sooner? Does a column in front of the door help? |
| Lesson page     | `modules/collective-behaviour/confusion-effect.qmd`                | `modules/collective-behaviour/crowd-exit.qmd`                                                  |
| Model           | `amlab/collective_motion/couzin_predator.py` (extends `couzin.py`) | `amlab/collective_motion/crowd_exit.py` (extends the social force model)                       |
| Local animation | `amlab/collective_motion/couzin_predator_animation.py`             | `amlab/collective_motion/crowd_exit_animation.py`                                              |
| Web figure      | `assets/viz/widgets/confusion.js` (`data-viz="confusion"`)         | `assets/viz/widgets/crowd-exit.js` (`data-viz="crowd-exit"`)                                   |

**Squid ink** (one prey against a school of fish): _can a squid hide from a school in its own ink?_

| Item            | File                                                                          |
| --------------- | ----------------------------------------------------------------------------- |
| Lesson page     | `modules/combining-ideas/squid-ink.qmd`                                  |
| Model           | `amlab/collective_motion/squid_ink.py` (ink spreading rule, vision, capture)  |
| Local animation | `amlab/collective_motion/squid_ink_animation.py`                              |
| Web figure      | `assets/viz/widgets/squid-ink.js` (`data-viz="squid-ink"`)                    |

Shared files: `amlab/collective_motion/ie_style.py` (IE palette and Matplotlib theme), `amlab/collective_motion/run_experiments.py` (parameter sweeps), `data/*.csv` and `img/*.png` (their results).

No new packages are needed: `numpy`, `scipy`, `matplotlib` and `pillow` from `requirements.txt` are enough.

### 1. Run the animations on your computer

Activate your environment (`conda activate amlab`) and run every command **from the repository root**, the folder that contains `requirements.txt`.

**Confusion effect**

```bash
python amlab/collective_motion/couzin_predator_animation.py
```

A window opens with the hunt on the left and a scoreboard on the right. The camera follows the school.

| Option                | Meaning                                                       | Default |
| --------------------- | ------------------------------------------------------------- | ------- |
| `--theme dark\|light` | IE Tech-black stage, or white background for slides and print | `dark`  |
| `--n 100`             | number of prey                                                | 60      |
| `--odd 2`             | number of "odd" prey (marked fish)                            | 0       |
| `--cohesion 4`        | width of the zone of attraction: the cohesion dial            | 10      |
| `--speed 2`           | simulation steps per frame (faster playback)                  | 1       |
| `--seed 7`            | random seed                                                   | 3       |

In the window: the sliders change the cohesion and the confusion strength _c_ live. School size and the number of odd prey restart the hunt. **New hunt** restarts with a new random school, **Pause / play** (or the space bar) pauses.

**Squid ink**

```bash
python amlab/collective_motion/squid_ink_animation.py
```

| Option                | Meaning                                              | Default |
| --------------------- | ---------------------------------------------------- | ------- |
| `--theme dark\|light` | as above                                             | `dark`  |
| `--fish 40`           | number of fish                                       | 24      |
| `--kappa 2.8`         | opacity of the ink                                   | 2.8     |
| `--thickness 5`       | peak amount of ink in a bolus                        | 3       |
| `--ink-use 0.3`       | share of the remaining ink used per release (0 = unlimited) | 0.6 |
| `--reload 10`         | minimum time between two releases                    | 5       |
| `--fixed`             | start with a fixed camera on the whole sea           | off     |
| `--no-ink`            | control: a squid without ink                         | off     |
| `--speed 1`           | simulation steps per frame (lower is lighter)        | 2       |
| `--seed 7`            | random seed                                          | 4       |

In the window: the opacity, thickness and ink-use sliders change live. The school size restarts the hunt. **Ink on/off**, **New hunt** and **Pause** (or the space bar) do what they say. **Camera** (or the C key) switches between following the squid and a fixed view of the whole sea.

**Crowd evacuation**

```bash
python amlab/collective_motion/crowd_exit_animation.py
```

| Option                | Meaning                                                                    | Default |
| --------------------- | -------------------------------------------------------------------------- | ------- |
| `--theme dark\|light` | as above                                                                   | `dark`  |
| `--v0 6`              | desired speed in m/s (how much in a hurry)                                 | 3       |
| `--door 1.5`          | door width in m                                                            | 1.0     |
| `--column 0.8`        | distance in m from the door to a column in front of it (0 means no column) | 0       |
| `--n 100`             | number of people                                                           | 150     |
| `--speed 8`           | playback speed (1 is real time)                                            | 4       |
| `--seed 2`            | random seed                                                                | 1       |

In the window: the desired speed and the playback speed change live. The door width, the column distance and the number of people restart the evacuation. People are coloured by the pressure they feel. An evacuation takes roughly half a minute to a minute of real time at the default speed.

If no window opens (for example in a remote session), your Matplotlib backend has no display. Use the `--save` option below.

### 2. Save a GIF or a video

```bash
python amlab/collective_motion/couzin_predator_animation.py --save hunt.gif --frames 300
python amlab/collective_motion/crowd_exit_animation.py --save exit.gif --frames 300 --v0 4 --column 0.8
python amlab/collective_motion/squid_ink_animation.py --save squid-ink.gif --frames 300
python amlab/collective_motion/crowd_exit_animation.py --save exit.mp4 --frames 600   # needs ffmpeg
```

A GIF is written with Pillow and needs nothing else. An `.mp4` needs [ffmpeg](https://ffmpeg.org) on your `PATH`. Rendering a few hundred frames takes about a minute.

### 3. Run the models without graphics

```bash
python amlab/collective_motion/couzin_predator.py
python amlab/collective_motion/crowd_exit.py
python amlab/collective_motion/squid_ink.py     # self-test and a short comparison with and without ink
```

The first checks that the vectorised Couzin step agrees with `couzin_equations` when the noise is off (the difference should be about `1e-13`) and prints the capture probability for five values of the cohesion. The second runs four evacuations at different desired speeds and prints the evacuation time, flow and pressure of each (about a minute).

You can use both models from your own code:

```python
from amlab.collective_motion.couzin_predator import HuntParams, ConfusionSim, sweep
from amlab.collective_motion.crowd_exit import CrowdParams, CrowdSim, run_evacuation

res = sweep("d_attract", [0, 4, 8, 12], base=HuntParams(), n_seeds=8)   # captures per attack
print(res["values"], res["capture_prob"])

r = run_evacuation(CrowdParams(v0=4.0, obstacle=True, obs_gap=0.8), seed=0)
print(r["evac_time"], r["flow"], r["door_pressure"])
```

### 4. Reproduce the figures of the lessons

```bash
python amlab/collective_motion/run_experiments.py confusion          # about 2 minutes
python amlab/collective_motion/run_experiments.py crowd              # about 20 minutes
python amlab/collective_motion/run_experiments.py squid --seeds 60   # about 15 minutes
python amlab/collective_motion/run_experiments.py all --seeds 3      # a faster draft
python amlab/collective_motion/run_experiments.py crowd --plot-only  # redraw from data/*.csv
```

The numbers are written to `data/` (CSV) and the figures to `img/`. Every point is an average over independent random runs, set with `--seeds`.

### 5. The interactive web figures

The same models run in the browser, as part of the Quarto site. To see them while you edit:

```bash
quarto preview modules/collective-behaviour/confusion-effect.qmd
quarto preview modules/collective-behaviour/crowd-exit.qmd
quarto preview modules/combining-ideas/squid-ink.qmd
```

They are drawn by `assets/viz/widgets/*.js` through the figure framework in `assets/viz/core.js`. To put one on another page, add this block to the `.qmd` file (the text inside becomes the "Try this" guide):

```markdown
::: {.viz data-viz="crowd-exit"}

1. First thing to try.
   :::
```

The physics of the crowd widget can also be run from the command line, without a browser, with Node.js:

```bash
node -e "const {createSim}=require('./assets/viz/widgets/crowd-exit.js'); const s=createSim({v0:4},1); while(!s.done()&&s.t<300)s.step(); console.log('evacuated in', s.exits[s.exits.length-1].toFixed(0), 's')"
```

### Troubleshooting

- `ModuleNotFoundError: No module named 'amlab'`: run the command from the repository root, not from inside `amlab/`.
- The window is slow: lower `--n`, or use `--speed 1` for the hunt and `--speed 2` for the crowd.
- Text looks different from the website: the IE font Montserrat is used if it is installed, and Arial otherwise.

## Running Streamlit

If you only want to explore the live app, use the hosted version: [bam-applied-math-lab.streamlit.app](https://bam-applied-math-lab.streamlit.app/).

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
