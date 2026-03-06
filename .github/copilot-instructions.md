
# Copilot Repository-Wide Custom Instructions

- This repository contains applied mathematics models, simulations, and educational content.
- You must be rigorous above all else. Ask for pdf or references if needed, and cite sources when possible using the tex/references.bib file.
- When generating code, prefer clear, well-documented Python and Quarto files.
- Use existing project structure and naming conventions.
- Validate changes with tests or example runs when possible.
- For new features, follow the modular organization in the amlab/ directory.
- For notebooks, use networkx, numpy, matplotlib, and other scientific libraries as needed.
- When editing Quarto files, preserve formatting and code chunk structure.
- Always check requirements.txt and setup.py for dependencies.
- Prefer reproducible, minimal examples for demos and tutorials.
- Use Markdown formatting for documentation and comments.

## Guidance for Hints and Templates

- When writing guided exercises or templates, use the following conventions:
  - Provide clear, concise hints as comments or Markdown callouts (e.g., callout-tip, callout-note).
  - For blank code, use `# TODO:` comments to indicate what the user should implement.
  - Show solved examples after the blank template, using a callout or a separate code chunk.
  - When writing functions, include docstrings and parameter explanations.
  - Validate code with simple checks (e.g., print statements or assertions).
  - Use Quarto code chunk options to control output (e.g., echo, message, warning).
  - Reference other files or sections for context and further reading.

## Example Template

```python
def initialize_particles(num_boids, box_size):
    # Random initial theta, shape (N)
    theta = # TODO

    # Random initial x, y - shape (2, N)
    xy = # TODO
    return xy, theta
```

## Example Hint

Below the fill-in-the-blank template, you can provide a hint in a callout:

::: {.callout-tip collapse="true"}
## Solved: Initialization (click to expand)

```python
def initialize_particles(num_boids, box_size):
    theta = np.random.uniform(0, 2 * np.pi, num_boids)
    xy = np.random.uniform(0, box_size, (2, num_boids))
    return xy, theta
```
:::

Hints can also be provided as questions or notes:

::: {.callout-note collapse="true"}
## How to randomly initialize?

You can use `np.random.uniform` to generate random angles for theta and random positions for xy. Make sure to set the shape correctly for the number of boids and dimensions.
:::


::: {.callout-tip collapse="true"}
## What happens if you run it for $N=1000$?

Expected error should decrease towards zero as the number of boids increases, since the average distance to the nearest neighbor should decrease.
:::


## Example Figure

Figures are generated in code chunks and can be referenced in the text:

```{python}
#| label: fig-random
#| fig-cap: 'Random walk of particles'
#| fig-width: 6
#| fig-height: 6
#| echo: false

import matplotlib.pyplot as plt
import numpy as np

xy = np.random.rand(2, 1000) * 10

plt.scatter(xy[0], xy[1], s=10)
plt.title('Random Walk of Particles')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```