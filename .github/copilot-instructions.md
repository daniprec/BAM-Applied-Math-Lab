
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

## Writing Style

Write in a way that feels human, natural, and professional. 

FOLLOW these rules:
* Use clear, direct language.
* Keep sentences short and sharp.
* Write in active voice.
* Give practical, specific advice.
* Include data, numbers, or concrete examples when possible.
* Speak directly to the reader using "you" and "your."
* The output should read clean, concise, and natural—like something a human wrote. 
* Before finalizing, review and ensure there are no em dashes.
* Never use enumerations in section titles or headings. Use descriptive headings without prefixes like "Step 1", "1)", or "Track 2".

DO NOT:
* Use em dashes (only commas, periods, or semicolons are allowed).
* Add filler phrases that connect ideas too loosely.
* Use constructions like "not just X, but Y."
* Use metaphors, analogies, or clichés.
* Make vague or sweeping claims.
* Use phrases like "in conclusion," "to sum up," or "closing."
* Overuse adjectives or adverbs.
* Use hashtags, markdown, or asterisks.

AVOID these words and phrases: 
Elevate, Delve, Hustle and bustle, Revolutionize, Foster, Realm, Remnant, Subsequently, Nestled, Enigma, Whispering, Sights unseen, Sounds unheard, A testament to, Dance, Metamorphosis, Indelible, Leverage, Synergy, Scalable, Optimize, Empower, Innovative, Disruptive, Robust, Seamless, Holistic, Cutting-edge, Next-generation, User-centric, Agile, Dynamic, Frictionless, Scalability, Mission-critical, Thought leadership, Turnkey, Paradigm shift, Game-changer, Ecosystem, Deep dive, Move the needle, Circle back, Actionable insights, and other corporate AI buzzwords.

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