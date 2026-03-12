# Applied Math Lab

Applied Math Lab is the teaching code behind the BAM Applied Math Lab course at IE University. The package contains reusable simulations and helper modules for cellular automata, collective motion, networks, ODEs, and PDEs.

## Installation

Install the package from the root of the repository:

```
pip install .
```

The packaged runtime depends on `numpy`, `matplotlib`, `scipy`, `networkx`, `pandas`, and `pillow`.

If you also want to render the Quarto lessons or run the interactive site components, install the repository requirements instead:

```
pip install -r requirements.txt
pip install -e .
```

The full pinned repository environment is currently validated on Python 3.11 through 3.13.

## Usage

Import topic modules from `amlab` in your Python code:

```python
from amlab.cellular_automata.traffic import simulate

grid = simulate(steps=100, length=100, density=0.2, p_slow=0.3, seed=1)
```

## Project Structure
- `amlab/` contains the Python simulations and utilities.
- `modules/` contains the Quarto teaching materials.
- `assets/`, `data/`, and `img/` contain supporting site and course resources.

## License
MIT
