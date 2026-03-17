# NeuralCellularAutomata

Small NumPy-based utilities for experimenting with Neural Cellular Automata grids.

## What it does

- Creates a 2D grid of cells where each cell stores a state vector.
- Supports random initialization of the full grid.
- Builds random binary masks and applies them to the grid.
- Extracts local neighborhoods around a given coordinate.

## Requirements

- Python 3.10+
- NumPy

Install dependency:

```bash
pip install numpy
```

## Quick usage

```python
from base import Grid

grid = Grid(width=10, height=10, state_size=3)
grid.randomize_grid(0.0, 1.0, "random", seed=42)

# Access and modify cells like a NumPy array
cell_state = grid[2, 4]          # shape: (3,)
grid[2, 4] = [0.1, 0.5, 0.9]

# Masked grid (same shape as original: height x width x state_size)
masked = grid.get_grid_from_mask("random")

# Local neighborhood around (x=4, y=2) with window size 3
patch = grid.vicinity(4, 2, 3)
```

## Shapes

- Grid shape: `(height, width, state_size)`
- Binary mask shape: `(height, width)`
- Neighborhood shape: depends on border position and window size

## Run example

```bash
python base.py
```
