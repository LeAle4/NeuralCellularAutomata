import numpy as np


class Grid:
    """2D grid where each cell stores a state vector of fixed size."""

    def __init__(self, width: int, height: int, state_size: int):
        """Create a grid with shape (height, width, state_size)."""
        self.width = width
        self.height = height
        self.state_size = state_size

        self.grid = np.zeros((height, width, state_size), dtype=np.float32)

    def randomize_grid(self, vmin: float, vmax: float, random_method: str, seed: int | None = None):
        """Fill the grid with random values in [vmin, vmax)."""
        if random_method == "random":
            rng = np.random.default_rng(seed)
            self.grid = rng.uniform(vmin, vmax, (self.height, self.width, self.state_size)).astype(np.float32)
        else:
            raise ValueError("Invalid random method. Choose 'random'.")

    def create_binary_mask(self, random_method: str, seed: int | None = None):
        """Create a random binary mask with shape (height, width)."""
        if random_method == "random":
            rng = np.random.default_rng(seed)
            mask = rng.integers(0, 2, size=(self.height, self.width), dtype=np.int8)
        else:
            raise ValueError("Invalid random method. Choose 'random'.")
        
        return mask

    def get_grid_from_mask(self, random_method: str):
        """Return a masked copy of the grid, keeping only cells selected by the mask."""
        mask = self.create_binary_mask(random_method)
        return np.where(mask[..., None] == 1, self.grid, 0.0).astype(np.float32)

    def vicinity(self, x: int, y: int, size: int):
        """Return the local neighborhood centered at (x, y) with a square window size."""
        half_size = size // 2
        x_min, x_max = np.clip([x - half_size, x + half_size + 1], 0, self.width).astype(int)
        y_min, y_max = np.clip([y - half_size, y + half_size + 1], 0, self.height).astype(int)

        return self.grid[y_min:y_max, x_min:x_max]

    def __getitem__(self, key):
        return self.grid[key]

    def __setitem__(self, key, value):
        self.grid[key] = value
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return getattr(self.grid, name)

    def __str__(self) -> str:
        return str(self.grid)

    def __repr__(self) -> str:
        return f"Grid(width={self.width}, height={self.height}, state_size={self.state_size}) data = \n{self.grid}"


if __name__ == "__main__":
    pass