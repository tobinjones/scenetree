# frameforest

A Python library for managing geometric objects across coordinate frames, designed for metrology, measurement, fiducialisation, alignment, and adjustment tasks.

## Overview

**frameforest** helps you organize and track geometric objects measured or constructed in different coordinate frames. It supports:

- **Objects**: Geometric entities (points, lines, spheres, etc.) from scikit-spatial, each with a unique ID
- **Scenes**: Collections of objects with coordinates in specific frames (e.g., from a particular instrument or construction step)
- **Configurations**: Different spatial arrangements defining how frames relate via rigid transforms
- **Transform graphs**: Acyclic undirected graphs (forests) connecting frames through transforms

## Installation

```bash
uv add frameforest
```

Or with pip:

```bash
pip install frameforest
```

## Quick Start

```python
from frameforest import Workspace

# Create a workspace
workspace = Workspace()

# scenes: dict mapping frame names to objects in that frame
# configurations: dict mapping configuration names to TransformManagers
```

## Dependencies

- `pytransform3d` - Transform graph management
- `scikit-spatial` - Geometric objects

## Development

Install development dependencies:

```bash
uv sync --all-groups
```

Run tests:

```bash
uv run pytest
```

Run linting and type checking:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

## License

See LICENSE file.
