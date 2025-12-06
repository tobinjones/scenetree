# CLAUDE.md

This document provides context for AI assistants working on the frameforest codebase.

## Project Purpose

frameforest is a Python library for managing geometric objects across coordinate frames, designed for metrology, measurement, fiducialisation, alignment, and adjustment tasks.

## Core Concepts

### Objects
Geometric entities from scikit-spatial (Point, Line, Sphere, etc.), each with a unique string ID like `"QP.F1"`. The same object ID can appear in multiple scenes, representing different observations or measurements of the same logical entity.

### Scenes
A mapping of `{frame_name: {object_id: object_data, ...}}`. Each frame→objects entry represents a specific "view" - typically from a particular instrument at a particular time, or from a construction step.

### Configurations
A mapping of `{configuration_name: TransformManager, ...}`. Each configuration represents a particular way of aligning frames. Multiple configurations may be useful - for example, one for actual current positions and another for ideal/goal positions.

### Transform Graph
A forest (acyclic undirected graph, possibly disconnected) of frames connected by rigid transforms, managed by `pytransform3d.transform_manager.TransformManager`.

## Design Rationale

### Why a Forest (Not a Tree)?
The forest structure allows measured or constructed objects to be imported first in a disconnected state, then aligned/registered/fitted in steps. This supports incremental workflows where measurements are taken independently and later brought into alignment.

### Why Separate Scenes from Configurations?
- **Scenes** represent raw observations - what was measured where and when
- **Configurations** represent spatial relationships - how those observation frames relate to each other
This separation allows the same measurement data to be analyzed under different alignment hypotheses.

### Why Allow Duplicate Object IDs Across Scenes?
In metrology workflows, the same physical object is often measured in different setups or with different instruments. Each measurement produces a scene with that object's coordinates in a different frame. This duplication is natural and expected.

## Current Implementation

### Workspace Class (`src/frameforest/workspace.py`)
A dataclass containing:
- `scenes: dict[str, dict[str, Any]]` - frame name → (object ID → object data)
- `configurations: dict[str, TransformManager]` - configuration name → transform graph

The Workspace is currently just a data container with no methods beyond `__init__`.

### Public API (`src/frameforest/__init__.py`)
Exports:
- `Workspace` class
- `__version__` string

## Development Patterns

### Type Hints
Use type hints whenever they are simple and helpful. The codebase uses modern Python 3.11+ style type hints (`dict` instead of `typing.Dict`).

### Immutability
Prefer immutability except for the top-level `Workspace` class, which should be mutable.

### Testing
Write unit tests with pytest. Test files should be placed in a `tests/` directory.

### Code Quality
The project uses:
- **ruff** for linting and formatting (configured in `pyproject.toml`)
- **ty** for type checking
- GitHub Actions runs checks on all PRs (see `.github/workflows/checks.yml`)

### Documentation
Keep this CLAUDE.md document updated when you learn new things about the project's design, rationale, or conventions.

## Dependencies

### Core Dependencies
- `pytransform3d>=3.14.4` - Transform graph management
- `scikit-spatial>=9.0.1` - Geometric objects (Point, Line, Sphere, etc.)

### Development Dependencies
- `ruff>=0.14.8` - Linting and formatting
- `ty>=0.0.0a6` - Type checking

## Project Structure

```
frameforest/
├── src/
│   └── frameforest/
│       ├── __init__.py       # Public API
│       ├── workspace.py      # Workspace class
│       └── py.typed          # PEP 561 marker for type hints
├── .github/
│   └── workflows/
│       └── checks.yml        # CI: ruff + ty checks
├── pyproject.toml            # Project config, dependencies, tool config
├── README.md                 # User-facing documentation
├── CLAUDE.md                 # This file
└── LICENSE
```

## Useful Commands

```bash
# Install dependencies (including dev)
uv sync --all-groups

# Run linting
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Check formatting
uv run ruff format --check .

# Format code
uv run ruff format .

# Type check
uv run ty check

# Run tests (when added)
uv run pytest
```
