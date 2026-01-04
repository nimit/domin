# Contributing to Sim Dataset Gen

Thank you for your interest in contributing to `sim_dataset_gen`! This guide provides detailed instructions on how to set up your development environment and install the necessary dependencies.

## Installation & Setup

This package relies on NVIDIA Isaac Sim and Isaac Lab.

**Prerequisites:**
- Python 3.10 or higher.
- `pip` (or `uv` for faster installation).

### Installation

You can install the package and its development dependencies in one go. You must specify the NVIDIA PyPI index.

```bash
# Install in editable mode with dev dependencies
pip install -e .[dev] --extra-index-url https://pypi.nvidia.com
```

*Note: This command will automatically install Isaac Sim, Isaac Lab, and other required dependencies.*

## Development

### Code Style

We use [`ruff`](https://github.com/astral-sh/ruff) for code formatting and linting.

```bash
# Check for linting errors
ruff check .

# Format code
ruff format .
```

### Procedural Generation Support

If you are working on procedural generation features, install the recommended extras:

```bash
uv pip install scene-synthesizer[recommend] xatlas
```
