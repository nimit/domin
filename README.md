# Sim Dataset Gen
A package for generating robotics datasets using Isaac Sim and Isaac Lab.

## Installation

You can install this package via pip. Note that you must specify the NVIDIA PyPI index for `isaacsim` and related dependencies.

```bash
pip install . --extra-index-url https://pypi.nvidia.com
```

### Development
For development, you can use the optional dev dependencies:

```bash
pip install -e .[dev] --extra-index-url https://pypi.nvidia.com
```

## Structure
- `sim_dataset_gen/`: Core package source code.
- `examples/`: Example scripts and configurations (e.g., `generate_dexterous.py`).
- `tests/`: Tests for the package.

## Usage
To run the example generation script:

```bash
# Verify installation
python examples/generate_dexterous.py --num_episodes 10 --num_envs 4
```
