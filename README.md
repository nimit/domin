# Sim Dataset Gen

`sim_dataset_gen` is a comprehensive package for generating large-scale robotics datasets using [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) and [Isaac Lab](https://github.com/isaac-sim/IsaacLab). It is designed to facilitate high-fidelity simulation and data collection for training Visual-Language-Action (VLA) models.

## Features

This package generates datasets in the popular **LeRobot** format, compatible with Hugging Face's [LeRobot](https://github.com/huggingface/lerobot) library. The core dataset builder is a modified and extended version of the LeRobot dataset builder, optimized for simulation environments.

Key features inherited from our extended `dataset_builder`:

-   **Simultaneous Episode Recording**: Record multiple episodes (environments) in parallel for high throughput, significantly speeding up data generation.
-   **Story Mode**: Episodes are grouped into "stories" (batches) for efficient management and synchronized resetting.
-   **Scheduled Re-recording**: Robust handling of failed episodes. If an episode fails, it is automatically cleared and scheduled for a retry in the next batch, ensuring dataset completeness without manual intervention.
-   **Metadata & Custom Metrics**: Easily save arbitrary metadata (e.g., success rates, simulation parameters) and automatically compute episode statistics in the dataset's `info.json`.
-   **LeRobot Format Compatibility**: Produces datasets in the standard LeRobot format (Parquet files with embedded or external images), ready for training.

## Installation

You can install this package via pip. Note that you must specify the NVIDIA PyPI index for `isaacsim` and related dependencies.

```bash
pip install . --extra-index-url https://pypi.nvidia.com
```

For advanced installation instructions, including setting up Isaac Sim and creating a development environment, please see [CONTRIBUTING.md](CONTRIBUTING.md).

## Usage

To generate a dataset, use the `generate_dataset.py` script with a dataset configuration file.

### Example

```bash
python -m sim_dataset_gen.generate_dataset examples/dexterous_dataset_config.py --num_envs 10 --num_episodes 100
```

### Arguments

-   `config_path`: Path to the Python file containing the dataset configuration (e.g., `examples/dexterous_dataset_config.py`).
-   `--num_envs`: (Optional) Number of parallel environments to simulate (default: 1).
-   `--num_episodes`: (Required) Total number of episodes to record.

## Acknowledgements

This project builds upon the excellent work of the **Hugging Face LeRobot** team. The `sim_dataset_gen.dataset_builder` module is a modified adaptation of their dataset building tools, tailored for the specific needs of massive parallel simulation in Isaac Lab. We gratefully acknowledge their contributions to the open-source robotics community.
