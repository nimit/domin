# isort: skip_file

"""
Script to run the DexSuite dataset generation.
"""

import argparse
from isaaclab.app import AppLauncher

# Argument Parser
parser = argparse.ArgumentParser(description="Generate DexSuite Dataset")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--num_episodes", type=int, required=True, help="Number of episodes to record/infer"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import after launching app
from src.dexsuite_dataset_config import DexsuiteDatasetConfig
from src.simulation_controller import SimulationController


def main():
    # Configuration
    dataset_config = DexsuiteDatasetConfig(
        num_envs=args_cli.num_envs,
        num_episodes=args_cli.num_episodes,
    )

    # Controller
    controller = SimulationController(
        config=dataset_config, app_launcher=app_launcher, args_cli=args_cli
    )

    # Run
    controller.record_dataset()


if __name__ == "__main__":
    main()
