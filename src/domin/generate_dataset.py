# Copyright 2026 Nimit Shah. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# isort: skip_file

"""
Script to run the generalized dataset generation.
"""


def load_config_from_path(path: str):
    """
    Dynamically load a python module from a file path and return the first class
    that inherits from BaseDatasetConfig (but is not BaseDatasetConfig itself).
    """
    import importlib.util
    import inspect
    import os
    import sys

    from domin.base_dataset_config import BaseDatasetConfig

    module_name = os.path.basename(path).replace(".py", "")

    config_dir = os.path.dirname(os.path.abspath(path))
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from path: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    config_cls = None
    for name, obj in inspect.getmembers(module):
        if (
            inspect.isclass(obj)
            and issubclass(obj, BaseDatasetConfig)
            and obj is not BaseDatasetConfig
        ):
            config_cls = obj
            break

    if config_cls is None:
        raise ValueError(
            f"No subclass of BaseDatasetConfig found in {path}. "
            "Please ensure your config class inherits from BaseDatasetConfig."
        )

    return config_cls


def main():
    import argparse
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description="Generate Dataset using Domin"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the python file containing the dataset configuration.",
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments to simulate."
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        required=True,
        help="Number of episodes to record/infer",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.enable_cameras = True
    config_path = args_cli.config_path

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    from domin.simulation_controller import SimulationController

    config_cls = load_config_from_path(config_path)

    dataset_config = config_cls(
        num_envs=args_cli.num_envs,
        num_episodes=args_cli.num_episodes,
    )  # type: ignore

    controller = SimulationController(
        config=dataset_config, app_launcher=app_launcher, args_cli=args_cli
    )

    controller.record_dataset()


if __name__ == "__main__":
    main()
