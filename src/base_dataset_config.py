"""
Base configuration class for dataset generation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import torch

from isaaclab.assets import (
    RigidObject,
    Articulation,
    ArticulationCfg,
    RigidObjectCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from .sim_state import SimProps, SimState
from .utils import sample_from_ellipsoid, will_overlap

# TODO:
# Add versioning based on random ranges of objects
# keep history of all generated datasets (random_ranges = list[tuple[float, float]])
# add suffix to dataset path (if random_obj: dataset_path+f'r{len(random_ranges)}')
# same for other randomizations


# Encode the whole config in a config file and add it to the dataset
# create config from file should also be an option
# hurdle: how to encode a function?


@dataclass
class BaseDatasetConfig:
    """
    Base configuration for a simulation dataset.
    """

    # Scene configuration
    scene_cfg: InteractiveSceneCfg

    # Robot configuration
    robot_cfg: ArticulationCfg

    random_start_enabled: bool = False
    eval_mode: bool = False

    # General settings
    dataset_path: str = ""
    hf_repo_id: str = ""
    num_envs: int = 1

    # camera_eye: torch.tensor

    def __post_init__(self):
        """
        Post-initialization processing.
        Generates a timestamp-based dataset path if not specified.
        """
        if not self.dataset_path:
            # Generate a unique name based on current timestamp
            # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            timestamp = datetime.now().isoformat(timespec="seconds")
            self.dataset_path = f"datasets/{self.__class__.__name__}_{timestamp}"

        self.scene_cfg.robot = self.robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

    def random_start(self) -> "BaseDatasetConfig":
        """
        Enable random start configuration.

        Returns:
            self for chaining.
        """
        self.random_start_enabled = True
        return self

    def eval(self) -> "BaseDatasetConfig":
        """
        Enable evaluation mode.

        Returns:
            self for chaining.
        """
        self.eval_mode = True
        return self

    # TODO (feat): Really important (but skip for now)
    # get start positions (robot, objects)
    # robot from_file/random (only quat/whole pose) - right now let's do only quat (complexity)
    # objects from_file/random

    # for now, only implement object random API (will return fixed poses instead of random poses (TODO))
    def get_random_object_pose(self, props: SimProps) -> Dict[str, torch.Tensor]:
        # TODO (feat): parameterize random_range
        """
        Get random object positions

        Returns:

        """
        return {
            k: torch.tensor([[0.1, 0.1, 0, 0, 0, 0, 0] for _ in range(self.num_envs)])
            for k in props.objs_size
        }
        # placed_pos = [[] for _ in range(self.num_envs)]

        # def get_asset_positions() -> np.ndarray:
        #     new_pos = []
        #     for already_placed_pos_env in placed_pos:
        #         while True:
        #             env_pos = sample_from_ellipsoid((0.16, 0.16, 0), (0.14, 0.18, 1e-5))
        #             if any(
        #                 map(lambda x: will_overlap(env_pos, x, ), already_placed_pos_env)
        #             ):
        #                 continue
        #             new_pos.append(env_pos)
        #             already_placed_pos_env.append(env_pos)
        #             break

        #     return np.array(new_pos)

        return {}

    def get_targets(self, start: SimState) -> Any:
        """
        Calculate or retrieve targets for all envs.
        List of positions the end effector should ik to...

        Args:
            Accepts start positions of all the rigid objects and the robot (for all envs)

        Returns:
            The target(s) for the environment.
        """
        raise NotImplementedError("get_targets must be implemented by subclass")

    def is_success(
        self, start: SimState, end: SimState
    ) -> tuple[
        np.ndarray[tuple[int], np.dtype[np.bool_]],
        np.ndarray[tuple[int], np.dtype[np.str_]],
    ]:
        """
        For either task (generation/eval) will check whether the simulation was successful.

        Args:
            Accepts start and end positions of all the rigid objects and the robot (for all envs)
            start: SimState
            end: SimState

        Returns:
            1D bool ndarray of shape: (num_envs, ) denoting success
            1D array of str shape: (num_envs, ) as a "key" to record statistics
        """

        raise NotImplementedError("is_success must be implemented by subclass")
