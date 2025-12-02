from test_scene import TestSceneCfg

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
from src.sim_state import SimProps, SimState
from src.utils import sample_from_ellipsoid, will_overlap
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from src.base_dataset_config import BaseDatasetConfig


@dataclass
class TestDatasetConfig(BaseDatasetConfig):
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(  # type: ignore
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    # took 12s for 5 envs, took 90s for 50 envs
    scene_cfg: InteractiveSceneCfg = TestSceneCfg(num_envs=10)

    # touch the first rigid object in dict
    def get_targets(self, start: SimState) -> torch.Tensor:
        """
        Calculate or retrieve targets for all envs.
        List of positions the end effector should ik to...

        Args:
            Accepts start positions of all the rigid objects and the robot (for all envs)

        Returns:
            The target(s) for the environment.
        """

        for x in start.objs_pose.values():
            return x + torch.tensor([0, 0, 0.1, 0, 0, 0, 0], device=x.device)
            # return torch.cat(
            #     (x[:, :3], torch.zeros((self.num_envs, 4), device=x.device)), dim=-1
            # )
        raise ValueError("No objects")

    # default success
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

        return np.ones(self.num_envs, dtype=bool), np.array(["success"] * self.num_envs)
