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
"""
Base configuration class for dataset generation.
"""

import csv
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from isaaclab.assets import (Articulation, ArticulationCfg, RigidObject,
                             RigidObjectCfg)
from isaaclab.scene import InteractiveSceneCfg

from .sim_state import SimProps, SimState
from .utils import sample_from_ellipsoid, will_overlap

# TODO:
# Add versioning based on random ranges of objects
# keep history of all generated datasets (random_ranges = list[tuple[float, float]])
# add suffix to dataset path (if random_obj: dataset_path+f'r{len(random_ranges)}')
# same for other randomizations


# TODO:
# Encode the whole config in a config file and add it to the dataset
# create config from file should also be an option
# hurdle: how to encode a function?


@dataclass(kw_only=True)
class BaseDatasetConfig(ABC):
    """
    Base configuration for a simulation dataset.
    """

    # Robot configuration
    robot_cfg: ArticulationCfg

    # Scene configuration
    scene_cfg: InteractiveSceneCfg

    eval_mode: bool = False

    # General settings
    dataset_path: str = ""

    # Path to CSV file with start poses
    start_poses_file: str = ""

    hf_repo_id: str = ""
    num_envs: int = 1
    version: int = 0

    # Robot specific configuration
    ee_body_name: str = "ee_link"
    arm_joint_names: str = ".*"
    hand_joint_names: str = ".*"

    # Randomization ranges
    # dict of object name -> (pos_range, rot_range)
    random_ranges: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )

    # Dataset recording settings
    default_task: str
    robot_type: str = "franka_panda"
    fps: int = 60
    episode_time_s: float = 60.0
    reset_time_s: float = 60.0
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = False
    tags: List[str] = field(default_factory=list)
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4

    # camera_eye: torch.tensor

    def __post_init__(self):
        """
        Post-initialization processing.
        Generates a timestamp-based dataset path if not specified.
        """
        if self.default_task is None:
            raise ValueError("default_task must be provided in config")

        if not self.dataset_path:
            # Generate a unique name based on current timestamp
            # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            timestamp = datetime.now().isoformat(timespec="seconds")
            self.dataset_path = f"datasets/{self.__class__.__name__}_{timestamp}_v{self.version}"

        self.scene_cfg.robot = self.robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore



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

    def load_start_poses(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Load start poses from CSV file.
        Format: ep_idx, robot, obj_name1, obj_name2, ...
        Each cell contains a list of floats (13 dims for root state) encoded as JSON string.
        
        Returns:
            Dict mapping episode_index -> {object_name: pose_tensor}
        """
        if not self.start_poses_file:
            return {}

        poses = {}
        with open(self.start_poses_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row: continue
                ep_idx = int(row['ep_idx'])
                poses[ep_idx] = {}
                
                for key, value in row.items():
                    if key == 'ep_idx': continue
                    if not value: continue
                    try:
                        state_list = json.loads(value)
                        poses[ep_idx][key] = torch.tensor(state_list)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode state for {key} in episode {ep_idx}")
                        
        return poses

    def save_start_poses(self, poses: Dict[int, Dict[str, torch.Tensor]], append: bool = False):
        """
        Save start poses to CSV file.
        """
        if not self.start_poses_file:
            return

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.start_poses_file), exist_ok=True)
        
        # Determine all object names from the first entry
        if not poses:
            return
            
        first_ep = next(iter(poses.values()))
        fieldnames = ['ep_idx'] + list(first_ep.keys())
        
        mode = 'a' if append and os.path.exists(self.start_poses_file) else 'w'
        write_header = mode == 'w' or os.path.getsize(self.start_poses_file) == 0

        with open(self.start_poses_file, mode) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                
            for ep_idx, obj_poses in poses.items():
                row = {'ep_idx': ep_idx}
                for obj_name, state in obj_poses.items():
                    row[obj_name] = json.dumps(state.tolist())
                writer.writerow(row)

    def get_random_object_pose(self, props: SimProps) -> Dict[str, torch.Tensor]:
        """
        Get random object positions
        """
        # Default implementation using random_ranges if available, otherwise fixed
        poses = {}
        for k in props.objs_size:
            if k in self.random_ranges:
                # TODO: Implement actual randomization logic using self.random_ranges[k]
                # For now, just return a default pose or implement simple randomization here
                # This is a placeholder for the actual randomization logic
                poses[k] = torch.tensor(
                    [[0.6, 0.05, 0.05, 1, 0, 0, 0] for _ in range(self.num_envs)]
                )
            else:
                poses[k] = torch.tensor(
                    [[0.6, 0.05, 0.05, 1, 0, 0, 0] for _ in range(self.num_envs)]
                )

        return poses
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

    @abstractmethod
    def get_targets(
        self, start: SimState
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Calculate or retrieve targets for all envs.

        Args:
            start: SimState containing start positions of all rigid objects and the robot.

        Returns:
            Tuple containing:
                - target_pose: Tensor of shape (num_envs, 7) for end-effector pose (pos + quat)
                - auxiliary_commands: Optional Tensor for other commands (e.g., gripper joint positions)
        """
        raise NotImplementedError("get_targets must be implemented by subclass")

    @abstractmethod
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
