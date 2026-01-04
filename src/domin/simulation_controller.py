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
Simulation controller for managing the simulation loop.
"""

import argparse

import isaaclab.sim as sim_utils
import torch
from isaaclab.app import AppLauncher
from isaaclab.assets.articulation import Articulation
from isaaclab.assets.rigid_object import RigidObject
from isaaclab.controllers import (DifferentialIKController,
                                  DifferentialIKControllerCfg)
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import subtract_frame_transforms

from .base_dataset_config import BaseDatasetConfig
from .dataset_builder import DatasetRecord, DatasetRecordConfig
from .sim_state import SimProps, SimState

# TODO (bug): video creation and replacement on re-record
# TODO (bug): images folder not deleted after episode finishes recording
# TODO (bug): loss of IK accuracy with >1 envs
# TODO (test): dataset record w/ multiple envs, partial and complete re-records, graceful exit

# TODO (optimization): resets/re-records happen for a whole story together. There is potential for optimization here.
# The current assumption is: since the tasks and environments are going to be more or less the same, the number of steps required for completion in each environment will be within a close range of each other. But, can add an optimization that does away with the "story" concept and starts/ends episodes individually while keeping others running

# TODO (feat): Add config param to record x frames after success (in final position)
# TODO (feat): Keyboard events (with toggle)
# TODO (feat): Record number of steps taken for each key (maybe save aggregate metrics in dataset metadata)
# TODO (feat): log positions and other data
# TODO (feat): using config.is_success
# 0. check config.is_success and depending on the result,
#     - if mode==eval, just update statistics based on string key
#     - if mode==generation, retry same episode if failure, new episode if success + write statistics


# Only supports homogeneous envs (config should be same among all envs)!
class SimulationController:
    """
    Controller to manage the simulation app, environment, and recording loop.
    """

    scene: InteractiveScene

    # dict of cameras with name and its config
    cameras: dict[str, CameraCfg]

    objects: dict[str, RigidObject]

    # TODO (feat): modify to dict for multi-robot setup (skipped rn because of complexity)
    robot: Articulation

    def __init__(
        self,
        config: "BaseDatasetConfig",
        app_launcher: AppLauncher,
        args_cli: argparse.Namespace | None = None,
    ):
        """
        Initialize the simulation controller.

        Args:
            config: The dataset configuration.
            args_cli: Command line arguments for AppLauncher. If None, defaults will be used/parsed.
            app_launcher: Existing AppLauncher instance. If provided, skips internal initialization.
        """
        self.config = config
        self.mode = "generation" if not config.eval_mode else "evaluation"
        self.app_launcher = app_launcher
        self.simulation_app = self.app_launcher.app

        # Initialize Simulation Context
        sim_cfg = sim_utils.SimulationCfg(
            dt=0.01,
            device=args_cli.device
            if args_cli and hasattr(args_cli, "device")
            else "cuda:0",
        )
        self.sim = sim_utils.SimulationContext(sim_cfg)

        # TODO: get from config
        self.sim.set_camera_view((0.0, -4.0, 4.0), (0.0, 0.0, 0.0))
        self.scene = InteractiveScene(config.scene_cfg)
        self.sim.reset()
        self.save_props()

        # self.cameras = {
        #     k: v.cfg
        #     for k, v in self.scene.sensors.items()
        #     if v is isinstance(v, Camera)
        # }
        # self.objects = {
        #     k: v for k, v in self.scene.rigid_objects.items() if "Object" in k
        # }
        # self.robot = self.scene.articulations["robot"]

        self.env = None  # Placeholder if we were using ManagerBasedEnv, but we are using InteractiveScene directly

        print(f"Simulation Controller initialized in {self.mode} mode.")


    def save_props(self):
        """
        Save properties (`SimProps`) that are not changing throughout the simulation.
        """
        self.cameras = {
            k: v.cfg for k, v in self.scene.sensors.items() if isinstance(v, Camera)
        }
        self.objects = {
            k: v for k, v in self.scene.rigid_objects.items() if "object" in k
        }
        self.robot = self.scene.articulations["robot"]

        rjoint_lims = self.robot.data.joint_pos_limits

        # Attempt to get object sizes from config if possible, otherwise default
        # TODO: Implement robust AABB computation or config parsing
        objs_size = {}
        for name, obj in self.objects.items():
            objs_size[name] = torch.tensor([0.05, 0.05, 0.05])  # Default small size

        self.props = SimProps(rjoint_lims, objs_size)

    def get_state(self):
        """
        Gets the current positions of all the rigid objects and robot in world frame.
        """
        env_origins = self.scene.env_origins
        objs_pose = {
            k: torch.cat((v.data.root_pos_w - env_origins, v.data.root_quat_w), dim=-1)
            for k, v in self.objects.items()
        }
        rdata = self.robot.data
        robot_joints = rdata.joint_pos
        robot_pose = torch.cat(
            (rdata.root_pos_w - env_origins, rdata.root_quat_w), dim=-1
        )
        return SimState(robot_joints, robot_pose, objs_pose)

    def reset(self, success_mask: list[bool] | None = None):
        """
        Reset the simulation and object positions.

        Args:
            success_mask: List of booleans indicating success for each env.
                          If None, assumes all need new episode (start).
                          True: Success, load next episode pose.
                          False: Failure, reload same episode pose (re-record).
        """
        # reset should:
        # 1. sim reset?
        # 2. set object positions to next episode's position (if next else restart from same position)
        # 3. if mode is generation, self.dataset.new_episode
        # 4. let simulation update for x number of steps (to reach the new positions; x is hardcoded right now, default/parameterized in config)

        # self.sim.reset()

        # Reset Robot
        root_state = self.robot.data.default_root_state
        local_root_pos = root_state[:, :3] + self.scene.env_origins
        local_root_pose = torch.cat((local_root_pos, root_state[:, 3:7]), dim=-1)
        self.robot.write_root_pose_to_sim(local_root_pose)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:])

        joint_pos = self.robot.data.default_joint_pos
        joint_vel = self.robot.data.default_joint_vel
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # Determine which episode index to use for each env
        # We need to know the *next* episode index for successful envs
        # and *current* episode index for failed envs.
        # DatasetRecord manages episode indices.
        # But we need to know what pose to load.

        # If success_mask is None, it's the first reset.
        if success_mask is None:
            success_mask = [True] * self.config.num_envs

        if not hasattr(self, "episode_poses") or self.episode_poses is None:
            self.episode_poses = {}  # object_name -> tensor (num_envs, 7)

        # Get current active episode indices from dataset
        active_episodes = self.dataset.active_episodes  # env_idx -> episode_index

        # Prepare poses
        # 1. If loaded_poses, use them.
        # 2. If not, generate random.
        #    - If re-record, reuse previous random pose?
        #    - If next, use new random pose.

        if self.loaded_poses:
            # Apply loaded poses
            # We need to construct the full tensor for all envs
            # Better: create a tensor and fill it

            # Initialize with default state
            # We need to handle robot and objects separately

            # Robot
            robot_pose_tensor = torch.zeros(
                (self.config.num_envs, 13), device=self.scene.device
            )
            # Fill with default first?
            robot_pose_tensor[:] = self.robot.data.default_root_state[:]

            # Objects
            obj_pose_tensors = {}
            for obj_name, obj in self.objects.items():
                obj_pose_tensors[obj_name] = obj.data.default_root_state.clone()

            for env_idx in range(self.config.num_envs):
                ep_idx = active_episodes[env_idx]
                if ep_idx in self.loaded_poses:
                    ep_poses = self.loaded_poses[ep_idx]

                    # Robot
                    if "robot" in ep_poses:
                        robot_pose_tensor[env_idx] = ep_poses["robot"].to(
                            self.scene.device
                        )

                    # Objects
                    for obj_name, pose in ep_poses.items():
                        if obj_name == "robot":
                            continue
                        if obj_name in obj_pose_tensors:
                            obj_pose_tensors[obj_name][env_idx] = pose.to(
                                self.scene.device
                            )

            # Apply
            # Robot
            robot_pose_tensor[:, :3] += self.scene.env_origins
            self.robot.write_root_state_to_sim(robot_pose_tensor)

            # Objects
            for obj_name, obj in self.objects.items():
                pose = obj_pose_tensors[obj_name]
                pose[:, :3] += self.scene.env_origins
                obj.write_root_state_to_sim(pose)
        else:
            # Random generation
            if not hasattr(self, "current_episode_poses"):
                self.current_episode_poses = {}  # env_idx -> {obj_name: pose}

            # Generate new random poses for *all* (batch efficiency)
            random_poses = self.config.get_random_object_pose(self.props)
            # random_poses: obj_name -> tensor(num_envs, 7 or 13)
            # Assuming get_random_object_pose returns 7-dim poses (pos+quat) or 13-dim?
            # The current implementation returns 7-dim.
            # We should probably upgrade it to 13-dim if we want full state, but for now 7 is fine (vels=0).

            # Also need robot pose?
            # For now assuming robot starts at default.
            # If we want to randomize robot, we need `get_random_robot_pose`.

            # Prepare data to save
            poses_to_save = {}  # ep_idx -> {obj_name: pose}

            for env_idx in range(self.config.num_envs):
                ep_idx = active_episodes[env_idx]

                # Initialize episode dict
                if ep_idx not in poses_to_save:
                    poses_to_save[ep_idx] = {}

                # Robot (save default for now if not randomized)
                # Or should we save the actual state after reset?
                # Let's save the default state we are setting.
                robot_state = self.robot.data.default_root_state[env_idx].clone()
                poses_to_save[ep_idx]["robot"] = robot_state.cpu()

                if success_mask[env_idx]:
                    # New episode: use new random poses
                    if env_idx not in self.current_episode_poses:
                        self.current_episode_poses[env_idx] = {}

                    for obj_name, batch_poses in random_poses.items():
                        # batch_poses is (num_envs, 7)
                        # Expand to 13 dim
                        p_7 = batch_poses[env_idx]
                        p_13 = torch.zeros(13, device=self.scene.device)
                        p_13[:7] = p_7
                        self.current_episode_poses[env_idx][obj_name] = p_13.clone()

                # If failure (success_mask[env_idx] == False), we keep existing self.current_episode_poses[env_idx]

                # Add to poses_to_save
                for obj_name, pose in self.current_episode_poses[env_idx].items():
                    poses_to_save[ep_idx][obj_name] = pose.cpu()

            # Apply poses to sim
            for obj_name, obj in self.objects.items():
                if obj_name in random_poses:
                    # Construct batch tensor
                    pose_tensor = torch.zeros(
                        (self.config.num_envs, 13), device=self.scene.device
                    )
                    for env_idx in range(self.config.num_envs):
                        if (
                            env_idx in self.current_episode_poses
                            and obj_name in self.current_episode_poses[env_idx]
                        ):
                            pose_tensor[env_idx] = self.current_episode_poses[env_idx][
                                obj_name
                            ]
                        else:
                            # Fallback
                            pose_tensor[env_idx] = obj.data.default_root_state[env_idx]

                    pose_tensor[:, :3] += self.scene.env_origins
                    obj.write_root_state_to_sim(pose_tensor)

            # Save to CSV (append)
            # Only save if it's a new episode (success_mask is True)?
            # Or always save?
            # If we re-record, we are using the same start pose.
            # We should only save when we *generate* a new start pose.
            # But `save_start_poses` appends.
            # We should filter `poses_to_save` to only include those where `success_mask` was True?
            # Yes, otherwise we duplicate entries for failed episodes.

            poses_to_save_filtered = {
                ep_idx: poses
                for env_idx, (ep_idx, poses) in enumerate(
                    zip(active_episodes.values(), poses_to_save.values())
                )  # active_episodes is dict, values() order?
                # active_episodes is dict {env_idx: ep_idx}
                # We need to iterate env_idx
            }

            poses_to_save_final = {}
            for env_idx in range(self.config.num_envs):
                if success_mask[env_idx]:
                    ep_idx = active_episodes[env_idx]
                    if ep_idx in poses_to_save:
                        poses_to_save_final[ep_idx] = poses_to_save[ep_idx]

            self.config.save_start_poses(poses_to_save_final, append=True)

        for _ in range(10):
            self.scene.update(self.sim.get_physics_dt())

    def record_dataset(self):
        """
        Main loop to run the simulation.
        Handles both recording and inference loops based on mode.
        """

        # Load start poses if available
        if self.config.start_poses_file:
            self.loaded_poses = self.config.load_start_poses()
        else:
            self.loaded_poses = None

        assert self.simulation_app.is_running()

        rec_cfg = DatasetRecordConfig(
            repo_id=self.config.hf_repo_id,
            robot_type=self.config.robot_type,
            default_task=self.config.default_task,
            joint_names=self.robot.joint_names,
            cameras={k: (v.width, v.height) for k, v in self.cameras.items()},
            root=self.config.dataset_path,
            fps=self.config.fps,
            episode_time_s=self.config.episode_time_s,
            reset_time_s=self.config.reset_time_s,
            num_episodes=self.config.num_episodes,
            video=self.config.video,
            push_to_hub=self.config.push_to_hub,
            tags=self.config.tags,
            num_image_writer_processes=self.config.num_image_writer_processes,
            num_image_writer_threads_per_camera=self.config.num_image_writer_threads_per_camera,
            num_envs=self.config.num_envs,
        )
        self.dataset = DatasetRecord(rec_cfg)

        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device
        )

        # TODO (CRITICAL): robot entity resolution IS ONLY TEMPORARY. FIGURE OUT A BETTER SOLUTION FOR IK
        # Resolve Robot Entities
        # Arm
        arm_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=[self.config.arm_joint_names],
            body_names=[self.config.ee_body_name],
        )
        arm_entity_cfg.resolve(self.scene)
        arm_joint_ids = arm_entity_cfg.joint_ids
        ee_body_id = arm_entity_cfg.body_ids[0]

        # Hand
        hand_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=[self.config.hand_joint_names],
        )
        hand_entity_cfg.resolve(self.scene)
        hand_joint_ids = hand_entity_cfg.joint_ids

        success_mask = [True] * self.config.num_envs

        with self.dataset:
            while self.simulation_app.is_running():
                # Check if we should stop
                if self.dataset.dataset.num_episodes >= self.config.num_episodes:
                    print("Recorded enough episodes. Exiting.")
                    break

                # Start new story (batch of episodes)
                self.dataset.new_story()

                # Reset simulation for this story
                self.reset(success_mask=success_mask)

                start_state = self.get_state()

                # Reset internal state of config
                # TODO (critical): Make this cleaner (remove config state and save/derive it from controller state in config)
                if hasattr(self.config, "_env_states"):
                    self.config._env_states.fill_(0)
                    self.config._state_timers.fill_(0)

                for _ in range(50):
                    self.sim.step()
                    self.scene.update(self.sim.get_physics_dt())

                prev_state = None
                # We loop until all envs are done (success or max steps)
                for step in range(500):
                    # Get targets (pose + auxiliary/gripper)
                    current_state = self.get_state()
                    targets_res = self.config.get_targets(current_state)
                    if isinstance(targets_res, tuple):
                        target_pose, aux_commands = targets_res
                    else:
                        target_pose, aux_commands = targets_res, None

                    diff_ik_controller.set_command(target_pose)

                    # Calculate IK for Arm
                    jacobian = self.robot.root_physx_view.get_jacobians()[
                        :, ee_body_id - 1, :, arm_joint_ids
                    ]
                    root_pose_w = self.robot.data.root_pose_w
                    joint_pos_arm = self.robot.data.joint_pos[:, arm_joint_ids]
                    ee_pose_w = self.robot.data.body_state_w[:, ee_body_id, 0:7]
                    ee_pos_b, ee_quat_b = subtract_frame_transforms(
                        root_pose_w[:, 0:3],
                        root_pose_w[:, 3:7],
                        ee_pose_w[:, 0:3],
                        ee_pose_w[:, 3:7],
                    )

                    joint_pos_des_arm = diff_ik_controller.compute(
                        ee_pos_b, ee_quat_b, jacobian, joint_pos_arm
                    )

                    # Apply Arm Targets
                    self.robot.set_joint_position_target(
                        joint_pos_des_arm, joint_ids=arm_joint_ids
                    )

                    # Apply Hand Targets (if any)
                    if aux_commands is not None and len(hand_joint_ids) > 0:
                        # Assuming aux_commands matches hand_joint_ids dimension
                        # If aux_commands is scalar (e.g. 1.0 for close), might need expansion
                        # For now assume it's the correct shape or broadcastable
                        self.robot.set_joint_position_target(
                            aux_commands, joint_ids=hand_joint_ids
                        )

                    # Construct full action vector for dataset
                    action = self.robot.data.joint_pos.clone()
                    action[:, arm_joint_ids] = joint_pos_des_arm
                    if aux_commands is not None and len(hand_joint_ids) > 0:
                        action[:, hand_joint_ids] = aux_commands

                    # Get camera observations
                    cam_obs = {}
                    for cam_name in self.cameras.keys():
                        # Find sensor object
                        sensor = self.scene.sensors[cam_name]
                        # Assuming "rgb" is the data type we want
                        if "rgb" in sensor.data.output:
                            cam_obs[cam_name] = sensor.data.output["rgb"]

                    if prev_state is not None:
                        self.dataset.step(
                            motor_obs=prev_state,
                            action=current_state.robot_joints,
                            cam_obs=cam_obs,
                        )
                    prev_state = current_state.robot_joints

                    self.scene.write_data_to_sim()
                    self.sim.step()
                    self.scene.update(self.sim.get_physics_dt())

                    # Check success
                    is_success, keys = self.config.is_success(
                        start_state, self.get_state()
                    )
                    success_mask = is_success.tolist()
                    self.dataset.finish_episodes(
                        torch.arange(self.config.num_envs)[is_success]  # type: ignore
                    )

                    if not self.dataset.active_episodes:
                        break

                self.dataset.rerecord(list(self.dataset.active_episodes.keys()))

        # doesn't exit the simulation app. have to close it manually using ctrl+c
        # self._close()

    def evaluate():
        pass

    def _close(self):
        """
        Clean up resources.
        """
        if self.simulation_app and self.simulation_app.is_running():
            self.simulation_app.close()
        print("Simulation closed.")
