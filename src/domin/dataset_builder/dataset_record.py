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
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .control_utils import sanity_check_dataset_resume
from .image_writer import safe_stop_image_writer
from .lerobot_dataset import LeRobotDataset
from .utils import build_dataset_frame, hw_to_dataset_features


@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    default_task: str
    # names of all the joints
    joint_names: list[str]
    robot_type: str
    # A name and camera resolution in (width, height) tuple
    cameras: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = False
    # Upload on private repository on the Hugging Face hub.
    private: bool = True
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable
    num_image_writer_threads_per_camera: int = 4

    resume_recording: bool = False

    num_envs: int = 1

    def __post_init__(self):
        if self.default_task is None:
            raise ValueError(
                "You need to provide a task as argument in `default_task`."
            )


class DatasetRecord:
    def __init__(self, cfg: DatasetRecordConfig):
        print("DatasetRecord init called")
        self.cfg = cfg
        self.motor_features = {motor: float for motor in cfg.joint_names}
        self.camera_features = {
            cam: (res[1], res[0], 3) for cam, res in cfg.cameras.items()
        }
        self.observation_features = {**self.motor_features, **self.camera_features}
        self.action_features = {**self.motor_features}
        self.features = {
            **hw_to_dataset_features(
                self.observation_features, "observation", cfg.video
            ),
            **hw_to_dataset_features(self.action_features, "action", cfg.video),  # type: ignore
        }

        self.current_task = None

        if cfg.resume_recording and os.path.exists(cfg.root):
            self.dataset = LeRobotDataset(cfg.repo_id, root=cfg.root)
            if cfg.cameras:
                self.dataset.start_image_writer(
                    num_processes=cfg.num_image_writer_processes,
                    num_threads=cfg.num_image_writer_threads_per_camera
                    * len(cfg.cameras),
                )
            sanity_check_dataset_resume(
                self.dataset, cfg.robot_type, cfg.fps, self.features
            )
        else:
            self.dataset = LeRobotDataset.create(
                cfg.repo_id,
                cfg.fps,
                root=cfg.root,
                robot_type=cfg.robot_type,
                features=self.features,
                use_videos=cfg.video,
                image_writer_processes=cfg.num_image_writer_processes,
                image_writer_threads=cfg.num_image_writer_threads_per_camera
                * len(cfg.cameras)
                if cfg.cameras
                else 0,
            )

        self.rerecord_count = 0
        self.active_episodes = {}  # env_idx -> episode_index
        self.pending_rerecords = {}  # env_idx -> episode_index (to be retried in next story)
        self.episode_counter = 0
        self.total_rerecords = 0
        self.recording_start_time = time.time()
        self.steps_in_story = 0

    def __enter__(self):
        print("Started Recording")
        if self.cfg.resume_recording:
            self.episode_counter = self.dataset.meta.total_episodes
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        print("Exited context manager....")
        self.finish_episodes(list(self.active_episodes.keys()))

        # Save metrics
        total_time = time.time() - self.recording_start_time
        self.save_metadata("total_time_s", total_time)
        self.save_metadata("total_rerecords", self.total_rerecords)

        if exc_type:
            print(f"Exception: {exc_type}, {exc_value}")
            print(traceback)

        return False

    def new_story(self, tasks: list[str] | None = None):
        self.current_tasks = tasks  # env_idx -> task str

        # Finish any remaining active episodes from previous story
        if self.steps_in_story and self.active_episodes:
            self.finish_episodes(list(self.active_episodes.keys()))

        print(f"Starting new story with {self.cfg.num_envs} episodes")
        for env_idx in range(self.cfg.num_envs):
            # Check if this env has a pending re-record
            if env_idx in self.pending_rerecords:
                episode_index = self.pending_rerecords.pop(env_idx)
                print(f"Env {env_idx}: Retrying episode {episode_index}")
            else:
                episode_index = self.episode_counter
                self.episode_counter += 1

            self.active_episodes[env_idx] = episode_index

    def finish_episodes(self, env_idxs: int | list[int] | torch.Tensor):
        if isinstance(env_idxs, int):
            env_idxs = [env_idxs]
        elif isinstance(env_idxs, torch.Tensor):
            env_idxs = env_idxs.tolist()

        for env_idx in env_idxs:
            if env_idx not in self.active_episodes:
                continue

            episode_index = self.active_episodes[env_idx]
            print(f"Saving episode {episode_index} (env {env_idx})")
            # TODO: DELETE IMAGES?
            self.dataset.save_episode(episode_index)
            del self.active_episodes[env_idx]

    def rerecord(self, env_idxs: int | list[int] | torch.Tensor):
        if isinstance(env_idxs, int):
            env_idxs = [env_idxs]
        elif isinstance(env_idxs, torch.Tensor):
            env_idxs = env_idxs.tolist()

        for env_idx in env_idxs:
            if env_idx not in self.active_episodes:
                continue

            episode_index = self.active_episodes[env_idx]
            if self.dataset.image_writer is not None:
                self.dataset.image_writer.wait_until_done()

            self.dataset.clear_episode_buffer(episode_index)

            # Schedule for next story
            self.pending_rerecords[env_idx] = episode_index

            # Remove from active episodes so we stop recording for this story
            del self.active_episodes[env_idx]

            self.total_rerecords += 1
            print(
                f"Rerecording env {env_idx}: scheduled retry of ep={episode_index} in next story"
            )

    def save_metadata(self, key: str, value: Any):
        self.dataset.save_metadata(key, value)
    
    @safe_stop_image_writer
    def step(
        self,
        motor_obs: torch.Tensor,
        action: torch.Tensor,
        cam_obs: dict[str, torch.Tensor] = {},
    ):
        joint_names = self.cfg.joint_names
        num_envs = self.cfg.num_envs

        # Validate shapes
        if motor_obs.shape[0] != num_envs:
            # Try to unsqueeze if num_envs is 1 and input is not batched
            if num_envs == 1 and motor_obs.ndim == 1:
                motor_obs = motor_obs.unsqueeze(0)
                action = action.unsqueeze(0)
                cam_obs = {k: v.unsqueeze(0) for k, v in cam_obs.items()}
            else:
                raise ValueError(
                    f"Expected batch size {num_envs}, got {motor_obs.shape[0]}"
                )

        for env_idx in range(num_envs):
            if env_idx not in self.active_episodes:
                continue

            episode_index = self.active_episodes[env_idx]

            # Extract single env data
            env_motor_obs = motor_obs[env_idx].cpu().numpy()
            env_action = action[env_idx].cpu().numpy()
            env_cam_obs = {k: v[env_idx].cpu().numpy() for k, v in cam_obs.items()}

            observation = {
                **{x[0]: x[1] for x in zip(joint_names, env_motor_obs)},
                **env_cam_obs,
            }
            action_dict = {x[0]: x[1] for x in zip(joint_names, env_action)}

            observation_frame = build_dataset_frame(
                self.features, observation, prefix="observation"
            )
            action_frame = build_dataset_frame(
                self.features, action_dict, prefix="action"
            )
            frame = {**observation_frame, **action_frame}

            # Determine task
            task = self.cfg.default_task
            if (
                hasattr(self, "current_tasks")
                and self.current_tasks
                and env_idx < len(self.current_tasks)
            ):
                task = self.current_tasks[env_idx]
            elif self.current_task is not None:
                task = self.current_task

            self.dataset.add_frame(frame, task=task, episode_index=episode_index)
        self.steps_in_story = 0
