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
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from domin.dataset_builder import DatasetRecord, DatasetRecordConfig


def test_simultaneous_recording():
    root_dir = Path("tmp_dataset_test")
    if root_dir.exists():
        shutil.rmtree(root_dir)

    # Config
    cfg = DatasetRecordConfig(
        repo_id="test/simultaneous",
        root=str(root_dir),
        num_envs=2,
        joint_names=["joint1", "joint2"],
        cameras={"cam1": (64, 64)},  # width, height
        default_task="test task",
        fps=10,
        num_episodes=10,  # Total episodes desired
        video=False,  # Disable video for faster test
        num_image_writer_processes=0,
        robot_type="SO100",
    )

    print("Initializing DatasetRecord...")
    with DatasetRecord(cfg) as recorder:
        # Check if new_story was called and active episodes are set
        assert len(recorder.active_episodes) == 2
        assert recorder.active_episodes[0] == 0
        assert recorder.active_episodes[1] == 1

        # Simulate steps
        print("Simulating steps...")
        for i in range(5):
            # Batched inputs
            motor_obs = torch.randn(2, 2)
            action = torch.randn(2, 2)
            cam_obs = {"cam1": torch.randint(0, 255, (2, 64, 64, 3), dtype=torch.uint8)}

            recorder.step(motor_obs, action, cam_obs)

        # Rerecord env 0
        print("Rerecording env 0...")
        recorder.rerecord(0)

        # Env 0 should be REMOVED from active episodes for this story
        assert 0 not in recorder.active_episodes
        # Env 1 should still be active
        assert recorder.active_episodes[1] == 1

        # Simulate more steps (env 0 should be ignored)
        print("Simulating more steps...")
        for i in range(5):
            motor_obs = torch.randn(2, 2)
            action = torch.randn(2, 2)
            cam_obs = {"cam1": torch.randint(0, 255, (2, 64, 64, 3), dtype=torch.uint8)}
            recorder.step(motor_obs, action, cam_obs)

        # Finish env 1
        print("Finishing env 1...")
        recorder.finish_episodes(1)
        assert 1 not in recorder.active_episodes

        # Start NEW STORY
        print("Starting new story...")
        recorder.new_story()

        # Env 0 should now be active with the OLD episode index (0)
        assert recorder.active_episodes[0] == 0
        # Env 1 should be active with NEW episode index (2)
        assert recorder.active_episodes[1] == 2

        # Simulate steps for new story
        print("Simulating steps for new story...")
        for i in range(5):
            motor_obs = torch.randn(2, 2)
            action = torch.randn(2, 2)
            cam_obs = {"cam1": torch.randint(0, 255, (2, 64, 64, 3), dtype=torch.uint8)}
            recorder.step(motor_obs, action, cam_obs)

        # Finish all
        recorder.finish_episodes([0, 1])

        # Save metadata
        print("Saving metadata...")
        recorder.save_metadata("test_metric", 123.45)

    # Verify files
    print("Verifying files...")
    # Expected episodes:
    # Episode 0: started, rerecorded (cleared), restarted in story 2, finished. Should exist.
    # Episode 1: finished in story 1. Should exist.
    # Episode 2: finished in story 2 (env 1). Should exist.

    data_dir = root_dir / "data/chunk-000"
    files = list(data_dir.glob("*.parquet"))
    filenames = [f.name for f in files]
    print(f"Found files: {filenames}")

    assert "episode_000000.parquet" in filenames
    assert "episode_000001.parquet" in filenames
    assert "episode_000002.parquet" in filenames

    # Verify metadata
    import json

    info_path = root_dir / "meta/info.json"
    with open(info_path, "r") as f:
        info = json.load(f)
        assert info["test_metric"] == 123.45
        assert info["total_rerecords"] == 1
        assert "total_time_s" in info

    print("Verification passed!")


if __name__ == "__main__":
    test_simultaneous_recording()
