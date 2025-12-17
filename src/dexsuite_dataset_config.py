from dataclasses import dataclass, field
import torch
import numpy as np
from typing import Tuple, Optional, Dict

from src.base_dataset_config import BaseDatasetConfig
from src.dexsuite_scene import DexSuiteSceneCfg
from src.sim_state import SimState

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG
from isaaclab.assets import ArticulationCfg


@dataclass
class DexsuiteDatasetConfig(BaseDatasetConfig):
    # Use the standalone scene config
    default_task: str = "pick up the cube"
    scene_cfg: DexSuiteSceneCfg = field(init=False)
    robot_cfg: ArticulationCfg = field(
        default_factory=lambda: KUKA_ALLEGRO_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
    )

    # Robot config
    ee_body_name: str = "palm_link"
    # Kuka IIWA + Allegro Hand
    arm_joint_names: str = "iiwa7_joint_.*"
    hand_joint_names: str = "(index|middle|ring|thumb)_joint_.*"

    target: str = "object_cube"
    # State tracking for each env
    STATE_NAMES = np.array(["Approach", "Grasp", "Lift"])
    # 0: Approach, 1: Grasp, 2: Lift
    _env_states: torch.Tensor = field(init=False)
    _state_timers: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.scene_cfg = DexSuiteSceneCfg(num_envs=self.num_envs)
        super().__post_init__()
        self._env_states = torch.zeros(self.num_envs, dtype=torch.long)
        self._state_timers = torch.zeros(self.num_envs, dtype=torch.long)

    def get_targets(
        self, start: SimState
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Simple State Machine for Pick and Place
        """
        device = start.robot_pose.device
        if self._env_states.device != device:
            self._env_states = self._env_states.to(device)
            self._state_timers = self._state_timers.to(device)

        # Target Pose (Position + Orientation)
        # Initialize with current EE pose to avoid jumps if not setting
        # But we usually want to set it.
        # Let's assume we want to pick "Object_Cube"
        # We need to find the object in start.objs_pose
        # Keys in objs_pose depend on how they are spawned.
        # In DexSuiteSceneCfg, we named it "object_cube".
        # SimulationController.save_props filters by "object" in name.
        # So "Object_Cube" should be there.

        obj_pose = start.objs_pose.get(self.target)  # (num_envs, 7)
        obj_pos = obj_pose[:, :3]

        target_pos = obj_pos.clone()
        target_pos[:, :3] -= torch.tensor([0.10, 0.11, -0.02], device=target_pos.device)
        # Point down
        target_rot = torch.tensor([0, 1, 0, 0], device=device).repeat(self.num_envs, 1)

        # Gripper Commands
        # 16 joints. Open = 0, Close = 1.0
        gripper_cmds = torch.zeros((self.num_envs, 16), device=device)

        # Update States
        # Simple timer based transitions for now
        # 0 -> 1 after 150 steps
        # 1 -> 2 after 100 steps

        self._state_timers += 1

        # Phase 0: Approach
        # Hover 20cm above object
        approach_mask = self._env_states == 0
        target_pos[approach_mask, 2] += 0.2
        gripper_cmds[approach_mask] = 0.0

        # Transition 0 -> 1
        transition_0_1 = (self._env_states == 0) & (self._state_timers > 150)
        self._env_states[transition_0_1] = 1
        self._state_timers[transition_0_1] = 0

        # Phase 1: Grasp
        # Go to object height (plus small offset for palm center)
        grasp_mask = self._env_states == 1
        # target_pos[grasp_mask, 2] += 0.05  # Palm center offset
        # Close gripper after a delay
        gripper_cmds[grasp_mask & (self._state_timers > 30)] = 1.0

        # Transition 1 -> 2
        transition_1_2 = (self._env_states == 1) & (self._state_timers > 100)
        self._env_states[transition_1_2] = 2
        self._state_timers[transition_1_2] = 0

        # Phase 2: Lift
        # Lift 30cm
        lift_mask = self._env_states == 2
        target_pos[lift_mask, 2] += 0.3
        gripper_cmds[lift_mask] = 1.0

        # Combine pos and rot
        target_pose = torch.cat([target_pos, target_rot], dim=-1)

        return target_pose, gripper_cmds

    def is_success(
        self, start: SimState, end: SimState
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        # Check if object is lifted
        # Get object Z from end state
        obj_z = end.objs_pose[self.target][:, 2].cpu().numpy()

        # Threshold: 0.1m
        success = obj_z > 0.1

        # Granular logging based on state
        # Map state indices to names
        current_states = self._env_states.cpu().numpy()
        # Ensure indices are within bounds (should be, but safety first)
        current_states = np.clip(current_states, 0, 2)
        log_info = self.STATE_NAMES[current_states]

        # Override log with "Success" if successful?
        # Or just keep "Lift" and rely on the boolean success flag?
        # User asked for "current state - approach, grasp, lift, etc".
        # If success, it's technically still in Lift phase (or holding).
        # But maybe "Success" is a nice explicit state.
        log_info = np.where(success, "Success", log_info)

        return success, log_info
