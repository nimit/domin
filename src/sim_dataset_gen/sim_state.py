import torch
from dataclasses import dataclass


@dataclass
class SimState:
    robot_joints: torch.Tensor
    robot_pose: torch.Tensor
    objs_pose: dict[str, torch.Tensor]


@dataclass
class SimProps:
    robot_joint_limits: torch.Tensor
    objs_size: dict[str, torch.Tensor]
