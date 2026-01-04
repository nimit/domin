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
Utility functions for the dataset builder.
These functions handle non-simulation tasks such as file I/O, data processing, and math helpers.
"""

import math

import numpy as np
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_from_euler_xyz


def calculate_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two positions.

    Args:
        pos1: First position.
        pos2: Second position.

    Returns:
        The Euclidean distance.
    """
    return np.linalg.norm(pos1 - pos2)


def xyz_to_quat(*xyz) -> torch.Tensor:
    """
    Convert Euler angles (in degrees) to quaternion (w, x, y, z).

    Args:
        x: Rotation around x-axis in degrees.
        y: Rotation around y-axis in degrees.
        z: Rotation around z-axis in degrees.

    Returns:
        Quaternion as a torch tensor (w, x, y, z).
    """

    # Convert degrees to radians and to tensor
    if len(xyz) == 3:
        x, y, z = xyz
    elif len(xyz) == 1 and len(xyz[0]) == 3:
        x, y, z = xyz[0]
    else:
        raise ValueError(
            "Need roll, pitch and yaw all the Euler angles to convert to quaternion form."
        )

    roll = torch.tensor(math.radians(x))
    pitch = torch.tensor(math.radians(y))
    yaw = torch.tensor(math.radians(z))

    return quat_from_euler_xyz(roll, pitch, yaw)


# TODO: optimize for multienv? (maybe using tensors will help) but very low priority since this will not be a bottleneck.
def sample_from_ellipsoid(
    outer_radii: tuple[float, float, float], inner_radii: tuple[float, float, float]
):
    """
    Generate a random point uniformly distributed in the region
    between two concentric ellipsoids, restricted to y <= 0.

    Parameters
    ----------
    outer_radii : tuple of float (rx, ry, rz)
        Radii of the outer ellipsoid.
    inner_radii : tuple of float (rx, ry, rz)
        Radii of the inner ellipsoid (smaller, inside the outer one).

    Returns
    -------
    point : np.ndarray of shape (3,)
        A point (x, y, z) within the shell.
    """
    rx_out, ry_out, rz_out = outer_radii
    rx_in, ry_in, rz_in = inner_radii
    rnd = np.random.default_rng()

    while True:
        # Sample uniformly in bounding box of the outer ellipsoid

        x = rnd.uniform(-rx_out, rx_out)
        # y values such that it only chooses the negative half (in front of the robot)
        y = rnd.uniform(-ry_out, -0.05)
        # z = rnd.uniform(-rz_out, rz_out)

        # Test "only picking up from the front of the robot"
        # if abs(x) > 0.05 and abs(y) > 0.05:
        #     continue

        # More strict (two quadrants only)
        if abs(y) < 0.05 or abs(x) < 0.05:
            continue

        # More permissive (allow more space in front of the robot)
        # if abs(x) + abs(y) < 0.175:
        #     continue

        # Check if point is inside outer ellipsoid
        # outer_check = (x / rx_out) ** 2 + (y / ry_out) ** 2 + (
        #     z / rz_out
        # ) ** 2 <= 1.0
        outer_check = (x / rx_out) ** 2 + (y / ry_out) ** 2 <= 1.0

        # Check if point is outside inner ellipsoid
        # inner_check = (x / rx_in) ** 2 + (y / ry_in) ** 2 + (z / rz_in) ** 2 >= 1.0
        inner_check = (x / rx_in) ** 2 + (y / ry_in) ** 2 >= 1.0

        if outer_check and inner_check:
            return np.array([x, y, rz_out])


# TODO: Improve. very crude function
def will_overlap(pos1: np.ndarray, pos2: np.ndarray, size) -> bool:
    """
    pos1, pos2: np.ndarray shape (3, )
    [x1, y1, z1]
    """
    raise NotImplementedError()

    # distance = ((pos2 - pos1) ** 2).sum() ** 0.5
    # return distance <= size.max()  # type: ignore


# TODO?: COMPARE?
# def check_overlap(pos1: torch.Tensor, size1: float, pos2: torch.Tensor, size2: float, buffer: float = 0.01) -> bool:
#     """
#     Check if two objects overlap based on their positions and sizes (approximated as spheres/cubes).

#     Args:
#         pos1: Position of the first object (x, y, z).
#         size1: Size (radius or half-extent) of the first object.
#         pos2: Position of the second object (x, y, z).
#         size2: Size (radius or half-extent) of the second object.
#         buffer: Minimum distance buffer between objects.

#     Returns:
#         True if they overlap, False otherwise.
#     """
#     distance = torch.norm(pos1 - pos2)
#     min_dist = size1 + size2 + buffer
#     return distance < min_dist


def randomize_object_positions(
    scene: "InteractiveScene",
    object_names: list[str],
    workspace_bounds: tuple[
        tuple[float, float], tuple[float, float]
    ],  # ((x_min, x_max), (y_min, y_max))
    z_height: float,
    object_sizes: dict[str, float],
    min_distance: float = 0.1,
) -> torch.Tensor:
    """
    Randomize positions of specified objects within workspace bounds, ensuring no overlap.

    Args:
        scene: The interactive scene containing the objects.
        object_names: List of names of objects to randomize.
        workspace_bounds: Tuple of ((x_min, x_max), (y_min, y_max)).
        z_height: The z-coordinate to place objects at.
        object_sizes: Dictionary mapping object names to their sizes (radius/extent).
        min_distance: Minimum distance between objects.

    Returns:
        Tensor of new positions (N, 3).
    """
    device = scene.device
    num_objects = len(object_names)
    positions = torch.zeros((num_objects, 3), device=device)

    x_bounds, y_bounds = workspace_bounds

    for i, name in enumerate(object_names):
        obj_size = object_sizes.get(name, 0.05)

        for attempt in range(100):
            # Sample random position
            x = torch.rand(1, device=device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
            y = torch.rand(1, device=device) * (y_bounds[1] - y_bounds[0]) + y_bounds[0]

            candidate_pos = torch.tensor([x, y, z_height], device=device)

            overlap = False
            # Check against already placed objects
            for j in range(i):
                prev_name = object_names[j]
                prev_size = object_sizes.get(prev_name, 0.05)
                if check_overlap(
                    candidate_pos,
                    obj_size,
                    positions[j],
                    prev_size,
                    buffer=min_distance,
                ):
                    overlap = True
                    break

            if not overlap:
                positions[i] = candidate_pos
                break
        else:
            print(
                f"Warning: Could not place object {name} without overlap after 100 attempts."
            )
            # Fallback to last sampled position or some default
            positions[i] = candidate_pos

    # Apply positions to simulation
    for i, name in enumerate(object_names):
        obj = scene[name]
        # Assuming obj is a RigidObject or similar that has write_root_pose_to_sim
        # We need to get the current orientation to preserve it, or reset it
        # Here we assume we just set position and keep default/current orientation
        # But write_root_pose_to_sim usually expects pose (pos + quat)

        # Get current root state to extract orientation
        # Note: This might be slow if done one by one, but fine for initialization
        current_pose = obj.data.root_link_pose_w[0].clone()  # (7,)
        new_pose = current_pose.clone()
        new_pose[:3] = positions[i]

        # Write to sim
        # write_root_pose_to_sim expects (num_instances, 7)
        obj.write_root_pose_to_sim(new_pose.unsqueeze(0))

    return positions


def reset_to_random_robot_pose(
    robot: "Articulation", joint_ranges: dict[str, tuple[float, float]]
):
    """
    Reset robot joints to random positions within specified ranges.

    Args:
        robot: The robot articulation.
        joint_ranges: Dictionary mapping joint names (or regex) to (min, max) ranges.
    """
    device = robot.device
    joint_pos = robot.data.default_joint_pos.clone()

    # We need to map joint names to indices.
    # robot.joint_names gives a list of names.

    for joint_name, (min_val, max_val) in joint_ranges.items():
        # Find indices matching the name (simple exact match or substring for now)
        # For a robust implementation, we might use regex if keys are regexes
        indices = [i for i, name in enumerate(robot.joint_names) if joint_name in name]

        for idx in indices:
            val = torch.rand(1, device=device) * (max_val - min_val) + min_val
            joint_pos[:, idx] = val

    robot.write_joint_position_to_sim(joint_pos)
    robot.write_joint_velocity_to_sim(torch.zeros_like(joint_pos))
