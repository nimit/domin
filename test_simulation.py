import argparse
import torch
import numpy as np
from isaaclab.app import AppLauncher

# Create the parser and add AppLauncher args
parser = argparse.ArgumentParser(description="Test Simulation Controller")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Initialize AppLauncher first!
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now imports
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from test_dataset_config import TestDatasetConfig
from src.simulation_controller import SimulationController
from src.utils import calculate_distance, xyz_to_quat, will_overlap


def test_utils():
    print("Testing utility functions...")

    # Test calculate_distance
    pos1 = np.array([0, 0, 0])
    pos2 = np.array([1, 0, 0])
    assert calculate_distance(pos1, pos2) == 1.0, "calculate_distance failed"

    # Test xyz_to_quat
    q = xyz_to_quat(0, 0, 0)
    expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert torch.allclose(q, expected, atol=1e-5), f"xyz_to_quat(0,0,0) failed: {q}"

    q_z90 = xyz_to_quat(0, 0, 90)
    expected_z90 = torch.tensor([0.7071068, 0.0, 0.0, 0.7071068])
    assert torch.allclose(q_z90, expected_z90, atol=1e-5), (
        f"xyz_to_quat(0,0,90) failed: {q_z90}"
    )

    # Test check_overlap
    # pos1_t = torch.tensor([0.0, 0.0, 0.0])
    # size1 = 0.1
    # pos2_t = torch.tensor([0.15, 0.0, 0.0])
    # size2 = 0.1
    # assert check_overlap(pos1_t, size1, pos2_t, size2), (
    #     "check_overlap failed (should overlap)"
    # )

    # pos3_t = torch.tensor([0.3, 0.0, 0.0])
    # assert not check_overlap(pos1_t, size1, pos3_t, size2), (
    #     "check_overlap failed (should not overlap)"
    # )

    print("Utility functions passed.")


def main():
    # Run unit tests for utils first (passed!. can check again if there are any changes later)
    # test_utils()

    # Create config
    dataset_cfg = TestDatasetConfig()

    # Create controller with existing app_launcher
    controller = SimulationController(dataset_cfg, app_launcher=app_launcher)
    print("Simulation Controller initialized successfully.")

    # Run simulation (short run)
    # We need to modify run to stop after a few steps or handle it manually
    # But for now let's just run it and expect user to Ctrl+C or we can mock run?
    # Actually, let's just print that we initialized successfully.
    # If we call run(), it enters a loop.
    # We can override run or just call internal methods if we want to verify setup.

    controller.record_dataset()

    # Close
    controller._close()


if __name__ == "__main__":
    main()
