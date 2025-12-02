"""
Simulation controller for managing the simulation loop.
"""

import argparse
import torch
import time
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.app import AppLauncher
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.assets.articulation import Articulation
from isaaclab.assets.rigid_object import RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from .base_dataset_config import BaseDatasetConfig
from .sim_state import SimProps, SimState


# TODO (feat): Keyboard events (with toggle)


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

    # TODO (CRITICAL)
    # using config.is_success
    # 0. check config.is_success and depending on the result,
    #     - if mode==eval, just update statistics based on string key
    #     - if mode==generation, retry same episode if failure, new episode if success + write statistics

    def save_props(self):
        """
        Save properties (`SimProps`) that are not changing throughout the simulation.
        """
        self.cameras = {
            k: v.cfg
            for k, v in self.scene.sensors.items()
            if v is isinstance(v, Camera)
        }
        self.objects = {
            k: v for k, v in self.scene.rigid_objects.items() if "object" in k
        }
        self.robot = self.scene.articulations["robot"]

        rjoint_lims = self.robot.data.joint_pos_limits
        # TODO (CRITICAL): fix
        objs_size = {k: torch.tensor([0.3, 0.3, 0.3]) for k in self.objects.keys()}
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

    def reset(self, next=True):
        # reset should:
        # 1. sim reset?
        # 2. set object positions to next episode's position (if next else restart from same position)
        # 3. if mode is generation, self.dataset.new_episode
        # 4. let simulation update for x number of steps (to reach the new positions; x is hardcoded right now, default/parameterized in config)

        # self.sim.reset()

        new_poses = self.config.get_random_object_pose(self.props)
        for name, obj in self.objects.items():
            obj.write_root_link_pose_to_sim(new_poses[name] + self.scene.env_origins)

        # new episode

        for _ in range(10):
            self.scene.update(self.sim.get_physics_dt())

    def record_dataset(self):
        """
        Main loop to run the simulation.
        Handles both recording and inference loops based on mode.
        """

        total_step_time = 0
        total_update_time = 0
        record_ds = time.perf_counter()

        assert self.simulation_app.is_running()

        from .dataset_builder import DatasetRecord, DatasetRecordConfig

        # have to modify dataset recorder to allow recording of multiple episodes simultaneously
        # rec_cfg = DatasetRecordConfig(
        #     repo_id=self.config.hf_repo_id,
        #     # TODO: get from config
        #     robot_type="franka_panda",
        #     # TODO: default task from config
        #     default_task="default_task",
        #     joint_names=self.scene["robot"].joint_names,
        #     cameras={k: (v.width, v.height) for k, v in self.cameras.items()},
        #     root=self.config.dataset_path,
        #     # TODO: from config/args
        #     num_episodes=self.config.num_envs * 10,
        #     push_to_hub=False,
        # )
        # self.dataset = DatasetRecord(rec_cfg)
        # self.dataset.start()

        # TODO: multiple targets
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        diff_ik_controller = DifferentialIKController(
            diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device
        )

        # TODO (CRITICAL): THIS IS ONLY TEMPORARY. FIGURE OUT A BETTER SOLUTION FOR IK
        robot_entity_cfg = SceneEntityCfg(
            "robot",
            joint_names=[".*"],
            body_names=[".*finger"],
            preserve_order=True,
        )
        robot_entity_cfg.resolve(self.scene)  # Forces name-to-ID resolution
        ee_body_id = robot_entity_cfg.body_ids[0]  # type: ignore

        try:
            while self.simulation_app.is_running():
                for _ in range(50):
                    self.sim.step()
                    self.scene.update(self.sim.get_physics_dt())
                print("start took", time.perf_counter() - record_ds)

                targets = self.config.get_targets(self.get_state())
                # print("TARGETS:", targets)
                diff_ik_controller.set_command(targets)

                for step in range(500):
                    # 2. Apply actions (Stub: just keeping current pos or zero actions)
                    # robot = self.scene["robot"]
                    # robot.set_joint_position_target(...)

                    # 3. Step simulation
                    jacobian = self.robot.root_physx_view.get_jacobians()[
                        :, ee_body_id - 1, :, robot_entity_cfg.joint_ids
                    ]
                    root_pose_w = self.robot.data.root_pose_w
                    joint_pos = self.robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
                    ee_pose_w = self.robot.data.body_state_w[:, ee_body_id, 0:7]
                    ee_pos_b, ee_quat_b = subtract_frame_transforms(
                        root_pose_w[:, 0:3],
                        root_pose_w[:, 3:7],
                        ee_pose_w[:, 0:3],
                        ee_pose_w[:, 3:7],
                    )

                    joint_pos_des = diff_ik_controller.compute(
                        ee_pos_b, ee_quat_b, jacobian, joint_pos
                    )

                    self.robot.set_joint_position_target(
                        joint_pos_des, joint_ids=robot_entity_cfg.joint_ids
                    )
                    self.scene.write_data_to_sim()
                    step_time = time.perf_counter()
                    self.sim.step()
                    total_step_time += time.perf_counter() - step_time

                    update_time = time.perf_counter()
                    self.scene.update(self.sim.get_physics_dt())
                    total_update_time += time.perf_counter() - update_time
                    # can control viewing fps with sleep (TODO if not headless)
                    # time.sleep(0.05)

                print("record_ds took", time.perf_counter() - record_ds)
                print("total_step_time:", total_step_time)
                print("total_update_time:", total_update_time)
                self._close()

        # 5. Handle resets
        # if done: ...

        except KeyboardInterrupt:
            print("Simulation stopped by user.")
        finally:
            # self.dataset.end()
            self._close()

    def evaluate():
        pass

    def _close(self):
        """
        Clean up resources.
        """
        if self.simulation_app and self.simulation_app.is_running():
            self.simulation_app.close()
        print("Simulation closed.")
