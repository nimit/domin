import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

from domin.utils import xyz_to_quat


@configclass
class DexSuiteSceneCfg(InteractiveSceneCfg):
    """Dexsuite Scene for pick and place with Kuka Allegro"""

    # defaults
    env_spacing: float = 3.0

    # robot
    robot: ArticulationCfg = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Table with distinguishing visual material
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.5, 0, 0), rot=(0.707, 0, 0, 0.707)
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.1, 0.1),  # Dark table
                roughness=0.8,
                metallic=0.2,
            ),
        ),
    )

    # Simple Object: Cube
    object_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, 0.05, 0.05), rot=(1, 0, 0, 0)
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 0.08),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),  # Red cube
                metallic=0.2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # Simple Object: Sphere
    # object_sphere = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object_Sphere",
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0.6, 0.3, 0.05), rot=(1, 0, 0, 0)
    #     ),
    #     spawn=sim_utils.SphereCfg(
    #         radius=0.03,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.0, 1.0, 0.0),  # Green sphere
    #             metallic=0.2,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    # )

    # Light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # cameras
    camera_front = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera_Front",
        update_period=0.1,
        height=400,
        width=400,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.35, -2.0, 1.25),
            rot=tuple(xyz_to_quat(0, 25, 90).tolist()),
            convention="world",
        ),
    )

    camera_left = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera_Left",
        update_period=0.1,
        height=400,
        width=400,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-2.15, 0.0, 1.75),
            rot=tuple(xyz_to_quat(0, 30, 0).tolist()),
            convention="world",
        ),
    )

    camera_right = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera_Right",
        update_period=0.1,
        height=400,
        width=400,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(2.85, 0.0, 1.75),
            rot=tuple(xyz_to_quat(0, 30, 180).tolist()),
            convention="world",
        ),
    )

    camera_top = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera_Top",
        update_period=0.1,
        height=400,
        width=400,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=1.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.35, 0.0, 3.0),
            rot=tuple(xyz_to_quat(0, 90, 0).tolist()),
            convention="world",
        ),
    )
