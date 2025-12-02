from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import CameraCfg


@configclass
class TestSceneCfg(InteractiveSceneCfg):
    """Configuration for a test scene."""

    # defaults
    num_envs: int = 2
    env_spacing: float = 5.0

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # Table
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
    #         scale=(1.0, 1.0, 1.0),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.25, -0.25, 0.0)),
    # )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0, 0), rot=(0.707, 0, 0, 0.707)
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(2.5, 2.5, 2.5),
        ),
    )

    object_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_Cube",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.5, 0.5, 0.055), rot=(1, 0, 0, 0)
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    object_beaker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_Beaker",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.2, 0.0, 0.055), rot=(1, 0, 0, 0)
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Beaker/beaker_500ml.usd",
            scale=(0.60, 0.60, 0.60),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 1.0), metallic=0.2
            ),
        ),
    )

    object_cone = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_Cone",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.3, 0.2, 0.055), rot=(1, 0, 0, 0)
        ),
        spawn=sim_utils.ConeCfg(
            radius=0.10,
            height=0.10,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0), metallic=0.2
            ),
        ),
    )

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
            pos=(0.0, -1.5, 0.35),
            rot=(0.6533, -0.2706, 0.2706, 0.6533),
            convention="world",
        ),
    )
