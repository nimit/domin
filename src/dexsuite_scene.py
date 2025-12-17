from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab_assets.robots import KUKA_ALLEGRO_CFG
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


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
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -1.05)
        ),  # Adjust based on table height if needed, but table is usually at 0?
        # Wait, SeattleLabTable usually has surface at some height.
        # If table is at (0.5, 0, 0), the surface is likely at z > 0.
        # Let's assume table surface is around z=0 if we want robot to interact easily,
        # but usually we place table on ground.
        # If ground is at -1.05, table base is at -1.05?
        # Let's put ground at 0 and table on ground.
    )
