from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UR3_URDF_PATH = PROJECT_ROOT / "lerobot_robot_sim_ur3e" / "assets" / "ur_description" / "urdf" / "ur3.urdf"
DEFAULT_UR3_PACKAGE_DIR = PROJECT_ROOT / "lerobot_robot_sim_ur3e" / "assets"
UR3_ARM_DOFS = 6

DEFAULT_MAX_JOINT_STEP = 0.05
DEFAULT_COLLISION_MARGIN = 0.0
DEFAULT_LINE_SEARCH_STEPS = 10

DEFAULT_IGNORED_COLLISION_PAIRS: tuple[tuple[str, str], ...] = (
    ("base_link_inertia_0", "shoulder_link_0"),
    ("shoulder_link_0", "upper_arm_link_0"),
    ("upper_arm_link_0", "forearm_link_0"),
    ("forearm_link_0", "wrist_1_link_0"),
    ("wrist_1_link_0", "wrist_2_link_0"),
    ("wrist_2_link_0", "wrist_3_link_0"),
)
