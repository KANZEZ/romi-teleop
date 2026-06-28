"""Configuration dataclass for the bimanual simulated UR5e robot plugin."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot_camera_mujoco import MujocoCameraConfig  # noqa: F401

## this start joint is ONLY for simulation and does not correspond to any real-world recommended start joint for the UR5e.
# It is chosen to be a reasonably "open" configuration that keeps the arms away from each other.
LEFT_UR5E_START_JOINTS = (
    0.0,
    -np.pi / 2,
    np.pi / 2,
    -np.pi / 2,
    -np.pi / 2,
    0.0,
    0.0,
)
RIGHT_UR5E_START_JOINTS = (
    np.pi,
    -np.pi / 2,
    -np.pi / 2,
    -np.pi / 2,
    np.pi / 2,
    0.0,
    0.0,
)


def default_cameras() -> dict[str, CameraConfig]:
    return {
        "global": MujocoCameraConfig(camera="global", width=640, height=480, fps=30),
        "left_wrist": MujocoCameraConfig(camera="left_wrist", width=640, height=480, fps=30),
        "right_wrist": MujocoCameraConfig(camera="right_wrist", width=640, height=480, fps=30),
    }


@RobotConfig.register_subclass("sim_bi_ur5e")
@dataclass
class SimBiUR5EConfig(RobotConfig):
    left_start_joints: tuple[float, ...] = LEFT_UR5E_START_JOINTS
    right_start_joints: tuple[float, ...] = RIGHT_UR5E_START_JOINTS

    cameras: dict[str, CameraConfig] = field(default_factory=default_cameras)

    show_viewer: bool = False
    command_substeps: int = 6
    gripper_command_substeps: int = 120

    project_root: Path | None = None
    ur5e_xml_path: Path | None = None
    robotiq_xml_path: Path | None = None
    extra_backend_kwargs: dict = field(default_factory=dict)

    def left_start_joints_array(self) -> np.ndarray:
        return np.asarray(self.left_start_joints, dtype=float)

    def right_start_joints_array(self) -> np.ndarray:
        return np.asarray(self.right_start_joints, dtype=float)
