"""Configuration dataclass for the simulated UR3e robot plugin."""

from dataclasses import dataclass, field

import numpy as np
from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot_camera_mujoco import MujocoCameraConfig  # noqa: F401


DEFAULT_START_JOINTS = (1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 1.5708, 0.0)


@RobotConfig.register_subclass("sim_ur3e")
@dataclass
class SimUR3EConfig(RobotConfig):
    start_joints: tuple[float, ...] = DEFAULT_START_JOINTS

    collision_debug: bool = False

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    show_viewer: bool = False
    command_substeps: int = 6
    gripper_command_substeps: int = 120
    extra_backend_kwargs: dict = field(default_factory=dict)

    def start_joints_array(self) -> np.ndarray:
        return np.asarray(self.start_joints, dtype=float)
