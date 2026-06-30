"""LeRobot interface for the MuJoCo UR3e + Robotiq 2F-85 simulation."""

import logging
from functools import cached_property
from typing import Any

import numpy as np
from lerobot.robots import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_sim_ur3e import SimUR3EConfig

logger = logging.getLogger(__name__)


class SimUR3E(Robot):
    config_class = SimUR3EConfig
    name = "sim_ur3e"

    def __init__(self, config: SimUR3EConfig):
        super().__init__(config)
        self.config = config
        self.backend = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "joint_0": float,
            "joint_1": float,
            "joint_2": float,
            "joint_3": float,
            "joint_4": float,
            "joint_5": float,
            "gripper": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            camera_name: (camera_config.height, camera_config.width, 3)
            for camera_name, camera_config in self.config.cameras.items()
        }

    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.backend is not None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            from .ur3e_mujoco_server import UR3MujocoBackend
        except Exception as exc:
            raise RuntimeError(
                "Failed to import the UR3e MuJoCo backend. Install the MuJoCo simulation "
                "dependencies before connecting."
            ) from exc

        self.backend = UR3MujocoBackend(
            start_joints=self.config.start_joints_array(),
            collision_debug=self.config.collision_debug,
            camera_configs=self.config.cameras,
            show_viewer=self.config.show_viewer,
            command_substeps=self.config.command_substeps,
            gripper_command_substeps=self.config.gripper_command_substeps,
            **self.config.extra_backend_kwargs,
        )

        logger.info(f"{self} connected.")

    def configure(self) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def _require_backend(self):
        if self.backend is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        return self.backend

    def get_observation(self) -> dict[str, Any]:
        backend = self._require_backend()
        sim_obs = backend.get_observations()
        joint_positions = np.asarray(sim_obs["joint_positions"], dtype=float)

        obs_dict = {f"joint_{i}": float(joint_positions[i]) for i in range(6)}
        obs_dict["gripper"] = float(joint_positions[6])

        if self.config.cameras:
            obs_dict.update(backend.get_camera_observations())

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        backend = self._require_backend()
        if set(action) != set(self.action_features):
            raise ValueError(f"Invalid action: {action}, features: {self.action_features}")

        joint_state = np.array([action[f"joint_{i}"] for i in range(6)] + [action["gripper"]], dtype=float)
        joint_state[-1] = float(np.clip(joint_state[-1], 0.0, 1.0))
        backend.command_joint_state(joint_state)

        sent_action = {f"joint_{i}": float(joint_state[i]) for i in range(6)}
        sent_action["gripper"] = float(joint_state[6])
        return sent_action

    def move_to_start_joints(self, wait: bool = True) -> None:
        del wait
        backend = self._require_backend()
        backend.command_joint_state(self.config.start_joints_array())

    def disconnect(self) -> None:
        if self.backend is None:
            return

        self.backend.stop()
        self.backend = None
        logger.info(f"{self} disconnected.")
