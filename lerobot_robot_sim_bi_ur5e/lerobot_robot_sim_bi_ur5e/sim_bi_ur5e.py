"""LeRobot interface for the bimanual MuJoCo UR5e simulation."""

import logging
from functools import cached_property
from typing import Any

import numpy as np
from lerobot.robots import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_sim_bi_ur5e import SimBiUR5EConfig

logger = logging.getLogger(__name__)


class SimBiUR5E(Robot):
    config_class = SimBiUR5EConfig
    name = "sim_bi_ur5e"

    def __init__(self, config: SimBiUR5EConfig):
        super().__init__(config)
        self.config = config
        self.backend = None
        self.cameras = config.cameras

    @property
    def _single_arm_motors_ft(self) -> dict[str, type]:
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
    def _motors_ft(self) -> dict[str, type]:
        return {
            **{f"left_{key}": value for key, value in self._single_arm_motors_ft.items()},
            **{f"right_{key}": value for key, value in self._single_arm_motors_ft.items()},
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            camera_name: (camera_config.height, camera_config.width, 3)
            for camera_name, camera_config in self.config.cameras.items()
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.backend is not None

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            from .bi_ur5e_mujoco_backend import BiUR5EMujocoBackend
        except Exception as exc:
            raise RuntimeError(
                "Failed to import the bimanual UR5e MuJoCo backend. Install the MuJoCo "
                "simulation dependencies before connecting."
            ) from exc

        self.backend = BiUR5EMujocoBackend(
            left_start_joints=self.config.left_start_joints_array(),
            right_start_joints=self.config.right_start_joints_array(),
            camera_configs=self.config.cameras,
            show_viewer=self.config.show_viewer,
            command_substeps=self.config.command_substeps,
            gripper_command_substeps=self.config.gripper_command_substeps,
            project_root=self.config.project_root,
            ur5e_xml_path=self.config.ur5e_xml_path,
            robotiq_xml_path=self.config.robotiq_xml_path,
            **self.config.extra_backend_kwargs,
        )
        logger.info("%s connected.", self)

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
        left_positions = np.asarray(sim_obs["left_joint_positions"], dtype=float)
        right_positions = np.asarray(sim_obs["right_joint_positions"], dtype=float)

        obs_dict = {}
        obs_dict.update({f"left_joint_{i}": float(left_positions[i]) for i in range(6)})
        obs_dict["left_gripper"] = float(left_positions[6])
        obs_dict.update({f"right_joint_{i}": float(right_positions[i]) for i in range(6)})
        obs_dict["right_gripper"] = float(right_positions[6])

        if self.config.cameras:
            obs_dict.update(backend.get_camera_observations())

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        backend = self._require_backend()
        if set(action) != set(self.action_features):
            raise ValueError(f"Invalid action: {action}, features: {self.action_features}")

        left_state = np.array(
            [action[f"left_joint_{i}"] for i in range(6)] + [action["left_gripper"]],
            dtype=float,
        )
        right_state = np.array(
            [action[f"right_joint_{i}"] for i in range(6)] + [action["right_gripper"]],
            dtype=float,
        )
        left_state[-1] = float(np.clip(left_state[-1], 0.0, 1.0))
        right_state[-1] = float(np.clip(right_state[-1], 0.0, 1.0))
        backend.command_joint_state(left_state, right_state)

        sent_action = {f"left_joint_{i}": float(left_state[i]) for i in range(6)}
        sent_action["left_gripper"] = float(left_state[6])
        sent_action.update({f"right_joint_{i}": float(right_state[i]) for i in range(6)})
        sent_action["right_gripper"] = float(right_state[6])
        return sent_action

    def move_to_start_joints(self, wait: bool = True) -> None:
        del wait
        backend = self._require_backend()
        backend.command_joint_state(
            self.config.left_start_joints_array(),
            self.config.right_start_joints_array(),
        )

    def disconnect(self) -> None:
        if self.backend is None:
            return

        self.backend.stop()
        self.backend = None
        logger.info("%s disconnected.", self)
