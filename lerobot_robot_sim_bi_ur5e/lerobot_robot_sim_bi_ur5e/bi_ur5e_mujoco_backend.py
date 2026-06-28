"""MuJoCo backend for the bimanual simulated UR5e robot."""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np
from lerobot.cameras import CameraConfig
from lerobot_camera_mujoco import MujocoCamera, MujocoCameraConfig

from .build_bi_ur5e_mujoco_env import build_bi_ur5e_mujoco_env
from .config_sim_bi_ur5e import LEFT_UR5E_START_JOINTS, RIGHT_UR5E_START_JOINTS

logger = logging.getLogger(__name__)

UR5E_ARM_DOFS = 6
UR5E_INTERFACE_DOFS = 7
UR5E_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
UR5E_ARM_ACTUATOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)
ROBOTIQ_CTRL_MAX = 255.0
ROBOTIQ_DRIVER_CLOSED = 0.9
GRIPPER_POSITION_EPS = 0.01
DEFAULT_COMMAND_SUBSTEPS = 6
DEFAULT_GRIPPER_COMMAND_SUBSTEPS = 120


def _project_root(project_root: str | Path | None) -> Path:
    return Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]


def _default_ur5e_path(project_root: Path) -> Path:
    return (
        project_root
        / "lerobot_robot_sim_bi_ur5e"
        / "assest"
        / "mujoco_menagerie"
        / "universal_robots_ur5e"
        / "ur5e.xml"
    )


def _default_robotiq_path(project_root: Path) -> Path:
    return (
        project_root
        / "lerobot_robot_sim_bi_ur5e"
        / "assest"
        / "mujoco_menagerie"
        / "robotiq_2f85_v4"
        / "2f85.xml"
    )


def _validate_asset_paths(ur5e_xml_path: Path, robotiq_xml_path: Path) -> None:
    missing = []
    if not ur5e_xml_path.is_file():
        missing.append(f"UR5e MJCF: {ur5e_xml_path}")
    if not robotiq_xml_path.is_file():
        missing.append(f"Robotiq MJCF: {robotiq_xml_path}")
    if missing:
        raise FileNotFoundError("Missing MuJoCo simulation assets:\n" + "\n".join(missing))


def _object_names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, object_type, object_id) or "" for object_id in range(count)]


def _matches_namespaced_name(object_name: str, local_name: str, namespaces: Sequence[str]) -> bool:
    if not namespaces:
        return object_name == local_name or object_name.endswith(f"/{local_name}")

    return any(
        object_name == f"{namespace}/{local_name}"
        or (object_name.startswith(f"{namespace}/") and object_name.endswith(f"/{local_name}"))
        for namespace in namespaces
    )


def _find_namespaced_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    count: int,
    local_name: str,
    namespaces: Sequence[str] = (),
) -> int:
    names = _object_names(model, object_type, count)
    for object_id, object_name in enumerate(names):
        if _matches_namespaced_name(object_name, local_name, namespaces):
            return object_id
    raise ValueError(
        f"{object_type.name} {local_name!r} not found in namespaces {list(namespaces)!r}. "
        f"Available: {names}"
    )


def _find_camera_id(model: mujoco.MjModel, name: str) -> int:
    return _find_namespaced_id(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam, name)


def _quat_from_mat(matrix: np.ndarray) -> np.ndarray:
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mju_mat2Quat(quat, matrix.reshape(-1))
    return quat


@dataclass(frozen=True)
class _ArmHandles:
    arm_qpos_adrs: np.ndarray
    arm_dof_adrs: np.ndarray
    arm_actuator_ids: np.ndarray
    fingers_actuator_id: int
    right_driver_qpos_adr: int
    wrist_body_id: int

    @classmethod
    def from_model(cls, model: mujoco.MjModel, side: str) -> "_ArmHandles":
        arm_namespaces = (f"{side}_ur5e",)
        gripper_namespaces = (f"{side}_ur5e", f"{side}_robotiq")
        joint_ids = [
            _find_namespaced_id(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt, name, arm_namespaces)
            for name in UR5E_ARM_JOINT_NAMES
        ]
        actuator_ids = [
            _find_namespaced_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu, name, arm_namespaces)
            for name in UR5E_ARM_ACTUATOR_NAMES
        ]
        fingers_actuator_id = _find_namespaced_id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            model.nu,
            "fingers_actuator",
            gripper_namespaces,
        )
        right_driver_joint_id = _find_namespaced_id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            model.njnt,
            "right_driver_joint",
            gripper_namespaces,
        )
        wrist_body_id = _find_namespaced_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            model.nbody,
            "wrist_3_link",
            arm_namespaces,
        )

        return cls(
            arm_qpos_adrs=model.jnt_qposadr[np.asarray(joint_ids, dtype=int)],
            arm_dof_adrs=model.jnt_dofadr[np.asarray(joint_ids, dtype=int)],
            arm_actuator_ids=np.asarray(actuator_ids, dtype=int),
            fingers_actuator_id=fingers_actuator_id,
            right_driver_qpos_adr=int(model.jnt_qposadr[right_driver_joint_id]),
            wrist_body_id=wrist_body_id,
        )


@dataclass
class _ArmState:
    joints: np.ndarray
    command: np.ndarray
    velocities: np.ndarray
    ee_pos_quat: np.ndarray

    @classmethod
    def from_start_joints(cls, start_joints: np.ndarray) -> "_ArmState":
        joints = _validate_joint_state(start_joints, "start_joints")
        return cls(
            joints=joints,
            command=joints.copy(),
            velocities=np.zeros(UR5E_INTERFACE_DOFS, dtype=float),
            ee_pos_quat=np.zeros(7, dtype=float),
        )


def _validate_joint_state(joint_state: np.ndarray, label: str) -> np.ndarray:
    joint_state = np.asarray(joint_state, dtype=float).copy()
    if joint_state.shape != (UR5E_INTERFACE_DOFS,):
        raise ValueError(f"Expected {label} shape {(UR5E_INTERFACE_DOFS,)}, got {joint_state.shape}.")
    joint_state[-1] = float(np.clip(joint_state[-1], 0.0, 1.0))
    return joint_state


class BiUR5EMujocoBackend:
    def __init__(
        self,
        left_start_joints: np.ndarray | None = None,
        right_start_joints: np.ndarray | None = None,
        camera_configs: dict[str, CameraConfig] | None = None,
        project_root: str | Path | None = None,
        ur5e_xml_path: str | Path | None = None,
        robotiq_xml_path: str | Path | None = None,
        show_viewer: bool = False,
        command_substeps: int = DEFAULT_COMMAND_SUBSTEPS,
        gripper_command_substeps: int = DEFAULT_GRIPPER_COMMAND_SUBSTEPS,
    ) -> None:
        project_root = _project_root(project_root)
        ur5e_xml_path = Path(ur5e_xml_path) if ur5e_xml_path is not None else _default_ur5e_path(project_root)
        robotiq_xml_path = Path(robotiq_xml_path) if robotiq_xml_path is not None else _default_robotiq_path(project_root)
        _validate_asset_paths(ur5e_xml_path, robotiq_xml_path)

        self._model = build_bi_ur5e_mujoco_env(ur5e_xml_path=ur5e_xml_path, robotiq_xml_path=robotiq_xml_path)
        self._model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self._data = mujoco.MjData(self._model)
        self._state_lock = threading.Lock()
        self._command_substeps = int(command_substeps)
        self._gripper_command_substeps = int(gripper_command_substeps)
        if self._command_substeps < 1 or self._gripper_command_substeps < 1:
            raise ValueError("MuJoCo command substeps must be positive.")

        self._left_handles = _ArmHandles.from_model(self._model, "left")
        self._right_handles = _ArmHandles.from_model(self._model, "right")
        self._left = _ArmState.from_start_joints(
            np.asarray(left_start_joints if left_start_joints is not None else LEFT_UR5E_START_JOINTS, dtype=float)
        )
        self._right = _ArmState.from_start_joints(
            np.asarray(right_start_joints if right_start_joints is not None else RIGHT_UR5E_START_JOINTS, dtype=float)
        )
        self._viewer = None
        self._cameras: list[tuple[str, MujocoCamera]] = []
        self._stopped = False

        with self._state_lock:
            self._initialize_arm_locked(self._left_handles, self._left.command)
            self._initialize_arm_locked(self._right_handles, self._right.command)
            mujoco.mj_forward(self._model, self._data)
            self._apply_joint_cmd_locked(substeps=100)

        self._configure_cameras(camera_configs or {})
        if show_viewer:
            self._launch_viewer()

    def _assert_running(self) -> None:
        if self._stopped:
            raise RuntimeError("Bimanual UR5e MuJoCo backend has been stopped.")

    def _initialize_arm_locked(self, handles: _ArmHandles, command: np.ndarray) -> None:
        self._data.qpos[handles.arm_qpos_adrs] = command[:UR5E_ARM_DOFS]
        self._data.qvel[handles.arm_dof_adrs] = 0.0
        self._data.ctrl[handles.arm_actuator_ids] = command[:UR5E_ARM_DOFS]
        self._data.ctrl[handles.fingers_actuator_id] = float(command[-1] * ROBOTIQ_CTRL_MAX)

    def _configure_cameras(self, camera_configs: dict[str, CameraConfig]) -> None:
        self._assert_running()
        for camera_key, camera_config in camera_configs.items():
            if not isinstance(camera_config, MujocoCameraConfig):
                raise TypeError(
                    f"Sim bimanual UR5e cameras must use type 'mujoco', got "
                    f"{getattr(camera_config, 'type', type(camera_config).__name__)!r} for {camera_key!r}."
                )
            camera_name = camera_config.camera if camera_config.camera is not None else camera_key
            camera = MujocoCamera(camera_config)
            camera.bind(self._model, self._data, _find_camera_id(self._model, str(camera_name)))
            camera.connect()
            self._cameras.append((camera_key, camera))

    def _launch_viewer(self) -> None:
        self._assert_running()
        try:
            import mujoco.viewer
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Could not import mujoco.viewer. Install MuJoCo viewer dependencies.") from exc

        self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
        logger.info("MuJoCo viewer launched.")

    def _sync_viewer_locked(self) -> None:
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def is_viewer_running(self) -> bool:
        return self._viewer is None or self._viewer.is_running()

    def _current_gripper_position_locked(self, handles: _ArmHandles) -> float:
        driver_pos = float(self._data.qpos[handles.right_driver_qpos_adr])
        return float(np.clip(driver_pos / ROBOTIQ_DRIVER_CLOSED, 0.0, 1.0))

    def _ee_pos_quat_locked(self, handles: _ArmHandles) -> np.ndarray:
        ee_pos = self._data.xpos[handles.wrist_body_id].copy()
        ee_mat = self._data.xmat[handles.wrist_body_id].reshape(3, 3)
        return np.concatenate([ee_pos, _quat_from_mat(ee_mat)])

    def _update_arm_observations_locked(self, handles: _ArmHandles, state: _ArmState) -> None:
        state.joints = np.concatenate(
            [
                self._data.qpos[handles.arm_qpos_adrs].copy(),
                [self._current_gripper_position_locked(handles)],
            ]
        )
        state.velocities[:UR5E_ARM_DOFS] = self._data.qvel[handles.arm_dof_adrs].copy()
        state.ee_pos_quat = self._ee_pos_quat_locked(handles)

    def _update_observations_locked(self) -> None:
        self._update_arm_observations_locked(self._left_handles, self._left)
        self._update_arm_observations_locked(self._right_handles, self._right)

    def _set_arm_controls_locked(self, handles: _ArmHandles, command: np.ndarray) -> None:
        self._data.ctrl[handles.arm_actuator_ids] = command[:UR5E_ARM_DOFS]
        self._data.ctrl[handles.fingers_actuator_id] = float(command[-1] * ROBOTIQ_CTRL_MAX)

    def _remember_grippers(self) -> tuple[float, float]:
        return float(self._left.joints[-1]), float(self._right.joints[-1])

    def _update_gripper_velocities(self, previous_grippers: tuple[float, float]) -> None:
        self._left.velocities[-1] = self._left.joints[-1] - previous_grippers[0]
        self._right.velocities[-1] = self._right.joints[-1] - previous_grippers[1]

    def _apply_joint_cmd_locked(self, substeps: int) -> None:
        previous_grippers = self._remember_grippers()
        for _ in range(max(int(substeps), 1)):
            self._set_arm_controls_locked(self._left_handles, self._left.command)
            self._set_arm_controls_locked(self._right_handles, self._right.command)
            mujoco.mj_step(self._model, self._data)
        self._update_observations_locked()
        self._update_gripper_velocities(previous_grippers)
        self._sync_viewer_locked()

    def step(self, substeps: int = 1, render_cameras: bool = False) -> None:
        """Advance simulation without overwriting viewer-edited actuator controls."""
        self._assert_running()
        with self._state_lock:
            previous_grippers = self._remember_grippers()
            for _ in range(max(int(substeps), 1)):
                mujoco.mj_step(self._model, self._data)
            self._update_observations_locked()
            self._update_gripper_velocities(previous_grippers)
            if render_cameras:
                self._render_cameras_locked()
            self._sync_viewer_locked()

    def command_joint_state(self, left_joint_state: np.ndarray, right_joint_state: np.ndarray) -> None:
        self._assert_running()
        left_joint_state = _validate_joint_state(left_joint_state, "left_joint_state")
        right_joint_state = _validate_joint_state(right_joint_state, "right_joint_state")

        with self._state_lock:
            gripper_gap = max(
                abs(float(left_joint_state[-1]) - float(self._left.joints[-1])),
                abs(float(right_joint_state[-1]) - float(self._right.joints[-1])),
            )
            substeps = (
                self._gripper_command_substeps
                if gripper_gap > GRIPPER_POSITION_EPS
                else self._command_substeps
            )
            self._left.command = left_joint_state.copy()
            self._right.command = right_joint_state.copy()
            self._apply_joint_cmd_locked(substeps=substeps)

    def get_observations(self) -> dict[str, np.ndarray]:
        self._assert_running()
        with self._state_lock:
            return {
                "left_joint_positions": self._left.joints.copy(),
                "right_joint_positions": self._right.joints.copy(),
                "left_joint_velocities": self._left.velocities.copy(),
                "right_joint_velocities": self._right.velocities.copy(),
                "left_ee_pos_quat": self._left.ee_pos_quat.copy(),
                "right_ee_pos_quat": self._right.ee_pos_quat.copy(),
                "left_gripper_position": np.array([self._left.joints[-1]], dtype=float),
                "right_gripper_position": np.array([self._right.joints[-1]], dtype=float),
            }

    def _render_cameras_locked(self) -> None:
        for _, camera in self._cameras:
            camera.render(self._data)

    def get_camera_observations(self) -> dict[str, np.ndarray]:
        self._assert_running()
        with self._state_lock:
            self._render_cameras_locked()
            return {camera_name: camera.read() for camera_name, camera in self._cameras}

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        for _, camera in self._cameras:
            camera.disconnect()
        self._cameras.clear()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
