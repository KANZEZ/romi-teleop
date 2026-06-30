"""MuJoCo backend utilities for the simulated UR3e robot."""

import logging
import shutil
import tempfile
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from dm_control import mjcf
from lerobot.cameras import CameraConfig
from lerobot_camera_mujoco import MujocoCamera, MujocoCameraConfig

if TYPE_CHECKING:
    from .safety.ur3_self_collision import UR3SelfCollisionChecker

logger = logging.getLogger(__name__)

DEFAULT_START_JOINTS = np.array(
    [1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 1.5708, 0.0],
    dtype=float,
)
UR3_ARM_DOFS = 6
UR3_INTERFACE_DOFS = 7
UR3_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ARM_ACTUATOR_NAMES = (
    "shoulder_pan_act",
    "shoulder_lift_act",
    "elbow_act",
    "wrist_1_act",
    "wrist_2_act",
    "wrist_3_act",
)
ARM_ACTUATOR_KP = (2200.0, 2200.0, 1800.0, 700.0, 700.0, 500.0)
ARM_ACTUATOR_KV = (120.0, 120.0, 100.0, 40.0, 40.0, 30.0)
ARM_ACTUATOR_FORCE_LIMIT = (220.0, 220.0, 180.0, 70.0, 70.0, 50.0)
UR3_ADJACENT_BODY_EXCLUDES = (
    ("shoulder_link", "upper_arm_link"),
    ("upper_arm_link", "forearm_link"),
    ("forearm_link", "wrist_1_link"),
    ("wrist_1_link", "wrist_2_link"),
    ("wrist_2_link", "wrist_3_link"),
)
UR3_BASE_BODY_EXCLUDE = ("ur3_robot/", "ur3_robot/shoulder_link")

ROBOTIQ_CTRL_MAX = 255.0
ROBOTIQ_DRIVER_CLOSED = 0.8
GRIPPER_POSITION_EPS = 0.01
DEFAULT_COMMAND_SUBSTEPS = 6
DEFAULT_GRIPPER_COMMAND_SUBSTEPS = 120


def _project_root(project_root: str | Path | None) -> Path:
    return Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]


def _default_urdf_path(project_root: Path) -> Path:
    return project_root / "lerobot_robot_sim_ur3e" / "assets" / "ur_description" / "urdf" / "ur3.urdf"


def _default_mesh_dir(project_root: Path) -> Path:
    return (
        project_root
        / "lerobot_robot_sim_ur3e"
        / "assets"
        / "ur_description"
        / "meshes"
        / "ur3"
        / "collision"
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


def _validate_asset_paths(source_urdf_path: Path, source_mesh_dir: Path, robotiq_xml_path: Path) -> None:
    missing = []
    if not source_urdf_path.is_file():
        missing.append(f"UR3 URDF: {source_urdf_path}")
    if not source_mesh_dir.is_dir():
        missing.append(f"UR3 mesh dir: {source_mesh_dir}")
    if not robotiq_xml_path.is_file():
        missing.append(f"Robotiq MJCF: {robotiq_xml_path}")
    if missing:
        raise FileNotFoundError("Missing MuJoCo simulation assets:\n" + "\n".join(missing))


def _materialize_ur3_sim_urdf(output_dir: str | Path, source_urdf_path: Path, source_mesh_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for mesh_path in source_mesh_dir.glob("*.stl"):
        shutil.copy2(mesh_path, output_dir / mesh_path.name)

    tree = ET.parse(source_urdf_path)
    for mesh in tree.getroot().findall(".//mesh"):
        mesh.attrib["filename"] = Path(mesh.attrib["filename"]).name.replace(".dae", ".stl")

    output_path = output_dir / "ur3_sim.urdf"
    tree.write(output_path)
    return output_path


def _materialize_ur3_mjcf(output_dir: str | Path, source_urdf_path: Path, source_mesh_dir: Path) -> Path:
    urdf_path = _materialize_ur3_sim_urdf(output_dir, source_urdf_path, source_mesh_dir)
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    mjcf_path = Path(output_dir) / "ur3_sim.xml"
    mujoco.mj_saveLastXML(str(mjcf_path), model)
    return mjcf_path


def _attach_hand_to_arm(arm_mjcf: mjcf.RootElement, gripper_mjcf: mjcf.RootElement) -> None:
    attachment_site = arm_mjcf.find("site", "attachment_site")
    if attachment_site is None:
        raise ValueError("attachment_site not found in UR3 MJCF")
    attachment_site.attach(gripper_mjcf)


def _add_arm_position_actuators(arm_mjcf: mjcf.RootElement) -> None:
    for actuator_name, joint_name, kp, kv, force_limit in zip(
        ARM_ACTUATOR_NAMES,
        UR3_ARM_JOINT_NAMES,
        ARM_ACTUATOR_KP,
        ARM_ACTUATOR_KV,
        ARM_ACTUATOR_FORCE_LIMIT,
        strict=True,
    ):
        arm_mjcf.actuator.add(
            "position",
            name=actuator_name,
            joint=joint_name,
            kp=str(kp),
            kv=str(kv),
            ctrlrange="-6.2831 6.2831",
            forcerange=f"-{force_limit} {force_limit}",
        )


def _add_arm_contact_excludes(arm_mjcf: mjcf.RootElement) -> None:
    for body1, body2 in UR3_ADJACENT_BODY_EXCLUDES:
        arm_mjcf.contact.add("exclude", body1=body1, body2=body2)


def _add_wrist_site_and_camera(arm_mjcf: mjcf.RootElement) -> None:
    wrist_3_link = arm_mjcf.find("body", "wrist_3_link")
    if wrist_3_link is None:
        raise ValueError("wrist_3_link not found in generated UR3 MJCF")

    wrist_3_link.add("site", name="attachment_site", pos="0 0 -0.007", quat="1 0 0 0")
    wrist_3_link.add(
        "camera",
        name="eye_in_hand",
        mode="fixed",
        pos="0 -0.085 -0.02",
        euler="2.70526 0 0",
        fovy="60",
    )


def _add_lights_floor_and_cameras(arena: mjcf.RootElement) -> None:
    arena.asset.add("material", name="floor_material", rgba="0.18 0.18 0.18 1")
    arena.worldbody.add(
        "light",
        name="key_light",
        pos="1.5 -1.0 2.5",
        dir="-0.4 0.2 -1.0",
        directional="true",
        diffuse="0.9 0.9 0.9",
        specular="0.2 0.2 0.2",
        castshadow="true",
    )
    arena.worldbody.add(
        "light",
        name="fill_light",
        pos="-1.0 1.0 1.5",
        dir="0.2 -0.3 -1.0",
        directional="true",
        diffuse="0.35 0.35 0.35",
        specular="0.05 0.05 0.05",
        castshadow="false",
    )
    arena.worldbody.add("geom", name="floor", type="plane", pos="0 0 0", size="3 3 0.05", material="floor_material")
    arena.worldbody.add(
        "camera",
        name="agentview",
        mode="fixed",
        pos="0.9 -0.85 0.8",
        xyaxes="0.884 0.468 0 -0.288 0.543 0.789",
        fovy="50",
    )
    arena.worldbody.add(
        "camera",
        name="sideview",
        mode="fixed",
        pos="0.15 -1.05 0.6",
        xyaxes="0.962 -0.275 0 0.124 0.433 0.893",
        fovy="50",
    )


def _collect_mjcf_assets(root: mjcf.RootElement) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    for asset in root.asset.all_children():
        if asset.tag == "mesh":
            mesh_file = asset.file
            assets[mesh_file.get_vfs_filename()] = mesh_file.contents
    return assets


def build_ur3_robotiq_model(
    output_dir: str | Path,
    source_urdf_path: str | Path,
    source_mesh_dir: str | Path,
    robotiq_xml_path: str | Path,
) -> mujoco.MjModel:
    mjcf_path = _materialize_ur3_mjcf(output_dir, Path(source_urdf_path), Path(source_mesh_dir))
    arm_mjcf = mjcf.from_path(str(mjcf_path))
    _add_arm_position_actuators(arm_mjcf)
    _add_arm_contact_excludes(arm_mjcf)
    _add_wrist_site_and_camera(arm_mjcf)

    gripper_mjcf = mjcf.from_path(str(robotiq_xml_path))
    _attach_hand_to_arm(arm_mjcf, gripper_mjcf)

    arena = mjcf.RootElement()
    _add_lights_floor_and_cameras(arena)
    arena.worldbody.attach(arm_mjcf)
    arena.contact.add("exclude", body1=UR3_BASE_BODY_EXCLUDE[0], body2=UR3_BASE_BODY_EXCLUDE[1])

    return mujoco.MjModel.from_xml_string(arena.to_xml_string(), _collect_mjcf_assets(arena))


def _object_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, object_id) or ""


def _name_matches(object_name: str, local_name: str) -> bool:
    return object_name == local_name or object_name.endswith(f"/{local_name}")


def _find_named_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int, name: str) -> int:
    for object_id in range(count):
        if _name_matches(_object_name(model, object_type, object_id), name):
            return object_id
    available = [_object_name(model, object_type, object_id) for object_id in range(count)]
    raise ValueError(f"{object_type.name} {name!r} not found. Available: {available}")


def _maybe_find_named_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int, name: str) -> int | None:
    for object_id in range(count):
        if _name_matches(_object_name(model, object_type, object_id), name):
            return object_id
    return None


def _find_camera_id(model: mujoco.MjModel, name: str) -> int:
    return _find_named_id(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam, name)


def _find_joint_id(model: mujoco.MjModel, name: str) -> int:
    return _find_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt, name)


def _find_body_id(model: mujoco.MjModel, name: str) -> int:
    return _find_named_id(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody, name)


def _maybe_find_site_id(model: mujoco.MjModel, name: str) -> int | None:
    return _maybe_find_named_id(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite, name)


def _find_actuator_id(model: mujoco.MjModel, name: str) -> int:
    return _find_named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu, name)


def _quat_from_mat(matrix: np.ndarray) -> np.ndarray:
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    mujoco.mju_mat2Quat(quat, matrix.reshape(-1))
    return quat


class UR3MujocoBackend:
    def __init__(
        self,
        start_joints: np.ndarray = DEFAULT_START_JOINTS,
        collision_checker: "UR3SelfCollisionChecker | None" = None,
        collision_debug: bool = False,
        camera_configs: dict[str, CameraConfig] | None = None,
        project_root: str | Path | None = None,
        source_urdf_path: str | Path | None = None,
        source_mesh_dir: str | Path | None = None,
        robotiq_xml_path: str | Path | None = None,
        show_viewer: bool = False,
        command_substeps: int = DEFAULT_COMMAND_SUBSTEPS,
        gripper_command_substeps: int = DEFAULT_GRIPPER_COMMAND_SUBSTEPS,
    ) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="ur3-mujoco-")
        project_root = _project_root(project_root)
        source_urdf_path = Path(source_urdf_path) if source_urdf_path is not None else _default_urdf_path(project_root)
        source_mesh_dir = Path(source_mesh_dir) if source_mesh_dir is not None else _default_mesh_dir(project_root)
        robotiq_xml_path = Path(robotiq_xml_path) if robotiq_xml_path is not None else _default_robotiq_path(project_root)
        _validate_asset_paths(source_urdf_path, source_mesh_dir, robotiq_xml_path)

        self._model = build_ur3_robotiq_model(self._temp_dir.name, source_urdf_path, source_mesh_dir, robotiq_xml_path)
        self._model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self._data = mujoco.MjData(self._model)
        self._state_lock = threading.Lock()
        self._collision_checker = collision_checker
        self._collision_debug = collision_debug
        self._command_substeps = int(command_substeps)
        self._gripper_command_substeps = int(gripper_command_substeps)
        if self._command_substeps < 1 or self._gripper_command_substeps < 1:
            raise ValueError("MuJoCo command substeps must be positive.")

        self._arm_dofs = UR3_ARM_DOFS
        self._interface_dofs = UR3_INTERFACE_DOFS
        self._joint_cmd = self._validate_joint_state(start_joints, "start_joints")
        self._joint_state = self._joint_cmd.copy()
        self._joint_velocities = np.zeros(self._interface_dofs, dtype=float)
        self._ee_pos_quat = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=float)

        self._arm_joint_ids = np.array([_find_joint_id(self._model, name) for name in UR3_ARM_JOINT_NAMES], dtype=int)
        self._arm_qpos_adrs = self._model.jnt_qposadr[self._arm_joint_ids]
        self._arm_dof_adrs = self._model.jnt_dofadr[self._arm_joint_ids]
        self._arm_actuator_ids = np.array([_find_actuator_id(self._model, name) for name in ARM_ACTUATOR_NAMES])
        self._fingers_actuator_id = _find_actuator_id(self._model, "fingers_actuator")
        self._right_driver_qpos_adr = self._model.jnt_qposadr[_find_joint_id(self._model, "right_driver_joint")]
        self._pinch_site_id = _maybe_find_site_id(self._model, "pinch")
        self._wrist_body_id = _find_body_id(self._model, "wrist_3_link")

        self._viewer = None
        self._cameras: list[tuple[str, MujocoCamera]] = []
        self._stopped = False

        with self._state_lock:
            self._initialize_locked()
            self._apply_joint_cmd_locked(substeps=100)

        if show_viewer:
            self._launch_viewer()
        self._configure_cameras(camera_configs or {})

    @staticmethod
    def _validate_joint_state(joint_state: np.ndarray, label: str) -> np.ndarray:
        joint_state = np.asarray(joint_state, dtype=float).copy()
        if joint_state.shape != (UR3_INTERFACE_DOFS,):
            raise ValueError(f"Expected {label} shape {(UR3_INTERFACE_DOFS,)}, got {joint_state.shape}.")
        joint_state[-1] = float(np.clip(joint_state[-1], 0.0, 1.0))
        return joint_state

    def _assert_running(self) -> None:
        if self._stopped:
            raise RuntimeError("UR3 MuJoCo backend has been stopped.")

    def _initialize_locked(self) -> None:
        self._data.qpos[self._arm_qpos_adrs] = self._joint_cmd[: self._arm_dofs]
        self._data.qvel[self._arm_dof_adrs] = 0.0
        self._data.ctrl[self._arm_actuator_ids] = self._joint_cmd[: self._arm_dofs]
        self._data.ctrl[self._fingers_actuator_id] = float(self._joint_cmd[-1] * ROBOTIQ_CTRL_MAX)
        mujoco.mj_forward(self._model, self._data)

    def _configure_cameras(self, camera_configs: dict[str, CameraConfig]) -> None:
        self._assert_running()
        for camera_key, camera_config in camera_configs.items():
            if not isinstance(camera_config, MujocoCameraConfig):
                raise TypeError(
                    f"Sim UR3e cameras must use type 'mujoco', got "
                    f"{getattr(camera_config, 'type', type(camera_config).__name__)!r} for {camera_key!r}."
                )
            camera_name = camera_config.camera if camera_config.camera is not None else camera_key
            camera = MujocoCamera(camera_config)
            camera.bind(
                self._model,
                self._data,
                _find_camera_id(self._model, str(camera_name)),
                data_lock=self._state_lock,
            )
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

    def num_dofs(self) -> int:
        self._assert_running()
        return self._interface_dofs

    def get_joint_state(self) -> np.ndarray:
        self._assert_running()
        with self._state_lock:
            return self._joint_state.copy()

    def _current_gripper_position_locked(self) -> float:
        driver_pos = float(self._data.qpos[self._right_driver_qpos_adr])
        return float(np.clip(driver_pos / ROBOTIQ_DRIVER_CLOSED, 0.0, 1.0))

    def _current_ee_pos_quat_locked(self) -> np.ndarray:
        if self._pinch_site_id is not None:
            pos = self._data.site_xpos[self._pinch_site_id].copy()
            mat = self._data.site_xmat[self._pinch_site_id].reshape(3, 3)
        else:
            pos = self._data.xpos[self._wrist_body_id].copy()
            mat = self._data.xmat[self._wrist_body_id].reshape(3, 3)
        return np.concatenate([pos, _quat_from_mat(mat)])

    def _update_observations_locked(self) -> None:
        arm_positions = self._data.qpos[self._arm_qpos_adrs].copy()
        arm_velocities = self._data.qvel[self._arm_dof_adrs].copy()
        self._joint_state = np.concatenate([arm_positions, [self._current_gripper_position_locked()]])
        self._joint_velocities[: self._arm_dofs] = arm_velocities
        self._ee_pos_quat = self._current_ee_pos_quat_locked()

    def _apply_joint_cmd_locked(self, substeps: int) -> None:
        arm_cmd = self._joint_cmd[: self._arm_dofs]
        gripper_ctrl = float(self._joint_cmd[-1] * ROBOTIQ_CTRL_MAX)
        prev_gripper = float(self._joint_state[-1])

        for _ in range(max(int(substeps), 1)):
            self._data.ctrl[self._arm_actuator_ids] = arm_cmd
            self._data.ctrl[self._fingers_actuator_id] = gripper_ctrl
            mujoco.mj_step(self._model, self._data)

        self._update_observations_locked()
        self._joint_velocities[-1] = self._joint_state[-1] - prev_gripper
        self._sync_viewer_locked()

    def _project_safe_arm_command_locked(self, joint_state: np.ndarray) -> np.ndarray:
        if self._collision_checker is None:
            return joint_state

        desired_arm = joint_state[: self._collision_checker.arm_dofs]
        current_arm = self._data.qpos[self._arm_qpos_adrs].copy()
        safe_arm = self._collision_checker.project_to_safe(current_arm, desired_arm)
        if self._collision_debug and not np.allclose(safe_arm, desired_arm):
            result = self._collision_checker.check(desired_arm)
            logger.debug(
                "Self-collision filter clipped UR3 sim command: %s",
                {
                    "current": np.round(current_arm, 4).tolist(),
                    "desired": np.round(desired_arm, 4).tolist(),
                    "safe": np.round(safe_arm, 4).tolist(),
                    "pairs": list(result.collision_pairs),
                    "minimum_distance": result.minimum_distance,
                },
            )

        joint_state = joint_state.copy()
        joint_state[: self._collision_checker.arm_dofs] = safe_arm
        return joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        self._assert_running()
        joint_state = self._validate_joint_state(joint_state, "joint_state")

        with self._state_lock:
            joint_state = self._project_safe_arm_command_locked(joint_state)
            gripper_gap = abs(float(joint_state[-1]) - float(self._joint_state[-1]))
            substeps = (
                self._gripper_command_substeps
                if gripper_gap > GRIPPER_POSITION_EPS
                else self._command_substeps
            )
            self._joint_cmd = joint_state.copy()
            self._apply_joint_cmd_locked(substeps=substeps)

    def get_observations(self) -> dict[str, np.ndarray]:
        self._assert_running()
        with self._state_lock:
            return {
                "joint_positions": self._joint_state.copy(),
                "joint_velocities": self._joint_velocities.copy(),
                "ee_pos_quat": self._ee_pos_quat.copy(),
                "gripper_position": np.array([self._joint_state[-1]], dtype=float),
            }

    def _render_cameras_locked(self) -> None:
        for _, camera in self._cameras:
            camera.render(self._data)

    def get_camera_observations(self) -> dict[str, np.ndarray]:
        self._assert_running()
        observations = {}
        for camera_name, camera in self._cameras:
            try:
                observations[camera_name] = camera.async_read(timeout_ms=5)
            except TimeoutError:
                observations[camera_name] = camera.read_cached()
        return observations

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
        self._temp_dir.cleanup()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass


UR3MujocoServer = UR3MujocoBackend
