"""Small factories for ROMI teleoperation scripts."""

from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path

import yaml
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot_camera_mujoco import MujocoCameraConfig
from lerobot_robot_bi_ur3 import BiUR3Config, LEFT_UR3_IP, LEFT_UR3_START_JOINTS, RIGHT_UR3_IP, RIGHT_UR3_START_JOINTS
from lerobot_robot_bi_ur5e import (
    BiUR5EConfig,
    LEFT_UR5E_GELLO_CALIBRATION_POSITION,
    LEFT_UR5E_IP,
    LEFT_UR5E_START_JOINTS,
    RIGHT_UR5E_GELLO_CALIBRATION_POSITION,
    RIGHT_UR5E_IP,
    RIGHT_UR5E_START_JOINTS,
)
from lerobot_robot_sim_bi_ur5e import (
    LEFT_UR5E_START_JOINTS as SIM_LEFT_UR5E_START_JOINTS,
    RIGHT_UR5E_START_JOINTS as SIM_RIGHT_UR5E_START_JOINTS,
    SimBiUR5EConfig,
)
from lerobot_robot_sim_ur3e import SimUR3EConfig
from lerobot_robot_ur3 import UR3Config
from lerobot_robot_ur5e import UR5EConfig
from lerobot_teleoperator_bi_gello import BiGelloConfig
from lerobot_teleoperator_gello import GelloConfig


UR3_LEFT_GELLO_HOME = [-1.5708, -1.5708, -1.5708, -1.5708, 1.5708, 1.5708]
UR3_RIGHT_GELLO_HOME = [1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 1.5708]
GELLO_SIGNS = [1, 1, -1, 1, 1, 1]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--robot", default=None, choices=("ur3", "ur3e", "ur5e", "bi_ur3", "bi_ur5e"))
    parser.add_argument("--mode", default="real", choices=("real", "sim"))
    parser.add_argument("--calibration-dir", type=Path, default=Path("calibration"))
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--cameras",
        default="{}",
        help="Camera config dict, e.g. '{\"cam\": {\"type\": \"realsense\", \"serial_number_or_name\": \"...\", \"width\": 640, \"height\": 480, \"fps\": 30}}'.",
    )
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--arm", choices=("left", "right"), default=None, help="Single-arm side for UR3/UR5e real setups.")
    parser.add_argument("--robot-ip")
    parser.add_argument("--left-robot-ip")
    parser.add_argument("--right-robot-ip")
    parser.add_argument("--teleop-port", default="/dev/ttyUSB0")
    parser.add_argument("--left-teleop-port", default="/dev/ttyUSB0")
    parser.add_argument("--right-teleop-port", default="/dev/ttyUSB1")
    parser.add_argument("--max-joint-delta", type=float, default=0.1)
    parser.add_argument("--command-substeps", type=int, default=6)
    parser.add_argument("--gripper-command-substeps", type=int, default=6)


def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, remaining = config_parser.parse_known_args()
    parser.add_argument("--config", type=Path, default=config_args.config)
    if config_args.config is not None:
        parser.set_defaults(**_load_yaml_defaults(config_args.config))
    return parser.parse_args(remaining)


def move_robot_home(robot, gripper_position: float, log_label: str | None = None) -> None:
    robot_label = log_label if log_label is not None else str(robot)
    if hasattr(robot, "move_to_start_joints"):
        logging.info("Moving %s to configured home/start joints.", robot_label)
        robot.move_to_start_joints(wait=True)
    if hasattr(robot, "reset_gripper"):
        logging.info("Resetting %s gripper.", robot_label)
        robot.reset_gripper(gripper_position)


def make_robot_config(args: argparse.Namespace):
    if args.robot is None:
        raise ValueError("Missing --robot. Set it in YAML or pass it on the command line.")
    if args.mode == "sim":
        return _make_sim_robot_config(args)
    return _make_real_robot_config(args)


def make_teleop_config(args: argparse.Namespace):
    if args.robot.startswith("bi_"):
        root = Path(args.calibration_dir) / "teleoperators" / f"bi_gello_{args.robot}_{args.mode}"
        left_home, right_home = _bi_gello_homes(args)
        return BiGelloConfig(
            id="bi_gello",
            calibration_dir=root,
            left_arm_config=_gello("left", args.left_teleop_port, root / "left", left_home),
            right_arm_config=_gello("right", args.right_teleop_port, root / "right", right_home),
    )
    arm = args.arm or "right"
    home = _single_gello_home(args, arm)
    port = args.left_teleop_port if arm == "left" else args.right_teleop_port
    return GelloConfig(
        id=f"{arm}_gello",
        port=port,
        calibration_dir=Path(args.calibration_dir) / "teleoperators" / f"gello_{args.robot}_{args.mode}" / arm,
        calibration_position=list(home[:6]),
        joint_signs=GELLO_SIGNS,
    )


def _make_real_robot_config(args: argparse.Namespace):
    root = Path(args.calibration_dir) / "robots"
    max_delta = args.max_joint_delta if args.max_joint_delta > 0 else None
    if args.robot == "ur3":
        arm = args.arm or "right"
        default_ip, start_joints = _ur3_single_arm_defaults(arm)
        side_ip = args.left_robot_ip if arm == "left" else args.right_robot_ip
        return UR3Config(
            id=f"{arm}_ur3",
            ip=side_ip or default_ip,
            calibration_dir=root / "ur3" / arm,
            cameras=_cameras(args),
            start_joints=start_joints,
            with_gripper=not args.no_gripper,
            gripper_auto_calibrate=False,
            max_joint_delta_per_step=max_delta,
        )
    if args.robot == "ur5e":
        arm = args.arm or "right"
        default_ip, start_joints = _ur5e_single_arm_defaults(arm)
        side_ip = args.left_robot_ip if arm == "left" else args.right_robot_ip
        return UR5EConfig(
            id=f"{arm}_ur5e",
            ip=side_ip or default_ip,
            calibration_dir=root / "ur5e" / arm,
            cameras=_cameras(args),
            start_joints=start_joints,
            with_gripper=not args.no_gripper,
            max_joint_delta_per_step=max_delta,
        )
    if args.robot == "bi_ur3":
        left_ip = args.left_robot_ip or LEFT_UR3_IP
        right_ip = args.right_robot_ip or RIGHT_UR3_IP
        return BiUR3Config(
            id="bi_ur3",
            calibration_dir=root / "bi_ur3",
            left_arm_config=_ur3("left", left_ip, root / "bi_ur3" / "left", LEFT_UR3_START_JOINTS, args, max_delta),
            right_arm_config=_ur3("right", right_ip, root / "bi_ur3" / "right", RIGHT_UR3_START_JOINTS, args, max_delta),
        )
    if args.robot == "bi_ur5e":
        left_ip = args.left_robot_ip or LEFT_UR5E_IP
        right_ip = args.right_robot_ip or RIGHT_UR5E_IP
        return BiUR5EConfig(
            id="bi_ur5e",
            calibration_dir=root / "bi_ur5e",
            left_arm_config=_ur5e("left", left_ip, root / "bi_ur5e" / "left", LEFT_UR5E_START_JOINTS, args, max_delta),
            right_arm_config=_ur5e("right", right_ip, root / "bi_ur5e" / "right", RIGHT_UR5E_START_JOINTS, args, max_delta),
        )
    raise ValueError(f"Unsupported real robot: {args.robot}")


def _ur3_single_arm_defaults(arm: str) -> tuple[str, tuple[float, ...]]:
    if arm == "left":
        return LEFT_UR3_IP, LEFT_UR3_START_JOINTS
    if arm == "right":
        return RIGHT_UR3_IP, RIGHT_UR3_START_JOINTS
    raise ValueError(f"Unsupported UR3 arm side: {arm!r}.")


def _ur5e_single_arm_defaults(arm: str) -> tuple[str, tuple[float, ...]]:
    if arm == "left":
        return LEFT_UR5E_IP, LEFT_UR5E_START_JOINTS
    if arm == "right":
        return RIGHT_UR5E_IP, RIGHT_UR5E_START_JOINTS
    raise ValueError(f"Unsupported UR5e arm side: {arm!r}.")


def _make_sim_robot_config(args: argparse.Namespace):
    root = Path(args.calibration_dir) / "robots"
    if args.robot == "ur3e":
        return SimUR3EConfig(
            id="sim_ur3e",
            calibration_dir=root / "sim_ur3e",
            cameras=_cameras(args),
            show_viewer=getattr(args, "viewer", False),
            command_substeps=args.command_substeps,
            gripper_command_substeps=args.gripper_command_substeps,
        )
    if args.robot == "bi_ur5e":
        return SimBiUR5EConfig(
            id="sim_bi_ur5e",
            calibration_dir=root / "sim_bi_ur5e",
            cameras=_cameras(args),
            show_viewer=getattr(args, "viewer", False),
            command_substeps=args.command_substeps,
            gripper_command_substeps=args.gripper_command_substeps,
        )
    raise ValueError("Simulation mode supports --robot ur3e or --robot bi_ur5e.")


def _ur3(side: str, ip: str, calibration_dir: Path, start_joints: tuple[float, ...], args, max_delta):
    return UR3Config(
        id=f"{side}_ur3",
        ip=ip,
        calibration_dir=calibration_dir,
        cameras={} if side == "right" else _cameras(args),
        start_joints=start_joints,
        with_gripper=not args.no_gripper,
        gripper_auto_calibrate=False,
        max_joint_delta_per_step=max_delta,
    )


def _ur5e(side: str, ip: str, calibration_dir: Path, start_joints: tuple[float, ...], args, max_delta):
    return UR5EConfig(
        id=f"{side}_ur5e",
        ip=ip,
        calibration_dir=calibration_dir,
        cameras={} if side == "right" else _cameras(args),
        start_joints=start_joints,
        with_gripper=not args.no_gripper,
        max_joint_delta_per_step=max_delta,
    )


def _gello(side: str, port: str, calibration_dir: Path, home: list[float] | tuple[float, ...]):
    return GelloConfig(
        id=f"{side}_gello",
        port=port,
        calibration_dir=calibration_dir,
        calibration_position=list(home[:6]),
        joint_signs=GELLO_SIGNS,
    )


def _bi_gello_homes(args: argparse.Namespace):
    if args.robot == "bi_ur3":
        return UR3_LEFT_GELLO_HOME, UR3_RIGHT_GELLO_HOME
    if args.mode == "sim":
        return SIM_LEFT_UR5E_START_JOINTS, SIM_RIGHT_UR5E_START_JOINTS
    return LEFT_UR5E_GELLO_CALIBRATION_POSITION, RIGHT_UR5E_GELLO_CALIBRATION_POSITION


def _single_gello_home(args: argparse.Namespace, arm: str) -> list[float] | tuple[float, ...]:
    if args.robot == "ur3":
        if arm == "left":
            return UR3_LEFT_GELLO_HOME
        return UR3_RIGHT_GELLO_HOME
    if args.robot == "ur5e":
        if arm == "left":
            return LEFT_UR5E_GELLO_CALIBRATION_POSITION
        return RIGHT_UR5E_GELLO_CALIBRATION_POSITION
    return GelloConfig().calibration_position


def _cameras(args: argparse.Namespace):
    camera_specs = _camera_specs(args.cameras)
    if not camera_specs:
        return {}
    return {name: _camera_config(name, spec, args.mode) for name, spec in camera_specs.items()}


def _camera_specs(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if value.strip().lower() in ("", "none", "null"):
        return {}
    try:
        specs = json.loads(value)
    except json.JSONDecodeError:
        specs = ast.literal_eval(value)
    if not isinstance(specs, dict):
        raise ValueError("--cameras must be a dict.")
    return specs


def _camera_config(name: str, spec: dict, mode: str):
    if not isinstance(spec, dict):
        raise ValueError(f"Camera {name!r} config must be a dict.")
    spec = dict(spec)
    camera_type = spec.pop("type", None)
    if camera_type is None:
        raise ValueError(f"Camera {name!r} config must include a 'type' field.")
    if camera_type == "realsense":
        if mode != "real":
            raise ValueError("RealSense cameras require --mode real.")
        if "serial" in spec and "serial_number_or_name" not in spec:
            spec["serial_number_or_name"] = spec.pop("serial")
        return RealSenseCameraConfig(**spec)
    if camera_type == "mujoco":
        if mode != "sim":
            raise ValueError("MuJoCo cameras require --mode sim.")
        spec.setdefault("camera", name)
        return MujocoCameraConfig(**spec)
    raise ValueError(f"Unsupported camera type {camera_type!r}.")


def _load_yaml_defaults(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return {_arg_name(key): _coerce_config_value(key, value) for key, value in data.items()}


def _arg_name(key: str) -> str:
    return str(key).replace("-", "_")


def _coerce_config_value(key: str, value):
    if _arg_name(key) in {"calibration_dir", "root"} and value is not None:
        return Path(value)
    return value
