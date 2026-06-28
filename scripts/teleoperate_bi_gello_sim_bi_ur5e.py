"""Teleoperate the bimanual MuJoCo UR5e simulation with two GELLO leaders."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import rerun as rr
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_teleoperate import teleop_loop
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun

from lerobot_camera_mujoco import MujocoCameraConfig
from lerobot_robot_sim_bi_ur5e import LEFT_UR5E_START_JOINTS, RIGHT_UR5E_START_JOINTS, SimBiUR5EConfig
from lerobot_teleoperator_bi_gello import BiGelloConfig
from lerobot_teleoperator_gello import GelloConfig


GELLO_JOINT_SIGNS = [1, 1, -1, 1, 1, 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleoperate MuJoCo bimanual UR5e simulation with two GELLO leaders.")
    parser.add_argument("--left-teleop-port", default="/dev/ttyUSB0", help="Left GELLO Dynamixel serial port.")
    parser.add_argument("--right-teleop-port", default="/dev/ttyUSB1", help="Right GELLO Dynamixel serial port.")
    parser.add_argument("--teleop-id", default="bi_gello", help="LeRobot id for the bimanual GELLO teleoperator.")
    parser.add_argument("--robot-id", default="sim_bi_ur5e", help="LeRobot id for the simulated bimanual UR5e.")
    parser.add_argument("--fps", type=int, default=30, help="Control loop frequency.")
    parser.add_argument("--teleop-time-s", type=float, default=None, help="Optional teleoperation duration.")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("calibration"),
        help="Root directory for LeRobot calibration files.",
    )
    parser.add_argument("--display-data", action="store_true", help="Log observations and actions to Rerun.")
    parser.add_argument("--no-cameras", action="store_true", help="Disable MuJoCo camera observations.")
    parser.add_argument("--no-wrist-cameras", action="store_true", help="Disable the two wrist cameras.")
    parser.add_argument("--camera-width", type=int, default=640, help="MuJoCo camera image width.")
    parser.add_argument("--camera-height", type=int, default=480, help="MuJoCo camera image height.")
    parser.add_argument("--no-viewer", action="store_true", help="Do not open the MuJoCo viewer window.")
    parser.add_argument("--command-substeps", type=int, default=6, help="MuJoCo substeps for ordinary arm commands.")
    parser.add_argument(
        "--gripper-command-substeps",
        type=int,
        default=120,
        help="MuJoCo substeps when a Robotiq gripper target changes.",
    )
    return parser.parse_args()


def make_mujoco_cameras(args: argparse.Namespace) -> dict[str, MujocoCameraConfig]:
    cameras = {
        "global": MujocoCameraConfig(
            camera="global",
            width=args.camera_width,
            height=args.camera_height,
            fps=args.fps,
        )
    }
    if not args.no_wrist_cameras:
        cameras["left_wrist"] = MujocoCameraConfig(
            camera="left_wrist",
            width=args.camera_width,
            height=args.camera_height,
            fps=args.fps,
        )
        cameras["right_wrist"] = MujocoCameraConfig(
            camera="right_wrist",
            width=args.camera_width,
            height=args.camera_height,
            fps=args.fps,
        )
    return cameras


def make_gello_config(
    *,
    side: str,
    port: str,
    calibration_dir: Path,
    calibration_position: tuple[float, ...],
) -> GelloConfig:
    return GelloConfig(
        port=port,
        id=f"{side}_gello",
        calibration_dir=calibration_dir,
        calibration_position=list(calibration_position[:6]),
        joint_signs=GELLO_JOINT_SIGNS,
    )


def main() -> None:
    args = parse_args()
    init_logging()
    logging.info("Starting BiGello <-> simulated BiUR5e teleoperation")

    register_third_party_plugins()

    robot_calibration_dir = args.calibration_dir / "robots" / "sim_bi_ur5e"
    teleop_calibration_dir = args.calibration_dir / "teleoperators" / "bi_gello_sim_bi_ur5e"

    robot_cfg = SimBiUR5EConfig(
        id=args.robot_id,
        calibration_dir=robot_calibration_dir,
        cameras={} if args.no_cameras else make_mujoco_cameras(args),
        show_viewer=not args.no_viewer,
        command_substeps=args.command_substeps,
        gripper_command_substeps=args.gripper_command_substeps,
    )
    teleop_cfg = BiGelloConfig(
        id=args.teleop_id,
        calibration_dir=teleop_calibration_dir,
        left_arm_config=make_gello_config(
            side="left",
            port=args.left_teleop_port,
            calibration_dir=teleop_calibration_dir / "left",
            calibration_position=LEFT_UR5E_START_JOINTS,
        ),
        right_arm_config=make_gello_config(
            side="right",
            port=args.right_teleop_port,
            calibration_dir=teleop_calibration_dir / "right",
            calibration_position=RIGHT_UR5E_START_JOINTS,
        ),
    )

    robot = make_robot_from_config(robot_cfg)
    teleop = make_teleoperator_from_config(teleop_cfg)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    if args.display_data:
        init_rerun(session_name="bi_gello_sim_bi_ur5e")

    try:
        robot.connect()
        teleop.connect()
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=args.fps,
            display_data=args.display_data,
            duration=args.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )
    except KeyboardInterrupt:
        logging.info("Teleoperation interrupted by user")
    finally:
        if args.display_data:
            rr.rerun_shutdown()
        if teleop.is_connected:
            teleop.disconnect()
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
