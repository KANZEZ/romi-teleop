"""Unified ROMI teleoperation entrypoint."""

from __future__ import annotations

import argparse
import time

from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, shutdown_rerun

from common import add_common_args, make_robot_config, make_teleop_config, move_robot_home, parse_args_with_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleoperate a ROMI robot.")
    add_common_args(parser)
    parser.add_argument("--time-s", type=float, default=None)
    parser.add_argument("--display-data", action="store_true")
    parser.add_argument("--viewer", action="store_true", help="Show MuJoCo viewer in sim mode.")
    parser.add_argument("--home-gripper-position", type=float, default=0.1)
    return parse_args_with_config(parser)


def main() -> None:
    args = parse_args()
    register_third_party_plugins()
    init_logging()

    if args.mode == "sim" and not args.display_data:
        args.cameras = {}

    robot = make_robot_from_config(make_robot_config(args))
    teleop = make_teleoperator_from_config(make_teleop_config(args))
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    robot_connected = False
    teleop_connected = False
    try:
        if args.mode == "sim":
            print("Loading MuJoCo simulation and viewer...")
        robot.connect()
        robot_connected = True
        move_robot_home(robot, args.home_gripper_position)
        if args.mode == "sim":
            print("MuJoCo viewer is ready. Robot is at home. Connecting teleoperator; calibration prompts may appear next.")
        teleop.connect()
        teleop_connected = True

        if args.display_data:
            init_rerun(session_name="teleoperation")

        _teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=args.fps,
            display_data=args.display_data,
            duration=args.time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=False,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if args.display_data:
            shutdown_rerun()
        if teleop_connected:
            teleop.disconnect()
        if robot_connected:
            robot.disconnect()


def _teleop_loop(
    teleop,
    robot,
    fps: int,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
) -> None:
    from lerobot.utils.visualization_utils import log_rerun_data

    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        obs = robot.get_observation() if display_data else {}

        raw_action = teleop.get_action()
        teleop_action = teleop_action_processor((raw_action, obs))
        robot_action_to_send = robot_action_processor((teleop_action, obs))
        robot.send_action(robot_action_to_send)

        if display_data:
            log_rerun_data(
                observation=robot_observation_processor(obs),
                action=teleop_action,
                compress_images=display_compressed_images,
            )

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)", end="\r")

        if duration is not None and time.perf_counter() - start >= duration:
            print()
            return


if __name__ == "__main__":
    main()
