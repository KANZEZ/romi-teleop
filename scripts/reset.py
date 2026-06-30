"""Unified ROMI reset entrypoint."""

from __future__ import annotations

import argparse

from lerobot.robots import make_robot_from_config
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

from common import add_common_args, make_robot_config, move_robot_home, parse_args_with_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset a ROMI robot.")
    add_common_args(parser)
    parser.add_argument("--gripper-position", type=float, default=0.1)
    parser.add_argument("--viewer", action="store_true", help="Show MuJoCo viewer in sim mode.")
    return parse_args_with_config(parser)


def main() -> None:
    args = parse_args()
    init_logging()
    register_third_party_plugins()

    args.cameras = {}
    robot = make_robot_from_config(make_robot_config(args))
    try:
        robot.connect()
        move_robot_home(robot, args.gripper_position, log_label=args.robot)
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
