"""Unified ROMI teleoperation entrypoint."""

from __future__ import annotations

import argparse

from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate
from lerobot.utils.import_utils import register_third_party_plugins

from common import add_common_args, make_robot_config, make_teleop_config, parse_args_with_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleoperate a ROMI robot.")
    add_common_args(parser)
    parser.add_argument("--time-s", type=float, default=None)
    parser.add_argument("--display-data", action="store_true")
    parser.add_argument("--viewer", action="store_true", help="Show MuJoCo viewer in sim mode.")
    return parse_args_with_config(parser)


def main() -> None:
    args = parse_args()
    register_third_party_plugins()
    teleoperate(
        TeleoperateConfig(
            robot=make_robot_config(args),
            teleop=make_teleop_config(args),
            fps=args.fps,
            teleop_time_s=args.time_s,
            display_data=args.display_data,
        )
    )


if __name__ == "__main__":
    main()
