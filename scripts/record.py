"""Unified ROMI recording entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.scripts import lerobot_record
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record
from lerobot.utils.import_utils import register_third_party_plugins

from common import add_common_args, make_robot_config, make_teleop_config, move_robot_home, parse_args_with_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record ROMI demonstrations.")
    add_common_args(parser)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--single-task", default=None)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--episode-time-s", type=float, default=60.0)
    parser.add_argument("--reset-time-s", type=float, default=10.0)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--display-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--streaming-encoding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--encoder-threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--play-sounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--viewer", action="store_true", help="Show MuJoCo viewer in sim mode.")
    parser.add_argument("--home-gripper-position", type=float, default=0.1)
    return parse_args_with_config(parser)


class _HomeBeforeTeleopRobot:
    def __init__(self, robot, gripper_position: float):
        self._robot = robot
        self._gripper_position = gripper_position

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def __str__(self) -> str:
        return str(self._robot)

    def connect(self, calibrate: bool = True) -> None:
        self._robot.connect(calibrate)
        move_robot_home(self._robot, self._gripper_position)


def _install_home_before_teleop_hook(gripper_position: float) -> None:
    make_robot_from_config = lerobot_record.make_robot_from_config

    def make_home_robot_from_config(config):
        return _HomeBeforeTeleopRobot(make_robot_from_config(config), gripper_position)

    lerobot_record.make_robot_from_config = make_home_robot_from_config


def main() -> None:
    args = parse_args()
    if args.repo_id is None or args.single_task is None:
        raise ValueError("Missing --repo-id or --single-task. Set them in YAML or pass them on the command line.")
    register_third_party_plugins()
    _install_home_before_teleop_hook(args.home_gripper_position)
    record(
        RecordConfig(
            robot=make_robot_config(args),
            teleop=make_teleop_config(args),
            dataset=DatasetRecordConfig(
                repo_id=args.repo_id,
                single_task=args.single_task,
                root=args.root,
                fps=args.fps,
                episode_time_s=args.episode_time_s,
                reset_time_s=args.reset_time_s,
                num_episodes=args.num_episodes,
                video=args.video,
                streaming_encoding=args.streaming_encoding,
                encoder_threads=args.encoder_threads,
                push_to_hub=args.push_to_hub,
                private=args.private,
            ),
            display_data=args.display_data,
            play_sounds=args.play_sounds,
            resume=args.resume,
        )
    )


if __name__ == "__main__":
    main()
