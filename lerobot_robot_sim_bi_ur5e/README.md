# lerobot_robot_sim_bi_ur5e

LeRobot robot plugin for a bimanual UR5e MuJoCo scene.

The package mirrors `lerobot_robot_sim_ur3e`, but the MuJoCo scene is built in
`lerobot_robot_sim_bi_ur5e/build_bi_ur5e_mujoco_env.py`.

## Scene

- Two fixed UR5e arms with Robotiq 2F-85 grippers.
- One ground plane.
- Three cameras:
  - `global`
  - `left_wrist`
  - `right_wrist`

The default robot and gripper MJCF files are bundled under
`lerobot_robot_sim_bi_ur5e/assest/mujoco_menagerie`:

- `universal_robots_ur5e/ur5e.xml`
- `robotiq_2f85_v4/2f85.xml`

The `assest/` directory is intentionally present for bundled or custom assets.
The misspelling follows the requested directory name.

## Usage

Teleoperate the empty MuJoCo scene with two GELLO leaders:

```bash
python scripts/teleoperate_bi_gello_sim_bi_ur5e.py \
  --left-teleop-port /dev/ttyUSB0 \
  --right-teleop-port /dev/ttyUSB1
```

```python
from lerobot_robot_sim_bi_ur5e import SimBiUR5E, SimBiUR5EConfig

robot = SimBiUR5E(SimBiUR5EConfig(show_viewer=True))
robot.connect()
obs = robot.get_observation()
action = {key: obs[key] for key in robot.action_features}
robot.send_action(action)
robot.disconnect()
```

Camera configs default to MuJoCo cameras. Override `SimBiUR5EConfig.cameras` if
you need different resolutions or only a subset of the three cameras.
