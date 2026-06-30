"""Build the MuJoCo scene for the bimanual UR5e simulation."""

from pathlib import Path

import mujoco
from dm_control import mjcf


# Viewer-facing assignment:
# - the arm on the viewer's right is the control "left" arm, driven by USB0
# - the arm on the viewer's left is the control "right" arm, driven by USB1
LEFT_ARM_BASE_POS = (-0.48, 0.62, 0.0)
RIGHT_ARM_BASE_POS = (0.48, 0.62, 0.0)
ARM_BASE_QUAT = (0.7071068, 0.0, 0.0, -0.7071068)
WRIST_CAMERA_DEFAULT_POS = "0.0 0.15 0.075"
WRIST_CAMERA_DEFAULT_EULER = "1.310526 0 0"


def attach_gripper_to_arm(arm_mjcf: mjcf.RootElement, gripper_mjcf: mjcf.RootElement) -> None:
    attachment_site = arm_mjcf.find("site", "attachment_site")
    if attachment_site is None:
        raise ValueError("attachment_site not found in UR5e MJCF")
    attachment_site.attach(gripper_mjcf)


def add_wrist_camera(arm_mjcf: mjcf.RootElement, camera_name: str) -> None:
    wrist_3_link = arm_mjcf.find("body", "wrist_3_link")
    if wrist_3_link is None:
        raise ValueError("wrist_3_link not found in UR5e MJCF")

    wrist_3_link.add(
        "camera",
        name=camera_name,
        mode="fixed",
        pos=WRIST_CAMERA_DEFAULT_POS,
        euler=WRIST_CAMERA_DEFAULT_EULER,
        fovy="60",
    )


def add_lights_and_floor(arena: mjcf.RootElement) -> None:
    arena.asset.add("material", name="floor_material", rgba="0.22 0.23 0.24 1")
    arena.worldbody.add(
        "light",
        name="key_light",
        pos="1.6 -1.2 2.8",
        dir="-0.4 0.25 -1",
        directional="true",
        diffuse="0.9 0.9 0.86",
        specular="0.25 0.25 0.25",
        castshadow="true",
    )
    arena.worldbody.add(
        "light",
        name="fill_light",
        pos="-1.4 1.1 1.8",
        dir="0.35 -0.25 -1",
        directional="true",
        diffuse="0.35 0.38 0.42",
        specular="0.05 0.05 0.05",
        castshadow="false",
    )
    arena.worldbody.add(
        "geom",
        name="floor",
        type="plane",
        pos="0 0 0",
        size="3 3 0.05",
        material="floor_material",
    )


def add_global_camera(arena: mjcf.RootElement) -> None:
    arena.worldbody.add(
        "camera",
        name="global",
        mode="fixed",
        pos="1.35 -1.45 1.45",
        xyaxes="0.732 0.681 0 -0.429 0.461 0.777",
        fovy="45",
    )


def build_single_arm(
    arm_xml_path: str | Path,
    gripper_xml_path: str | Path,
    side: str,
    base_pos: tuple[float, float, float],
    base_quat: tuple[float, float, float, float],
) -> mjcf.RootElement:
    arm_mjcf = mjcf.from_path(str(arm_xml_path))
    arm_mjcf.model = f"{side}_ur5e"
    gripper_mjcf = mjcf.from_path(str(gripper_xml_path))
    gripper_mjcf.model = f"{side}_robotiq"

    base_body = arm_mjcf.find("body", "base")
    if base_body is None:
        raise ValueError("base body not found in UR5e MJCF")
    base_body.pos = base_pos
    base_body.quat = base_quat

    add_wrist_camera(arm_mjcf, f"{side}_wrist")
    attach_gripper_to_arm(arm_mjcf, gripper_mjcf)
    return arm_mjcf


def build_bi_ur5e_mujoco_env(
    ur5e_xml_path: str | Path,
    robotiq_xml_path: str | Path,
) -> mujoco.MjModel:
    arena = mjcf.RootElement(model="bi_ur5e")
    arena.option.timestep = 0.002
    arena.option.integrator = "implicitfast"

    add_lights_and_floor(arena)
    add_global_camera(arena)

    left_arm = build_single_arm(
        arm_xml_path=ur5e_xml_path,
        gripper_xml_path=robotiq_xml_path,
        side="left",
        base_pos=LEFT_ARM_BASE_POS,
        base_quat=ARM_BASE_QUAT,
    )
    right_arm = build_single_arm(
        arm_xml_path=ur5e_xml_path,
        gripper_xml_path=robotiq_xml_path,
        side="right",
        base_pos=RIGHT_ARM_BASE_POS,
        base_quat=ARM_BASE_QUAT,
    )
    arena.worldbody.attach(left_arm)
    arena.worldbody.attach(right_arm)

    assets: dict[str, bytes] = {}
    for asset in arena.asset.all_children():
        if asset.tag == "mesh":
            mesh_file = asset.file
            assets[mesh_file.get_vfs_filename()] = mesh_file.contents

    return mujoco.MjModel.from_xml_string(arena.to_xml_string(), assets)
