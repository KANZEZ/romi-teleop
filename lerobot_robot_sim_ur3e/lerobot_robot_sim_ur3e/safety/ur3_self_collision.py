from pathlib import Path
from typing import Iterable

import numpy as np
import pinocchio as pin

from .collision_checker import CollisionCheckResult
from .ur3_self_collision_config import (
    DEFAULT_COLLISION_MARGIN,
    DEFAULT_IGNORED_COLLISION_PAIRS,
    DEFAULT_LINE_SEARCH_STEPS,
    DEFAULT_MAX_JOINT_STEP,
    DEFAULT_UR3_PACKAGE_DIR,
    DEFAULT_UR3_URDF_PATH,
    UR3_ARM_DOFS,
)


class UR3SelfCollisionChecker:
    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_UR3_URDF_PATH,
        package_dir: str | Path = DEFAULT_UR3_PACKAGE_DIR,
        ignored_collision_pairs: Iterable[tuple[str, str]] = DEFAULT_IGNORED_COLLISION_PAIRS,
        collision_margin: float = DEFAULT_COLLISION_MARGIN,
        line_search_steps: int = DEFAULT_LINE_SEARCH_STEPS,
        max_joint_step: float = DEFAULT_MAX_JOINT_STEP,
    ) -> None:
        self.arm_dofs = UR3_ARM_DOFS
        self.urdf_path = Path(urdf_path)
        self.package_dir = Path(package_dir)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"UR3 URDF not found: {self.urdf_path}")
        if not self.package_dir.exists():
            raise FileNotFoundError(f"UR3 package dir not found: {self.package_dir}")

        self.collision_margin = float(collision_margin)
        self.line_search_steps = int(line_search_steps)
        self.max_joint_step = float(max_joint_step)
        self._ignored_collision_pairs = {
            frozenset(pair) for pair in ignored_collision_pairs
        }

        self._model, self._collision_model = pin.buildModelsFromUrdf(
            str(self.urdf_path),
            package_dirs=[str(self.package_dir)],
            geometry_types=[pin.GeometryType.COLLISION],
        )
        self._collision_model.addAllCollisionPairs()
        self._data = self._model.createData()
        self._collision_data = pin.GeometryData(self._collision_model)
        self._active_pair_indices = self._configure_collision_pairs()

    def _configure_collision_pairs(self) -> tuple[int, ...]:
        active_pair_indices: list[int] = []
        for pair_index, pair in enumerate(self._collision_model.collisionPairs):
            names = self._pair_names(pair_index)
            if frozenset(names) in self._ignored_collision_pairs:
                self._collision_data.deactivateCollisionPair(pair_index)
                continue
            active_pair_indices.append(pair_index)
        return tuple(active_pair_indices)

    def _pair_names(self, pair_index: int) -> tuple[str, str]:
        pair = self._collision_model.collisionPairs[pair_index]
        first = self._collision_model.geometryObjects[pair.first].name
        second = self._collision_model.geometryObjects[pair.second].name
        return first, second

    def _validate_joints(self, joints: np.ndarray) -> np.ndarray:
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (self.arm_dofs,):
            raise ValueError(
                f"Expected {self.arm_dofs} arm joints, got shape {joints.shape}"
            )
        return joints

    def check(self, joints: np.ndarray) -> CollisionCheckResult:
        joints = self._validate_joints(joints)
        pin.computeDistances(
            self._model,
            self._data,
            self._collision_model,
            self._collision_data,
            joints,
        )

        minimum_distance: float | None = None
        collision_pairs: list[tuple[str, str]] = []
        for pair_index in self._active_pair_indices:
            distance = float(self._collision_data.distanceResults[pair_index].min_distance)
            if minimum_distance is None or distance < minimum_distance:
                minimum_distance = distance
            if distance <= self.collision_margin:
                collision_pairs.append(self._pair_names(pair_index))

        return CollisionCheckResult(
            in_collision=bool(collision_pairs),
            collision_pairs=tuple(collision_pairs),
            minimum_distance=minimum_distance,
        )

    def is_state_safe(self, joints: np.ndarray) -> bool:
        return not self.check(joints).in_collision

    def _clamp_delta(self, current_joints: np.ndarray, desired_joints: np.ndarray) -> np.ndarray:
        delta = desired_joints - current_joints
        max_delta = float(np.abs(delta).max())
        if self.max_joint_step <= 0.0 or max_delta <= self.max_joint_step:
            return desired_joints
        return current_joints + delta * (self.max_joint_step / max_delta)

    def project_to_safe(
        self, current_joints: np.ndarray, desired_joints: np.ndarray
    ) -> np.ndarray:
        current_joints = self._validate_joints(current_joints)
        desired_joints = self._validate_joints(desired_joints)
        desired_joints = self._clamp_delta(current_joints, desired_joints)

        if self.is_state_safe(desired_joints):
            return desired_joints
        if not self.is_state_safe(current_joints):
            return current_joints.copy()

        low = 0.0
        high = 1.0
        safe_joints = current_joints.copy()
        delta = desired_joints - current_joints

        for _ in range(max(self.line_search_steps, 0)):
            alpha = 0.5 * (low + high)
            candidate = current_joints + alpha * delta
            if self.is_state_safe(candidate):
                low = alpha
                safe_joints = candidate
            else:
                high = alpha

        return safe_joints
