"""LeRobot camera implementation backed by a MuJoCo renderer."""

from __future__ import annotations

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import mujoco
import numpy as np
from lerobot.cameras.camera import Camera
from lerobot.utils.errors import DeviceNotConnectedError

from .configuration_mujoco import MujocoCameraConfig

logger = logging.getLogger(__name__)


class MujocoCamera(Camera):
    def __init__(self, config: MujocoCameraConfig):
        super().__init__(config)
        self.config = config
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._camera: int | str | None = config.camera
        self._renderer: mujoco.Renderer | None = None
        self._frame_lock = Lock()
        self._render_lock = Lock()
        self._data_lock: Lock | None = None
        self._latest_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self._latest_depth = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self._latest_timestamp = 0.0
        self._latest_frame_id = 0
        self._last_consumed_frame_id = 0
        self._new_frame_event = Event()
        self._stop_event: Event | None = None
        self._thread: Thread | None = None
        self._ready_event = Event()
        self._thread_error: BaseException | None = None
        self._connected = False

    def __repr__(self) -> str:
        return f"MujocoCamera(camera={self._camera}, width={self.width}, height={self.height})"

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return []

    def bind(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera: int | str,
        data_lock: Lock | None = None,
    ) -> None:
        self._model = model
        self._data = data
        self._camera = camera
        self._data_lock = data_lock

    def connect(self, warmup: bool = True) -> None:
        if self._model is None or self._data is None or self._camera is None:
            raise DeviceNotConnectedError(f"{self} is not bound to a MuJoCo model/data.")
        self._connected = True
        self._start_read_thread()
        if self._thread_error is not None:
            self.disconnect()
            raise RuntimeError(f"Failed to start render thread for {self}.") from self._thread_error
        if warmup:
            self.async_read(timeout_ms=10000)

    def _ensure_renderer(self) -> mujoco.Renderer:
        if self._model is None:
            raise DeviceNotConnectedError(f"{self} is not bound to a MuJoCo model.")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=self.height, width=self.width)
        return self._renderer

    def render(self, data: mujoco.MjData | None = None) -> None:
        if data is not None:
            self._data = data
        if self._data is None:
            raise DeviceNotConnectedError(f"{self} is not bound to MuJoCo data.")

        self._render_current_data()

    def _copy_source_state(self, render_data: mujoco.MjData) -> None:
        if self._data is None:
            raise DeviceNotConnectedError(f"{self} is not bound to MuJoCo data.")

        def copy_state() -> None:
            render_data.qpos[:] = self._data.qpos
            render_data.qvel[:] = self._data.qvel
            render_data.act[:] = self._data.act
            render_data.mocap_pos[:] = self._data.mocap_pos
            render_data.mocap_quat[:] = self._data.mocap_quat

        if self._data_lock is None:
            copy_state()
        else:
            with self._data_lock:
                copy_state()

    def _render_from_snapshot(self, renderer: mujoco.Renderer, render_data: mujoco.MjData) -> None:
        if self._model is None:
            raise DeviceNotConnectedError(f"{self} is not bound to a MuJoCo model.")

        self._copy_source_state(render_data)
        mujoco.mj_forward(self._model, render_data)

        renderer.disable_depth_rendering()
        renderer.update_scene(render_data, camera=self._camera)
        image = renderer.render().copy()

        if self.config.use_depth:
            renderer.enable_depth_rendering()
            renderer.update_scene(render_data, camera=self._camera)
            depth = renderer.render().copy()[:, :, None].astype(np.float32)
            renderer.disable_depth_rendering()
        else:
            with self._frame_lock:
                depth = self._latest_depth

        self._store_frame(image, depth)

    def _render_current_data(self) -> None:
        with self._render_lock:
            if self._data is None:
                raise DeviceNotConnectedError(f"{self} is not bound to MuJoCo data.")

            renderer = self._ensure_renderer()
            renderer.disable_depth_rendering()
            renderer.update_scene(self._data, camera=self._camera)
            image = renderer.render().copy()

            if self.config.use_depth:
                renderer.enable_depth_rendering()
                renderer.update_scene(self._data, camera=self._camera)
                depth = renderer.render().copy()[:, :, None].astype(np.float32)
                renderer.disable_depth_rendering()
            else:
                depth = self._latest_depth

        self._store_frame(image, depth)

    def _store_frame(self, image: np.ndarray, depth: np.ndarray) -> None:
        with self._frame_lock:
            self._latest_image = image
            self._latest_depth = depth
            self._latest_timestamp = time.perf_counter()
            self._latest_frame_id += 1
            self._new_frame_event.set()

    def _start_read_thread(self) -> None:
        if self._thread is not None:
            return
        self._stop_event = Event()
        self._ready_event.clear()
        self._thread_error = None
        self._thread = Thread(target=self._read_loop, name=f"{self._camera}-mujoco-camera", daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=10.0):
            raise TimeoutError(f"Timed out starting render thread for {self}.")

    def _read_loop(self) -> None:
        assert self._stop_event is not None
        if self._model is None:
            self._ready_event.set()
            raise DeviceNotConnectedError(f"{self} is not bound to a MuJoCo model.")

        period_s = 1.0 / max(float(self.fps or 60), 1.0)
        render_data = mujoco.MjData(self._model)
        try:
            with self._render_lock:
                self._renderer = mujoco.Renderer(self._model, height=self.height, width=self.width)
                renderer = self._renderer
            self._ready_event.set()

            while not self._stop_event.is_set():
                start = time.perf_counter()
                try:
                    self._render_from_snapshot(renderer, render_data)
                except Exception as exc:
                    if not self._stop_event.is_set():
                        logger.exception("Failed to render %s.", self)
                    self._thread_error = exc
                elapsed_s = time.perf_counter() - start
                self._stop_event.wait(max(period_s - elapsed_s, 0.0))
        except BaseException as exc:
            self._thread_error = exc
            self._ready_event.set()
            if not self._stop_event.is_set():
                logger.exception("MuJoCo render thread stopped for %s.", self)
        finally:
            self._close_renderer()

    def _close_renderer(self) -> None:
        with self._render_lock:
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None

    def read(self) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        with self._frame_lock:
            return self._latest_image.copy()

    def read_depth(self) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        with self._frame_lock:
            return self._latest_depth.copy()

    def read_resized(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        image = self.read()
        depth = self.read_depth()
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth[:, :, 0], (width, height), interpolation=cv2.INTER_LINEAR)[:, :, None]
        return image, depth

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self._thread_error is not None and (self._thread is None or not self._thread.is_alive()):
            raise RuntimeError(f"{self} render thread failed.") from self._thread_error

        deadline = time.perf_counter() + timeout_ms / 1000.0
        while True:
            with self._frame_lock:
                if self._latest_frame_id != self._last_consumed_frame_id:
                    self._last_consumed_frame_id = self._latest_frame_id
                    self._new_frame_event.clear()
                    return self._latest_image.copy()

            remaining_s = deadline - time.perf_counter()
            if remaining_s <= 0 or not self._new_frame_event.wait(remaining_s):
                raise TimeoutError(f"Timed out waiting for a new frame from {self}.")

    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        with self._frame_lock:
            if self._latest_frame_id == 0:
                raise RuntimeError(f"{self} has not captured any frames yet.")
            age_ms = (time.perf_counter() - self._latest_timestamp) * 1000
            if age_ms > max_age_ms:
                raise TimeoutError(f"Latest frame from {self} is stale ({age_ms:.1f} ms old).")
            return self._latest_image.copy()

    def read_cached(self) -> np.ndarray:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        with self._frame_lock:
            return self._latest_image.copy()

    def disconnect(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None
        self._stop_event = None
        self._close_renderer()
        self._connected = False
