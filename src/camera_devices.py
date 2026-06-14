"""Local OpenCV camera enumeration and opening helpers."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable

import cv2


def camera_backends() -> list[int | None]:
    if platform.system() == "Darwin":
        return [cv2.CAP_AVFOUNDATION, None]
    return [None]


def open_camera(
    camera_index: int,
    width: int = 1280,
    height: int = 720,
    capture_factory: Callable = cv2.VideoCapture,
):
    """Open a camera, preferring AVFoundation on macOS."""
    errors: list[str] = []
    for backend in camera_backends():
        try:
            capture = (
                capture_factory(camera_index, backend)
                if backend is not None
                else capture_factory(camera_index)
            )
        except Exception as exc:
            errors.append(str(exc))
            continue
        if capture is not None and capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return capture
        if capture is not None:
            capture.release()
    detail = "; ".join(errors) if errors else "设备未打开"
    raise RuntimeError(f"无法打开摄像头 index={camera_index}：{detail}")


def read_camera_preview(camera_index: int):
    capture = open_camera(camera_index)
    try:
        for _ in range(5):
            ok, frame = capture.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.1)
        raise RuntimeError(f"摄像头 index={camera_index} 已打开，但读取不到画面")
    finally:
        capture.release()


def list_available_cameras(
    max_index: int = 5,
    capture_factory: Callable = cv2.VideoCapture,
) -> list[dict]:
    """Probe camera indexes independently so one failure never aborts the scan."""
    devices: list[dict] = []
    for index in range(max_index + 1):
        capture = None
        try:
            capture = open_camera(index, capture_factory=capture_factory)
            for _ in range(3):
                ok, frame = capture.read()
                if ok and frame is not None:
                    devices.append({"index": index, "name": f"Camera {index}"})
                    break
                time.sleep(0.05)
        except Exception:
            continue
        finally:
            if capture is not None:
                capture.release()
    return devices
