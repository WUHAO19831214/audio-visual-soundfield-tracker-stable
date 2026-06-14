"""YOLO-first object detection with an offline OpenCV HOG fallback."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

from .config import PRIMARY_YOLO_WEIGHTS, SECONDARY_YOLO_WEIGHTS


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int = 0
    class_name: str = "person"
    track_id: int | None = None
    tracking_id_available: bool = True

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


def _scalar(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            return float(value.item())
        array = np.asarray(value).reshape(-1)
        return float(array[0]) if array.size else default
    except (TypeError, ValueError):
        return default


def _coordinates(value) -> list[int]:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value).reshape(-1)[:4].astype(int).tolist()


class Detector:
    """Prefer local Ultralytics YOLO weights and fall back without crashing."""

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        person_only: bool = True,
        tracker: str = "bytetrack.yaml",
        model_paths: Iterable[str | Path] | None = None,
        yolo_factory: Callable | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.person_only = person_only
        self.tracker = tracker
        self.backend_name = "OpenCV HOG"
        self.weight_path: Path | None = None
        self.fallback_reason = ""
        self.tracking_warning = ""
        self._model = None
        self._model_lock = threading.RLock()
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        candidates = list(model_paths or (PRIMARY_YOLO_WEIGHTS, SECONDARY_YOLO_WEIGHTS))
        selected_path = next((Path(path) for path in candidates if Path(path).exists()), None)
        if selected_path is None:
            self.fallback_reason = (
                "未找到 models/yolov8n.pt 或项目根目录下的 yolov8n.pt"
            )
            return

        try:
            if yolo_factory is None:
                from ultralytics import YOLO

                yolo_factory = YOLO
            self._model = yolo_factory(str(selected_path))
            self.backend_name = "YOLO"
            self.weight_path = selected_path.resolve()
        except Exception as exc:
            self._model = None
            self.fallback_reason = f"YOLO 加载失败：{exc}"

    @property
    def supports_native_tracking(self) -> bool:
        return self._model is not None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._model is not None:
            try:
                with self._model_lock:
                    results = self._model.predict(
                        source=frame,
                        classes=[0] if self.person_only else None,
                        conf=self.confidence_threshold,
                        verbose=False,
                    )
                return self._parse_yolo_result(results[0], use_track_ids=False)
            except Exception as exc:
                self.fallback_reason = f"YOLO 推理失败，当前帧使用 HOG：{exc}"
        return self._detect_hog(frame)

    def track(self, frame: np.ndarray) -> list[Detection]:
        """Run Ultralytics ByteTrack and include IDs when the backend supports it."""
        if self._model is None:
            return self._detect_hog(frame)
        try:
            with self._model_lock:
                results = self._model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker,
                    classes=[0] if self.person_only else None,
                    conf=self.confidence_threshold,
                    verbose=False,
                )
            detections = self._parse_yolo_result(results[0], use_track_ids=True)
            if any(not item.tracking_id_available for item in detections):
                self.tracking_warning = (
                    "ByteTrack 未返回部分 track_id，已临时使用当前帧检测序号"
                )
            else:
                self.tracking_warning = ""
            return detections
        except Exception as exc:
            self.tracking_warning = f"ByteTrack 不可用，已改用质心追踪：{exc}"
            return self.detect(frame)

    def _parse_yolo_result(self, result, use_track_ids: bool) -> list[Detection]:
        names = getattr(result, "names", None) or getattr(self._model, "names", {})
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        for index, box in enumerate(boxes):
            x1, y1, x2, y2 = _coordinates(box.xyxy[0])
            class_id = int(_scalar(getattr(box, "cls", None), 0.0))
            track_value = getattr(box, "id", None) if use_track_ids else None
            tracking_available = track_value is not None
            track_id = int(_scalar(track_value)) if tracking_available else None
            if use_track_ids and track_id is None:
                track_id = index + 1
            if isinstance(names, dict):
                class_name = str(names.get(class_id, class_id))
            else:
                class_name = str(names[class_id]) if class_id < len(names) else str(class_id)
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=_scalar(getattr(box, "conf", None)),
                    class_id=class_id,
                    class_name=class_name,
                    track_id=track_id,
                    tracking_id_available=tracking_available or not use_track_ids,
                )
            )
        return detections

    def _detect_hog(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        scale = min(1.0, 720.0 / max(width, 1))
        working = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame
        boxes, weights = self._hog.detectMultiScale(
            working, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        candidates: list[Detection] = []
        for (x, y, w, h), weight in zip(boxes, weights):
            confidence = float(weight)
            if confidence < self.confidence_threshold:
                continue
            inv = 1.0 / scale
            candidates.append(
                Detection(
                    int(x * inv),
                    int(y * inv),
                    min(width - 1, int((x + w) * inv)),
                    min(height - 1, int((y + h) * inv)),
                    confidence,
                )
            )

        candidates.sort(key=lambda item: item.confidence, reverse=True)
        kept: list[Detection] = []
        for candidate in candidates:
            if all(_iou(candidate, existing) < 0.45 for existing in kept):
                kept.append(candidate)
        return kept


def _iou(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (
        max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
        + max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
        - intersection
    )
    return intersection / union if union else 0.0


PersonDetector = Detector

