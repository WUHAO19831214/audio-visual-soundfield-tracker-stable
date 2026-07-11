"""Experimental single-object tracking built on OpenCV trackers."""

from __future__ import annotations

from collections.abc import Sequence

import cv2


TRACKER_FALLBACK_ORDER = ("CSRT", "KCF", "MIL")


def validate_bbox(frame, bbox: Sequence[float]) -> tuple[float, float, float, float]:
    """Validate and normalize an (x, y, width, height) box for a frame."""
    if frame is None or getattr(frame, "ndim", 0) < 2:
        raise ValueError("目标初始化画面无效")
    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox 必须是 (x, y, width, height)")
    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox 必须全部为数字") from exc

    frame_height, frame_width = frame.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("ROI 宽度和高度必须大于 0")
    if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
        raise ValueError(
            f"ROI 超出画面范围：画面 {frame_width}x{frame_height}，"
            f"bbox=({x:g}, {y:g}, {width:g}, {height:g})"
        )
    return x, y, width, height


def scale_bbox_to_frame(
    bbox: Sequence[float],
    template_size: Sequence[int],
    frame,
) -> tuple[float, float, float, float]:
    """Scale a template-space bbox to a live frame with the same framing."""
    if template_size is None or len(template_size) != 2:
        raise ValueError("模板尺寸必须是 (width, height)")
    template_width, template_height = (float(value) for value in template_size)
    if template_width <= 0 or template_height <= 0:
        raise ValueError("模板尺寸必须大于 0")
    x, y, width, height = (float(value) for value in bbox)
    frame_height, frame_width = frame.shape[:2]
    scaled = (
        x * frame_width / template_width,
        y * frame_height / template_height,
        width * frame_width / template_width,
        height * frame_height / template_height,
    )
    return validate_bbox(frame, scaled)


def create_opencv_tracker(tracker_type: str):
    """Create one OpenCV tracker, supporting modern and legacy namespaces."""
    normalized = str(tracker_type).upper()
    if normalized not in TRACKER_FALLBACK_ORDER:
        raise RuntimeError(
            f"不支持的 OpenCV tracker：{tracker_type}；可选 CSRT、KCF、MIL。"
        )
    factory_name = f"Tracker{normalized}_create"
    factory = getattr(cv2, factory_name, None)
    if callable(factory):
        return factory()
    legacy = getattr(cv2, "legacy", None)
    legacy_factory = getattr(legacy, factory_name, None) if legacy is not None else None
    if callable(legacy_factory):
        return legacy_factory()
    raise RuntimeError(
        f"当前 OpenCV {cv2.__version__} 不提供 {normalized} tracker。"
        "可继续尝试其他 tracker；若需要 CSRT/KCF，请单独评估安装 "
        "opencv-contrib-python，避免与现有 opencv-python 冲突。"
    )


def _lost_result(error: str = "") -> dict:
    return {
        "ok": False,
        "bbox_x1": None,
        "bbox_y1": None,
        "bbox_x2": None,
        "bbox_y2": None,
        "bbox_width": None,
        "bbox_height": None,
        "center_x": None,
        "center_y": None,
        "confidence": None,
        "class_name": "custom_object",
        "track_id": 1,
        "status": "lost",
        "tracking_mode": "custom_object_template",
        "tracking_status": "lost",
        "error": error,
    }


def _tracking_result(bbox: Sequence[float]) -> dict:
    x, y, width, height = (float(value) for value in bbox)
    return {
        "ok": True,
        "bbox_x1": x,
        "bbox_y1": y,
        "bbox_x2": x + width,
        "bbox_y2": y + height,
        "bbox_width": width,
        "bbox_height": height,
        "center_x": x + width / 2.0,
        "center_y": y + height / 2.0,
        "confidence": None,
        "class_name": "custom_object",
        "track_id": 1,
        "status": "tracking",
        "tracking_mode": "custom_object_template",
        "tracking_status": "tracking",
        "error": "",
    }


class ObjectTemplateTracker:
    """Single-object tracker with CSRT -> KCF -> MIL fallback."""

    def __init__(self, tracker_type: str = "CSRT") -> None:
        self.requested_tracker_type = str(tracker_type).upper()
        self.actual_tracker_type: str | None = None
        self.tracker = None
        self.initialized = False
        self.last_error = ""
        self.last_result = _lost_result("尚未初始化指定物体追踪")
        self.lost_frame_count = 0
        self.consecutive_lost_frames = 0
        self.total_frame_count = 0
        self.tracked_frame_count = 0

    def _candidate_types(self) -> list[str]:
        if self.requested_tracker_type in TRACKER_FALLBACK_ORDER:
            start = TRACKER_FALLBACK_ORDER.index(self.requested_tracker_type)
            return list(TRACKER_FALLBACK_ORDER[start:])
        return list(TRACKER_FALLBACK_ORDER)

    def initialize(self, frame, bbox: Sequence[float]) -> bool:
        """Initialize on a BGR frame; failures are captured instead of raised."""
        self.reset()
        try:
            normalized_bbox = validate_bbox(frame, bbox)
        except ValueError as exc:
            self.last_error = str(exc)
            self.last_result = _lost_result(self.last_error)
            return False

        errors: list[str] = []
        opencv_bbox = tuple(int(round(value)) for value in normalized_bbox)
        for candidate in self._candidate_types():
            try:
                tracker = create_opencv_tracker(candidate)
                initialized = tracker.init(frame, opencv_bbox)
                if initialized is False:
                    errors.append(f"{candidate} 初始化返回 False")
                    continue
                self.tracker = tracker
                self.actual_tracker_type = candidate
                self.initialized = True
                self.last_result = _tracking_result(opencv_bbox)
                return True
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")

        self.last_error = "；".join(errors) or "没有可用的 OpenCV tracker"
        self.last_result = _lost_result(self.last_error)
        return False

    def update(self, frame) -> dict:
        """Update the tracker and return a stable dictionary result."""
        if not self.initialized or self.tracker is None:
            return _lost_result(self.last_error or "尚未初始化指定物体追踪")
        self.total_frame_count += 1
        try:
            ok, bbox = self.tracker.update(frame)
        except Exception as exc:
            ok, bbox = False, None
            self.last_error = str(exc)
        if not ok or bbox is None:
            self.lost_frame_count += 1
            self.consecutive_lost_frames += 1
            self.last_result = _lost_result(self.last_error)
            return self.last_result.copy()

        try:
            validate_bbox(frame, bbox)
        except ValueError as exc:
            self.lost_frame_count += 1
            self.consecutive_lost_frames += 1
            self.last_error = str(exc)
            self.last_result = _lost_result(self.last_error)
            return self.last_result.copy()
        self.tracked_frame_count += 1
        self.consecutive_lost_frames = 0
        self.last_result = _tracking_result(bbox)
        return self.last_result.copy()

    @property
    def tracking_success_rate(self) -> float:
        if self.total_frame_count <= 0:
            return 1.0 if self.initialized else 0.0
        return self.tracked_frame_count / self.total_frame_count

    def reset(self) -> None:
        self.tracker = None
        self.actual_tracker_type = None
        self.initialized = False
        self.last_error = ""
        self.last_result = _lost_result("尚未初始化指定物体追踪")
        self.lost_frame_count = 0
        self.consecutive_lost_frames = 0
        self.total_frame_count = 0
        self.tracked_frame_count = 0


def draw_object_tracking_result(frame, result: dict):
    """Draw a custom-object bbox and status label on a BGR frame copy."""
    rendered = frame.copy()
    if result.get("ok"):
        x1 = int(round(result["bbox_x1"]))
        y1 = int(round(result["bbox_y1"]))
        x2 = int(round(result["bbox_x2"]))
        y2 = int(round(result["bbox_y2"]))
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.circle(
            rendered,
            (int(round(result["center_x"])), int(round(result["center_y"]))),
            5,
            (0, 0, 255),
            -1,
        )
        label = "CUSTOM OBJECT | track 1"
        color = (255, 180, 0)
    else:
        label = "CUSTOM OBJECT LOST"
        color = (0, 0, 255)
    cv2.putText(
        rendered,
        label,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )
    return rendered
