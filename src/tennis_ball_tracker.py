"""Lightweight HSV-based tracking for a bright tennis-ball marker."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import cv2
import numpy as np


DEFAULT_HSV_LOWER = (25, 70, 70)
DEFAULT_HSV_UPPER = (75, 255, 255)


def _hsv_triplet(values: Sequence[int | float]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError("HSV 阈值必须包含 H、S、V 三个值")
    return (
        int(np.clip(values[0], 0, 179)),
        int(np.clip(values[1], 0, 255)),
        int(np.clip(values[2], 0, 255)),
    )


def make_tennis_mask(
    frame: np.ndarray,
    hsv_lower: Sequence[int | float] = DEFAULT_HSV_LOWER,
    hsv_upper: Sequence[int | float] = DEFAULT_HSV_UPPER,
) -> np.ndarray:
    """Return a cleaned binary mask for the configured tennis-ball color."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame 必须是有效的 BGR 三通道图像")
    lower = _hsv_triplet(hsv_lower)
    upper = _hsv_triplet(hsv_upper)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if lower[0] <= upper[0]:
        mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    else:
        low_mask = cv2.inRange(
            hsv,
            np.array((0, lower[1], lower[2]), np.uint8),
            np.array(upper, np.uint8),
        )
        high_mask = cv2.inRange(
            hsv,
            np.array(lower, np.uint8),
            np.array((179, upper[1], upper[2]), np.uint8),
        )
        mask = cv2.bitwise_or(low_mask, high_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def find_ball_candidates(mask: np.ndarray) -> list[dict]:
    """Extract contour geometry used for marker filtering and ranking."""
    if mask is None or mask.ndim != 2:
        raise ValueError("mask 必须是单通道图像")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (
            float(4.0 * math.pi * area / (perimeter * perimeter))
            if perimeter > 0.0
            else 0.0
        )
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        x, y, width, height = cv2.boundingRect(contour)
        candidates.append(
            {
                "center_x": float(center_x),
                "center_y": float(center_y),
                "radius": float(radius),
                "area": area,
                "circularity": circularity,
                "bbox": (float(x), float(y), float(width), float(height)),
                "contour": contour,
            }
        )
    return candidates


def choose_best_candidate(
    candidates: Iterable[dict],
    previous_center: Sequence[int | float] | None = None,
) -> dict | None:
    """Choose the most plausible marker, favoring temporal continuity."""
    options = list(candidates)
    if not options:
        return None
    if previous_center is None:
        return max(
            options,
            key=lambda item: float(item.get("area", 0.0))
            * max(float(item.get("circularity", 0.0)), 0.05),
        )
    previous_x, previous_y = map(float, previous_center[:2])
    return min(
        options,
        key=lambda item: math.hypot(
            float(item.get("center_x", 0.0)) - previous_x,
            float(item.get("center_y", 0.0)) - previous_y,
        )
        / max(0.25 + float(item.get("circularity", 0.0)), 0.25),
    )


def estimate_hsv_range_from_roi(
    frame: np.ndarray,
    bbox: Sequence[int | float],
    h_margin: int = 12,
    s_margin: int = 60,
    v_margin: int = 60,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Estimate a robust HSV range from a marker ROI."""
    if frame is None or frame.ndim != 3:
        raise ValueError("frame 必须是有效图像")
    if len(bbox) != 4:
        raise ValueError("bbox 必须是 (x, y, width, height)")
    x, y, width, height = (int(round(float(value))) for value in bbox)
    frame_height, frame_width = frame.shape[:2]
    if width <= 0 or height <= 0 or x < 0 or y < 0:
        raise ValueError("ROI 坐标和尺寸无效")
    if x + width > frame_width or y + height > frame_height:
        raise ValueError("ROI 超出图像边界")
    roi = frame[y : y + height, x : x + width]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    useful = hsv[(hsv[:, 1] >= 35) & (hsv[:, 2] >= 35)]
    if useful.size == 0:
        useful = hsv
    center = np.median(useful, axis=0)
    lower = (
        max(0, int(round(center[0])) - int(h_margin)),
        max(0, int(round(center[1])) - int(s_margin)),
        max(0, int(round(center[2])) - int(v_margin)),
    )
    upper = (
        min(179, int(round(center[0])) + int(h_margin)),
        min(255, int(round(center[1])) + int(s_margin)),
        min(255, int(round(center[2])) + int(v_margin)),
    )
    return lower, upper


def _lost_result(lost_frames: int, error: str = "") -> dict:
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
        "class_name": "tennis_ball_marker",
        "track_id": 1,
        "status": "lost",
        "tracking_status": "lost",
        "tracking_mode": "tennis_ball_color",
        "marker_radius": None,
        "marker_area": None,
        "marker_circularity": None,
        "lost_frame_count": int(lost_frames),
        "error": error,
    }


class TennisBallTracker:
    """Track one bright marker using HSV color and contour geometry."""

    def __init__(
        self,
        hsv_lower: Sequence[int | float] = DEFAULT_HSV_LOWER,
        hsv_upper: Sequence[int | float] = DEFAULT_HSV_UPPER,
        min_area: float = 80.0,
        max_area: float = 50000.0,
        min_circularity: float = 0.45,
        smoothing: float = 0.35,
        max_lost_frames: int = 30,
    ) -> None:
        self.hsv_lower = _hsv_triplet(hsv_lower)
        self.hsv_upper = _hsv_triplet(hsv_upper)
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.min_circularity = float(min_circularity)
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.max_lost_frames = max(1, int(max_lost_frames))
        if self.min_area < 0 or self.max_area <= self.min_area:
            raise ValueError("marker 面积阈值无效")
        self.reset()

    def reset(self) -> None:
        self.previous_center: tuple[float, float] | None = None
        self.consecutive_lost_frames = 0
        self.total_frames = 0
        self.tracked_frames = 0
        self.last_mask: np.ndarray | None = None
        self.last_result = _lost_result(0)

    def update(self, frame: np.ndarray) -> dict:
        self.total_frames += 1
        try:
            mask = make_tennis_mask(frame, self.hsv_lower, self.hsv_upper)
            self.last_mask = mask
            candidates = [
                candidate
                for candidate in find_ball_candidates(mask)
                if self.min_area <= candidate["area"] <= self.max_area
                and candidate["circularity"] >= self.min_circularity
            ]
            candidate = choose_best_candidate(candidates, self.previous_center)
        except Exception as exc:
            self.consecutive_lost_frames += 1
            self.last_result = _lost_result(self.consecutive_lost_frames, str(exc))
            return self.last_result.copy()
        if candidate is None:
            self.consecutive_lost_frames += 1
            self.last_result = _lost_result(self.consecutive_lost_frames)
            return self.last_result.copy()

        raw_center = (float(candidate["center_x"]), float(candidate["center_y"]))
        if self.previous_center is None:
            center = raw_center
        else:
            alpha = self.smoothing
            center = (
                alpha * raw_center[0] + (1.0 - alpha) * self.previous_center[0],
                alpha * raw_center[1] + (1.0 - alpha) * self.previous_center[1],
            )
        self.previous_center = center
        self.consecutive_lost_frames = 0
        self.tracked_frames += 1
        radius = float(candidate["radius"])
        x1, y1 = center[0] - radius, center[1] - radius
        x2, y2 = center[0] + radius, center[1] + radius
        self.last_result = {
            "ok": True,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "bbox_width": radius * 2.0,
            "bbox_height": radius * 2.0,
            "center_x": center[0],
            "center_y": center[1],
            "confidence": float(np.clip(candidate["circularity"], 0.0, 1.0)),
            "class_name": "tennis_ball_marker",
            "track_id": 1,
            "status": "tracking",
            "tracking_status": "tracking",
            "tracking_mode": "tennis_ball_color",
            "marker_radius": radius,
            "marker_area": float(candidate["area"]),
            "marker_circularity": float(candidate["circularity"]),
            "lost_frame_count": 0,
            "error": "",
        }
        return self.last_result.copy()

    @property
    def tracking_success_rate(self) -> float:
        return self.tracked_frames / self.total_frames if self.total_frames else 0.0


def draw_tennis_tracking_result(frame: np.ndarray, result: dict) -> np.ndarray:
    """Draw the selected marker or a nonfatal lost-state notice."""
    annotated = frame.copy()
    if result.get("ok"):
        center = (int(round(result["center_x"])), int(round(result["center_y"])))
        radius = max(2, int(round(result.get("marker_radius") or 2)))
        x1 = int(round(result.get("bbox_x1", center[0] - radius)))
        y1 = int(round(result.get("bbox_y1", center[1] - radius)))
        x2 = int(round(result.get("bbox_x2", center[0] + radius)))
        y2 = int(round(result.get("bbox_y2", center[1] + radius)))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.circle(annotated, center, radius, (0, 255, 255), 3)
        cv2.circle(annotated, center, 3, (0, 0, 255), -1)
        label = (
            "Tennis marker #1 "
            f"circ={float(result.get('marker_circularity') or 0.0):.2f}"
        )
        cv2.putText(
            annotated,
            label,
            (max(0, center[0] - radius), max(24, center[1] - radius - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        lost = int(result.get("lost_frame_count", 0))
        cv2.putText(
            annotated,
            f"Tennis marker lost ({lost})",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated
