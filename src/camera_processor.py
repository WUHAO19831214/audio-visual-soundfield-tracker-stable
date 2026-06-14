"""Offline person detection, lightweight tracking, and primary-track selection."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
import cv2
import numpy as np

from .detector import Detection, Detector, PersonDetector


@dataclass
class Track:
    track_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    first_seen: float
    last_seen: float
    hits: int = 1
    missed: int = 0
    class_name: str = "person"
    tracking_id_available: bool = True

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def bbox_width(self) -> float:
        return float(self.x2 - self.x1)

    @property
    def bbox_height(self) -> float:
        return float(self.y2 - self.y1)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    def to_dict(self) -> dict:
        row = asdict(self)
        row.update(
            center_x=self.center_x,
            center_y=self.center_y,
            bbox_width=self.bbox_width,
            bbox_height=self.bbox_height,
            duration_sec=self.duration_sec,
        )
        return row


class CentroidTracker:
    """Assign stable IDs using nearest-centroid matching."""

    def __init__(self, max_missed: int = 12, max_distance_ratio: float = 0.18):
        self.max_missed = max_missed
        self.max_distance_ratio = max_distance_ratio
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(
        self,
        detections: list[Detection],
        timestamp: float,
        frame_shape: tuple[int, ...],
    ) -> list[Track]:
        height, width = frame_shape[:2]
        max_distance = math.hypot(width, height) * self.max_distance_ratio
        unmatched_tracks = set(self._tracks)
        unmatched_detections = set(range(len(detections)))
        pairs: list[tuple[float, int, int]] = []

        for track_id, track in self._tracks.items():
            for index, detection in enumerate(detections):
                dx = track.center_x - detection.center[0]
                dy = track.center_y - detection.center[1]
                pairs.append((math.hypot(dx, dy), track_id, index))

        for distance, track_id, detection_index in sorted(pairs):
            if distance > max_distance:
                break
            if track_id not in unmatched_tracks or detection_index not in unmatched_detections:
                continue
            detection = detections[detection_index]
            track = self._tracks[track_id]
            track.x1, track.y1 = detection.x1, detection.y1
            track.x2, track.y2 = detection.x2, detection.y2
            track.confidence = detection.confidence
            track.class_name = detection.class_name
            track.tracking_id_available = detection.tracking_id_available
            track.last_seen = timestamp
            track.hits += 1
            track.missed = 0
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(detection_index)

        for track_id in unmatched_tracks:
            self._tracks[track_id].missed += 1

        for detection_index in unmatched_detections:
            detection = detections[detection_index]
            self._tracks[self._next_id] = Track(
                track_id=self._next_id,
                x1=detection.x1,
                y1=detection.y1,
                x2=detection.x2,
                y2=detection.y2,
                confidence=detection.confidence,
                first_seen=timestamp,
                last_seen=timestamp,
                class_name=detection.class_name,
                tracking_id_available=detection.tracking_id_available,
            )
            self._next_id += 1

        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missed > self.max_missed
        ]
        for track_id in expired:
            del self._tracks[track_id]

        return [track for track in self._tracks.values() if track.missed == 0]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1


class PrimaryTrackSelector:
    """Favor persistent tracks near the image center, with mild stickiness."""

    def __init__(self) -> None:
        self.current_track_id: int | None = None

    def select(
        self, tracks: list[Track], frame_shape: tuple[int, ...]
    ) -> Track | None:
        if not tracks:
            self.current_track_id = None
            return None
        if len(tracks) == 1:
            self.current_track_id = tracks[0].track_id
            return tracks[0]

        height, width = frame_shape[:2]
        diagonal = max(math.hypot(width, height), 1.0)

        def score(track: Track) -> float:
            center_distance = math.hypot(
                track.center_x - width / 2.0, track.center_y - height / 2.0
            ) / diagonal
            persistence = math.log1p(track.hits)
            stickiness = 0.35 if track.track_id == self.current_track_id else 0.0
            return 2.0 * persistence - 1.5 * center_distance + stickiness

        selected = max(tracks, key=score)
        self.current_track_id = selected.track_id
        return selected


class CameraProcessor:
    """Detect people, update IDs, choose the main target, and annotate frames."""

    def __init__(self, detector: Detector | None = None) -> None:
        self.detector = detector or PersonDetector()
        self.tracker = CentroidTracker()
        self.selector = PrimaryTrackSelector()
        self._native_tracks: dict[int, Track] = {}
        self.tracking_warning = ""

    @property
    def backend_name(self) -> str:
        return self.detector.backend_name

    def process_frame(
        self, frame: np.ndarray, timestamp: float
    ) -> tuple[np.ndarray, list[dict], dict | None]:
        if self.detector.supports_native_tracking:
            detections = self.detector.track(frame)
            if detections and all(item.track_id is not None for item in detections):
                tracks = self._update_native_tracks(detections, timestamp)
            else:
                tracks = self.tracker.update(detections, timestamp, frame.shape)
        else:
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections, timestamp, frame.shape)
        self.tracking_warning = getattr(self.detector, "tracking_warning", "")
        primary = self.selector.select(tracks, frame.shape)
        annotated = frame.copy()

        for track in tracks:
            is_primary = primary is not None and track.track_id == primary.track_id
            color = (0, 220, 0) if is_primary else (255, 170, 0)
            thickness = 3 if is_primary else 2
            cv2.rectangle(
                annotated,
                (track.x1, track.y1),
                (track.x2, track.y2),
                color,
                thickness,
            )
            label = f"{track.class_name} #{track.track_id} {track.confidence:.2f}"
            if is_primary:
                label += " MAIN"
            cv2.putText(
                annotated,
                label,
                (track.x1, max(22, track.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.circle(
                annotated,
                (int(track.center_x), int(track.center_y)),
                5,
                color,
                -1,
            )

        cv2.putText(
            annotated,
            f"Persons: {len(tracks)}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (40, 230, 255),
            2,
            cv2.LINE_AA,
        )
        primary_row = primary.to_dict() if primary is not None else None
        return annotated, [track.to_dict() for track in tracks], primary_row

    def _update_native_tracks(
        self, detections: list[Detection], timestamp: float
    ) -> list[Track]:
        current_ids: set[int] = set()
        for detection in detections:
            track_id = int(detection.track_id)
            current_ids.add(track_id)
            track = self._native_tracks.get(track_id)
            if track is None:
                track = Track(
                    track_id=track_id,
                    x1=detection.x1,
                    y1=detection.y1,
                    x2=detection.x2,
                    y2=detection.y2,
                    confidence=detection.confidence,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    class_name=detection.class_name,
                    tracking_id_available=detection.tracking_id_available,
                )
                self._native_tracks[track_id] = track
            else:
                track.x1, track.y1 = detection.x1, detection.y1
                track.x2, track.y2 = detection.x2, detection.y2
                track.confidence = detection.confidence
                track.class_name = detection.class_name
                track.tracking_id_available = detection.tracking_id_available
                track.last_seen = timestamp
                track.hits += 1
                track.missed = 0

        for track_id, track in list(self._native_tracks.items()):
            if track_id not in current_ids:
                track.missed += 1
            if track.missed > 30:
                del self._native_tracks[track_id]
        return [self._native_tracks[track_id] for track_id in current_ids]

    def detect_image(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        detections = self.detector.detect(frame)
        annotated = frame.copy()
        for index, detection in enumerate(detections, start=1):
            cv2.rectangle(
                annotated,
                (detection.x1, detection.y1),
                (detection.x2, detection.y2),
                (0, 220, 0),
                2,
            )
            cv2.putText(
                annotated,
                f"{detection.class_name} {detection.confidence:.2f}",
                (detection.x1, max(22, detection.y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )
        return annotated, detections

    def reset(self) -> None:
        self.tracker.reset()
        self._native_tracks.clear()
        self.selector.current_track_id = None
