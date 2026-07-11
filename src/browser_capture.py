"""WebRTC processors, imported only when a browser capture mode is selected."""

from __future__ import annotations

import threading
import time

import av
from streamlit_webrtc import AudioProcessorBase, VideoProcessorBase

from .camera_processor import CameraProcessor
from .detector import Detector
from .fusion import SynchronizedCaptureSession
from .object_template_tracker import (
    ObjectTemplateTracker,
    draw_object_tracking_result,
    scale_bbox_to_frame,
)
from .tennis_ball_tracker import TennisBallTracker, draw_tennis_tracking_result


class LiveCountingProcessor(VideoProcessorBase):
    def __init__(self, confidence_threshold: float, person_only: bool) -> None:
        self.processor = CameraProcessor(
            Detector(
                confidence_threshold=confidence_threshold,
                person_only=person_only,
            )
        )
        self.lock = threading.Lock()
        self.latest_status: dict = {"count": 0, "primary": None}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        annotated, tracks, primary = self.processor.process_frame(image, time.time())
        with self.lock:
            self.latest_status = {"count": len(tracks), "primary": primary}
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def status(self) -> dict:
        with self.lock:
            return self.latest_status.copy()


class FusionVideoProcessor(VideoProcessorBase):
    def __init__(
        self,
        session: SynchronizedCaptureSession,
        confidence_threshold: float,
        person_only: bool,
        tracking_mode: str = "person_yolo",
        object_tracker_config: dict | None = None,
    ) -> None:
        self.session = session
        self.tracking_mode = tracking_mode
        self.object_tracker_config = dict(object_tracker_config or {})
        self.object_tracker = (
            ObjectTemplateTracker(self.object_tracker_config.get("tracker_type", "CSRT"))
            if tracking_mode == "custom_object_template"
            else None
        )
        self.tennis_tracker = (
            TennisBallTracker(**self.object_tracker_config)
            if tracking_mode == "tennis_ball_color"
            else None
        )
        self._object_initialization_attempted = False
        self.processor = CameraProcessor(
            Detector(
                confidence_threshold=confidence_threshold,
                person_only=person_only,
            )
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        timestamp = self.session.clock.now()
        image = frame.to_ndarray(format="bgr24")
        if self.tracking_mode == "custom_object_template":
            annotated = self._process_custom_object_frame(image, timestamp)
        elif self.tracking_mode == "tennis_ball_color":
            annotated = self._process_tennis_ball_frame(image, timestamp)
        else:
            annotated, _, primary = self.processor.process_frame(image, timestamp)
            self.session.add_visual_track(timestamp, primary)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def _process_custom_object_frame(self, image, timestamp: float):
        if self.object_tracker is None:
            return draw_object_tracking_result(image, {"ok": False})
        if not self._object_initialization_attempted:
            self._object_initialization_attempted = True
            try:
                bbox = scale_bbox_to_frame(
                    self.object_tracker_config["bbox"],
                    self.object_tracker_config["template_size"],
                    image,
                )
            except Exception as exc:
                self.object_tracker.last_error = str(exc)
                initialized = False
            else:
                initialized = self.object_tracker.initialize(image, bbox)
            result = self.object_tracker.last_result.copy()
            if not initialized:
                result["error"] = self.object_tracker.last_error
        else:
            result = self.object_tracker.update(image)
        self.session.add_visual_track(
            timestamp,
            result if result.get("ok") else None,
            tracking_mode="custom_object_template",
            track_class="custom_object",
            tracking_status=result.get("tracking_status", "lost"),
            record_lost=True,
        )
        return draw_object_tracking_result(image, result)

    def _process_tennis_ball_frame(self, image, timestamp: float):
        if self.tennis_tracker is None:
            return draw_tennis_tracking_result(image, {"ok": False})
        result = self.tennis_tracker.update(image)
        self.session.add_visual_track(
            timestamp,
            result if result.get("ok") else None,
            tracking_mode="tennis_ball_color",
            track_class="tennis_ball_marker",
            tracking_status=result.get("tracking_status", "lost"),
            record_lost=True,
            lost_frame_count=result.get("lost_frame_count", 0),
        )
        return draw_tennis_tracking_result(image, result)


class FusionAudioProcessor(AudioProcessorBase):
    def __init__(self, session: SynchronizedCaptureSession) -> None:
        self.session = session

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        samples = frame.to_ndarray()
        self.session.audio_recorder.add_samples(samples, frame.sample_rate)
        return frame
