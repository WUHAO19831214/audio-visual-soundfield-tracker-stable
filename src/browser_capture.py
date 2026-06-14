"""WebRTC processors, imported only when a browser capture mode is selected."""

from __future__ import annotations

import threading
import time

import av
from streamlit_webrtc import AudioProcessorBase, VideoProcessorBase

from .camera_processor import CameraProcessor
from .detector import Detector
from .fusion import SynchronizedCaptureSession


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
    ) -> None:
        self.session = session
        self.processor = CameraProcessor(
            Detector(
                confidence_threshold=confidence_threshold,
                person_only=person_only,
            )
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        timestamp = self.session.clock.now()
        image = frame.to_ndarray(format="bgr24")
        annotated, _, primary = self.processor.process_frame(image, timestamp)
        self.session.add_visual_track(timestamp, primary)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


class FusionAudioProcessor(AudioProcessorBase):
    def __init__(self, session: SynchronizedCaptureSession) -> None:
        self.session = session

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        samples = frame.to_ndarray()
        self.session.audio_recorder.add_samples(samples, frame.sample_rate)
        return frame

