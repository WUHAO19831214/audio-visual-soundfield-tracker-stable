"""Background OpenCV camera and sounddevice capture workers."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .audio_devices import AudioDeviceError, check_audio_input_device
from .camera_devices import open_camera
from .camera_processor import CameraProcessor
from .detector import Detector
from .fusion import CaptureArtifacts, SynchronizedCaptureSession


def check_writable_directory(path: str | Path) -> tuple[bool, str]:
    directory = Path(path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".write_test"
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
        return True, str(directory)
    except Exception as exc:
        return False, f"{directory}: {exc}"


class LocalCameraWorker:
    def __init__(
        self,
        camera_index: int,
        confidence_threshold: float = 0.25,
        person_only: bool = True,
    ) -> None:
        self.camera_index = camera_index
        self.processor = CameraProcessor(
            Detector(
                confidence_threshold=confidence_threshold,
                person_only=person_only,
            )
        )
        self._capture = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._latest_frame: np.ndarray | None = None
        self._status = {"count": 0, "primary": None, "error": ""}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._capture = open_camera(self.camera_index)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self._capture.release()
            self._capture = None
            raise RuntimeError(f"摄像头 index={self.camera_index} 无法读取第一帧")
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        failed_reads = 0
        try:
            while not self._stop_event.is_set():
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    failed_reads += 1
                    if failed_reads >= 20:
                        raise RuntimeError("摄像头连续读取失败")
                    time.sleep(0.02)
                    continue
                failed_reads = 0
                annotated, tracks, primary = self.processor.process_frame(
                    frame, time.time()
                )
                with self._lock:
                    self._latest_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    self._status = {
                        "count": len(tracks),
                        "primary": primary,
                        "error": "",
                        "tracking_warning": getattr(
                            self.processor, "tracking_warning", ""
                        ),
                    }
        except Exception as exc:
            with self._lock:
                self._status["error"] = str(exc)
        finally:
            if self._capture is not None:
                self._capture.release()
                self._capture = None

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def status(self) -> dict:
        with self._lock:
            result = self._status.copy()
        result.update(
            running=self.running,
            backend=self.processor.backend_name,
            weight_path=str(self.processor.detector.weight_path or ""),
        )
        return result

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


class LocalFusionWorker(LocalCameraWorker):
    def __init__(
        self,
        camera_index: int,
        audio_device_index: int | None,
        output_dir: str | Path,
        confidence_threshold: float = 0.25,
        person_only: bool = True,
    ) -> None:
        super().__init__(camera_index, confidence_threshold, person_only)
        self.audio_device_index = audio_device_index
        self.session = SynchronizedCaptureSession(output_dir=output_dir)
        self._audio_stream = None
        self._audio_sample_rate = 48_000
        self.audio_error = ""

    def start(self) -> None:
        super().start()
        try:
            device = check_audio_input_device(self.audio_device_index)
            import sounddevice as sd

            block_size = max(1, int(device["sample_rate"] * 0.1))
            self._audio_sample_rate = device["sample_rate"]
            self._audio_stream = sd.InputStream(
                device=self.audio_device_index,
                channels=1,
                samplerate=device["sample_rate"],
                blocksize=block_size,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._audio_stream.start()
        except Exception as exc:
            self.audio_error = str(exc)

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.audio_error = str(status)
        self.session.audio_recorder.add_samples(
            np.asarray(indata).copy(), int(self._audio_sample_rate)
        )

    def _run(self) -> None:
        failed_reads = 0
        try:
            while not self._stop_event.is_set():
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    failed_reads += 1
                    if failed_reads >= 20:
                        raise RuntimeError("摄像头连续读取失败")
                    time.sleep(0.02)
                    continue
                failed_reads = 0
                timestamp = self.session.clock.now()
                annotated, tracks, primary = self.processor.process_frame(frame, timestamp)
                self.session.add_visual_track(timestamp, primary)
                with self._lock:
                    self._latest_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    self._status = {
                        "count": len(tracks),
                        "primary": primary,
                        "error": "",
                        "tracking_warning": getattr(
                            self.processor, "tracking_warning", ""
                        ),
                    }
        except Exception as exc:
            with self._lock:
                self._status["error"] = str(exc)
        finally:
            if self._capture is not None:
                self._capture.release()
                self._capture = None

    def status(self) -> dict:
        result = super().status()
        result.update(self.session.status())
        result["audio_error"] = self.audio_error
        return result

    def stop_and_finalize(self) -> CaptureArtifacts:
        self.stop()
        return self.session.finalize()

    def stop(self) -> None:
        if self._audio_stream is not None:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            finally:
                self._audio_stream = None
        super().stop()
