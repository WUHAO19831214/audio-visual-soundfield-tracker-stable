"""Background OpenCV camera and sounddevice capture workers."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .audio_devices import (
    AudioDeviceError,
    build_input_device_fallback_indices,
    check_audio_input_device,
)
from .camera_devices import open_camera
from .camera_processor import CameraProcessor
from .detector import Detector
from .fusion import CaptureArtifacts, SynchronizedCaptureSession
from .object_template_tracker import (
    ObjectTemplateTracker,
    draw_object_tracking_result,
    scale_bbox_to_frame,
)
from .tennis_ball_tracker import TennisBallTracker, draw_tennis_tracking_result


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


def _level_from_samples(samples: np.ndarray) -> tuple[float, float]:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0, -120.0
    rms = float(np.sqrt(np.mean(values**2)))
    db = 20.0 * math.log10(max(rms, 1e-12))
    return rms, db


class SoundDeviceCapture:
    """A small stateful sounddevice InputStream wrapper used by capture modes."""

    def __init__(
        self,
        audio_device_index: int | None,
        session: SynchronizedCaptureSession,
    ) -> None:
        self.audio_device_index = audio_device_index
        self.session = session
        self._audio_sample_rate = 48_000
        self._ready_event = threading.Event()
        self._lock = threading.RLock()
        self._state = {
            "is_running": False,
            "stream_started": False,
            "receiving": False,
            "started_at": None,
            "last_chunk_at": None,
            "chunk_count": 0,
            "last_rms": 0.0,
            "last_db": -120.0,
            "last_error": "",
            "device_index": audio_device_index,
            "device_name": "",
            "sample_rate": 0,
            "channels": 0,
        }

    @property
    def running(self) -> bool:
        return self.session.audio_recorder.is_running()

    @property
    def audio_error(self) -> str:
        with self._lock:
            state_error = str(self._state.get("last_error", ""))
        return state_error or self.session.audio_recorder.last_error

    def start(self, require_ready: bool = False, timeout_sec: float = 2.0) -> None:
        if self.running:
            if require_ready and not self.wait_until_ready(timeout_sec):
                raise RuntimeError("音频流已启动，但未收到音频 chunk")
            return
        self.stop()
        errors: list[str] = []
        try:
            candidates = build_input_device_fallback_indices(self.audio_device_index)
        except Exception as exc:
            candidates = []
            errors.append(str(exc))
        if not candidates:
            message = "没有可用的麦克风输入设备"
            self._set_error(message)
            raise AudioDeviceError(message)

        import sounddevice as sd

        for candidate_index in candidates:
            self._ready_event.clear()
            try:
                device = check_audio_input_device(candidate_index)
                block_size = max(1, int(device["default_samplerate"] * 0.1))
                self._audio_sample_rate = int(device["default_samplerate"])
                channels = 1 if int(device["max_input_channels"]) >= 1 else 0
                if channels <= 0:
                    raise AudioDeviceError(f"麦克风 index={candidate_index} 没有输入通道")
                with self._lock:
                    self._state.update(
                        is_running=False,
                        stream_started=False,
                        receiving=False,
                        started_at=time.time(),
                        last_chunk_at=None,
                        chunk_count=0,
                        last_rms=0.0,
                        last_db=-120.0,
                        last_error="",
                        device_index=device["index"],
                        device_name=device["name"],
                        sample_rate=self._audio_sample_rate,
                        channels=channels,
                    )
                self.session.set_audio_device_info(self.status())

                def stream_factory(callback, device=device, channels=channels):
                    return sd.InputStream(
                        device=device["index"],
                        channels=channels,
                        samplerate=self._audio_sample_rate,
                        blocksize=block_size,
                        dtype="float32",
                        callback=callback,
                    )

                self.session.audio_recorder.start(
                    stream_factory,
                    sample_rate=self._audio_sample_rate,
                    on_chunk=self._handle_audio_chunk,
                )
                with self._lock:
                    self._state.update(is_running=True, stream_started=True)
                self.session.set_audio_device_info(self.status())
                if require_ready and not self.wait_until_ready(timeout_sec):
                    raise RuntimeError("音频流已启动，但没有收到输入数据")
                return
            except Exception as exc:
                errors.append(f"index={candidate_index}: {exc}")
                self.session.audio_recorder.close()

        message = (
            "音频采集异常，请重新选择麦克风或重启采集。"
            + "；".join(errors)
        )
        self._set_error(message)
        raise AudioDeviceError(message)

    def _set_error(self, error: str) -> None:
        with self._lock:
            self._state["last_error"] = error
        self.session.set_audio_error(error)
        self.session.set_audio_device_info(self.status())

    def _handle_audio_chunk(self, indata, status: str = "") -> None:
        if status:
            self._set_error(status)
        chunk = np.asarray(indata, dtype=np.float32)
        rms, db = _level_from_samples(chunk)
        with self._lock:
            self._state.update(
                receiving=True,
                last_chunk_at=time.time(),
                chunk_count=int(self._state["chunk_count"]) + 1,
                last_rms=rms,
                last_db=db,
            )
        self._ready_event.set()
        self.session.set_audio_device_info(self.status())

    def wait_until_ready(self, timeout_sec: float = 2.0) -> bool:
        return self._ready_event.wait(timeout=max(0.0, timeout_sec))

    def status(self) -> dict:
        with self._lock:
            state = self._state.copy()
        recorder_error = self.session.audio_recorder.last_error
        if recorder_error and not state.get("last_error"):
            state["last_error"] = recorder_error
        state["audio_device_index"] = state.get("device_index")
        state["audio_device_name"] = state.get("device_name")
        return state

    def reset(self) -> None:
        self.stop()
        self._ready_event.clear()
        with self._lock:
            self._state.update(
                is_running=False,
                stream_started=False,
                receiving=False,
                started_at=None,
                last_chunk_at=None,
                chunk_count=0,
                last_rms=0.0,
                last_db=-120.0,
                last_error="",
            )

    def stop(self) -> None:
        self.session.audio_recorder.close()
        with self._lock:
            self._state["is_running"] = False
            self._state["stream_started"] = False
        self.session.set_audio_device_info(self.status())

    close = stop


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
        tracking_mode: str = "person_yolo",
        object_tracker_config: dict | None = None,
    ) -> None:
        super().__init__(camera_index, confidence_threshold, person_only)
        self.audio_device_index = audio_device_index
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
        self.session = SynchronizedCaptureSession(
            output_dir=output_dir,
            tracking_mode=tracking_mode,
            track_class=(
                "custom_object"
                if tracking_mode == "custom_object_template"
                else (
                    "tennis_ball_marker"
                    if tracking_mode == "tennis_ball_color"
                    else "person"
                )
            ),
        )
        self.audio_capture = SoundDeviceCapture(audio_device_index, self.session)

    @property
    def audio_error(self) -> str:
        return self.audio_capture.audio_error

    def start(self) -> None:
        try:
            self.audio_capture.start(require_ready=True, timeout_sec=2.0)
        except Exception as exc:
            self.session.set_audio_error(str(exc))
        super().start()

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
                if self.tracking_mode == "custom_object_template":
                    annotated, tracks, primary = self._process_custom_object_frame(
                        frame, timestamp
                    )
                elif self.tracking_mode == "tennis_ball_color":
                    annotated, tracks, primary = self._process_tennis_ball_frame(
                        frame, timestamp
                    )
                else:
                    annotated, tracks, primary = self.processor.process_frame(
                        frame, timestamp
                    )
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

    def _process_custom_object_frame(self, frame, timestamp: float):
        if self.object_tracker is None:
            raise RuntimeError("指定物体 tracker 未创建")
        if not self._object_initialization_attempted:
            self._object_initialization_attempted = True
            try:
                bbox = scale_bbox_to_frame(
                    self.object_tracker_config["bbox"],
                    self.object_tracker_config["template_size"],
                    frame,
                )
            except Exception as exc:
                self.object_tracker.last_error = str(exc)
                initialized = False
            else:
                initialized = self.object_tracker.initialize(frame, bbox)
            result = self.object_tracker.last_result.copy()
            if not initialized:
                result["error"] = self.object_tracker.last_error
        else:
            result = self.object_tracker.update(frame)
        track = result if result.get("ok") else None
        self.session.add_visual_track(
            timestamp,
            track,
            tracking_mode="custom_object_template",
            track_class="custom_object",
            tracking_status=result.get("tracking_status", "lost"),
            record_lost=True,
        )
        annotated = draw_object_tracking_result(frame, result)
        return annotated, [result] if track else [], track

    def _process_tennis_ball_frame(self, frame, timestamp: float):
        if self.tennis_tracker is None:
            raise RuntimeError("网球标记 tracker 未创建")
        result = self.tennis_tracker.update(frame)
        track = result if result.get("ok") else None
        self.session.add_visual_track(
            timestamp,
            track,
            tracking_mode="tennis_ball_color",
            track_class="tennis_ball_marker",
            tracking_status=result.get("tracking_status", "lost"),
            record_lost=True,
            lost_frame_count=result.get("lost_frame_count", 0),
        )
        annotated = draw_tennis_tracking_result(frame, result)
        return annotated, [result] if track else [], track

    def status(self) -> dict:
        result = super().status()
        result.update(self.session.status())
        audio_status = self.audio_capture.status()
        result["audio_capture"] = audio_status
        result["audio_error"] = audio_status.get("last_error", "")
        result["tracking_mode"] = self.tracking_mode
        if self.object_tracker is not None:
            result["object_tracker_type"] = self.object_tracker.actual_tracker_type
            result["object_tracking_status"] = self.object_tracker.last_result.get(
                "tracking_status", "lost"
            )
            result["consecutive_lost_frames"] = (
                self.object_tracker.consecutive_lost_frames
            )
        if self.tennis_tracker is not None:
            result["object_tracker_type"] = "HSV color + contour"
            result["object_tracking_status"] = self.tennis_tracker.last_result.get(
                "tracking_status", "lost"
            )
            result["consecutive_lost_frames"] = (
                self.tennis_tracker.consecutive_lost_frames
            )
            result["marker_radius"] = self.tennis_tracker.last_result.get(
                "marker_radius"
            )
            result["marker_area"] = self.tennis_tracker.last_result.get("marker_area")
            result["marker_circularity"] = self.tennis_tracker.last_result.get(
                "marker_circularity"
            )
        return result

    def stop_and_finalize(self, **finalize_options) -> CaptureArtifacts:
        self.stop()
        return self.session.finalize(**finalize_options)

    def stop(self) -> None:
        self.audio_capture.stop()
        super().stop()


class LocalAudioWorker:
    """Capture a sounddevice microphone into an existing fusion session."""

    def __init__(
        self,
        audio_device_index: int | None,
        session: SynchronizedCaptureSession,
    ) -> None:
        self.audio_device_index = audio_device_index
        self.session = session
        self.audio_capture = SoundDeviceCapture(audio_device_index, session)

    @property
    def running(self) -> bool:
        return self.audio_capture.running

    @property
    def audio_error(self) -> str:
        return self.audio_capture.audio_error

    def start(self, require_ready: bool = False, timeout_sec: float = 2.0) -> None:
        self.audio_capture.start(require_ready=require_ready, timeout_sec=timeout_sec)

    def status(self) -> dict:
        return self.audio_capture.status()

    def stop(self) -> None:
        self.audio_capture.stop()

    close = stop


class AudioLevelWorker:
    """Continuously monitor a sounddevice input level for the UI meter."""

    def __init__(self, audio_device_index: int | None) -> None:
        self.audio_device_index = audio_device_index
        self._audio_stream = None
        self._lock = threading.RLock()
        self._stream_lock = threading.RLock()
        self._stream_generation = 0
        self._running = False
        self._closed = False
        self._level = {
            "name": "",
            "sample_rate": 0,
            "rms": 0.0,
            "peak": 0.0,
            "dbfs": -120.0,
            "meter": 0.0,
        }
        self.error = ""

    @property
    def running(self) -> bool:
        with self._stream_lock:
            return self._running and self._audio_stream is not None

    def start(self) -> None:
        if self.running:
            return
        self.stop()
        try:
            device = check_audio_input_device(self.audio_device_index)
            import sounddevice as sd

            sample_rate = int(device["default_samplerate"])
            block_size = max(1, int(sample_rate * 0.08))
            with self._lock:
                self._level.update(
                    name=device["name"],
                    sample_rate=sample_rate,
                )
            with self._stream_lock:
                self._stream_generation += 1
                generation = self._stream_generation
                self._closed = False
            stream = sd.InputStream(
                device=device["index"],
                channels=1,
                samplerate=sample_rate,
                blocksize=block_size,
                dtype="float32",
                callback=lambda indata, frames, time_info, status: self._audio_callback(
                    generation, indata, frames, time_info, status
                ),
            )
            with self._stream_lock:
                self._audio_stream = stream
                self._running = True
            stream.start()
        except Exception as exc:
            self.error = str(exc)
            self.stop()

    def _audio_callback(self, generation, indata, frames, time_info, status) -> None:
        del frames, time_info
        try:
            with self._stream_lock:
                if (
                    not self._running
                    or self._closed
                    or generation != self._stream_generation
                ):
                    return
            if status:
                self.error = str(status)
            samples = np.asarray(indata, dtype=np.float32).reshape(-1)
            if samples.size:
                rms = float(np.sqrt(np.mean(samples**2)))
                peak = float(np.max(np.abs(samples)))
            else:
                rms = 0.0
                peak = 0.0
            dbfs = 20.0 * math.log10(max(rms, 1e-12))
            meter = min(1.0, max(0.0, (dbfs + 80.0) / 60.0))
            with self._lock:
                self._level.update(rms=rms, peak=peak, dbfs=dbfs, meter=meter)
        except Exception as exc:
            self.error = f"输入电平 callback 异常：{exc}"

    def status(self) -> dict:
        with self._lock:
            return self._level.copy()

    def stop(self) -> None:
        with self._stream_lock:
            self._running = False
            self._closed = True
            self._stream_generation += 1
            stream = self._audio_stream
            self._audio_stream = None
        if stream is not None:
            try:
                stream.stop()
            except Exception as exc:
                self.error = f"停止输入电平流失败：{exc}"
            try:
                stream.close()
            except Exception as exc:
                self.error = f"关闭输入电平流失败：{exc}"

    close = stop
