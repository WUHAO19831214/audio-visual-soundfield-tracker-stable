import sys
from pathlib import Path

import numpy as np

import src.local_capture as local_capture
from src.fusion import SynchronizedCaptureSession


class FakeInputStream:
    instances = []

    def __init__(self, callback, **kwargs) -> None:
        self.callback = callback
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False
        FakeInputStream.instances.append(self)

    def start(self) -> None:
        self.started = True
        self.callback(np.ones((4800, 1), dtype=np.float32) * 0.1, 4800, None, None)

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class FakeSoundDevice:
    InputStream = FakeInputStream


def test_sounddevice_capture_state_and_close(monkeypatch, tmp_path: Path) -> None:
    FakeInputStream.instances.clear()
    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice)
    monkeypatch.setattr(
        local_capture,
        "build_input_device_fallback_indices",
        lambda index: [index],
    )
    monkeypatch.setattr(
        local_capture,
        "check_audio_input_device",
        lambda index: {
            "index": index,
            "name": "Wireless Mic Rx",
            "sample_rate": 48_000,
            "channels": 1,
            "default_samplerate": 48_000.0,
            "max_input_channels": 1,
        },
    )
    session = SynchronizedCaptureSession(output_dir=tmp_path)
    capture = local_capture.SoundDeviceCapture(3, session)

    capture.start(require_ready=True, timeout_sec=0.2)
    status = capture.status()

    assert status["stream_started"] is True
    assert status["receiving"] is True
    assert status["chunk_count"] == 1
    assert status["device_index"] == 3
    assert status["device_name"] == "Wireless Mic Rx"
    assert status["last_rms"] > 0
    assert session.audio_snapshot()

    capture.stop()

    assert FakeInputStream.instances[0].stopped is True
    assert FakeInputStream.instances[0].closed is True
    assert capture.status()["stream_started"] is False
