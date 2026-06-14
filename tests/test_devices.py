import numpy as np
import pytest

import src.audio_devices as audio_devices
from src.audio_devices import AudioDeviceError
from src.camera_devices import list_available_cameras


class FakeCapture:
    def __init__(self, index: int, opened: bool) -> None:
        self.index = index
        self.opened = opened

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop, value) -> None:
        pass

    def read(self):
        if self.opened:
            return True, np.zeros((10, 10, 3), dtype=np.uint8)
        return False, None

    def release(self) -> None:
        pass


def test_camera_scan_skips_failed_indexes() -> None:
    def factory(index, backend=None):
        if index == 0:
            raise RuntimeError("camera busy")
        return FakeCapture(index, opened=index == 1)

    devices = list_available_cameras(max_index=3, capture_factory=factory)

    assert devices == [{"index": 1, "name": "Camera 1"}]


def test_audio_enumeration_failure_has_friendly_message(monkeypatch) -> None:
    class BrokenSoundDevice:
        @staticmethod
        def query_devices(*args, **kwargs):
            raise RuntimeError("PortAudio unavailable")

    monkeypatch.setattr(audio_devices, "_load_sounddevice", lambda: BrokenSoundDevice())

    with pytest.raises(AudioDeviceError, match="无法枚举音频输入设备"):
        audio_devices.list_audio_input_devices()


def test_default_input_accepts_sounddevice_pair_object(monkeypatch) -> None:
    class Pair:
        def __getitem__(self, index):
            return (2, 3)[index]

    class Default:
        device = Pair()

    class FakeSoundDevice:
        default = Default()

        @staticmethod
        def query_devices(index, kind):
            assert index == 2
            assert kind == "input"
            return {
                "name": "Test Microphone",
                "max_input_channels": 1,
                "default_samplerate": 48_000,
            }

    monkeypatch.setattr(audio_devices, "_load_sounddevice", lambda: FakeSoundDevice())

    assert audio_devices.get_default_input_device()["name"] == "Test Microphone"
