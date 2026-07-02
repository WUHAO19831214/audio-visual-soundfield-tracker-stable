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


def test_default_input_index_uses_input_side_of_pair(monkeypatch) -> None:
    class Default:
        device = [4, 9]

    class FakeSoundDevice:
        default = Default()

        @staticmethod
        def query_devices(index=None, kind=None):
            assert index == 4
            assert kind == "input"
            return {
                "name": "Pair Input",
                "max_input_channels": 2,
                "default_samplerate": 48_000,
            }

    monkeypatch.setattr(audio_devices, "_load_sounddevice", lambda: FakeSoundDevice())

    index, note = audio_devices.get_default_input_device_index()

    assert index == 4
    assert "默认输入" in note


def test_audio_device_list_only_returns_inputs(monkeypatch) -> None:
    class FakeSoundDevice:
        @staticmethod
        def query_devices(*args, **kwargs):
            return [
                {"name": "Output Only", "max_input_channels": 0, "default_samplerate": 48_000},
                {"name": "Wireless Mic Rx", "max_input_channels": 1, "default_samplerate": 48_000},
            ]

    monkeypatch.setattr(audio_devices, "_load_sounddevice", lambda: FakeSoundDevice())

    devices = audio_devices.list_audio_input_devices()

    assert devices == [
        {
            "index": 1,
            "name": "Wireless Mic Rx",
            "max_input_channels": 1,
            "default_samplerate": 48_000.0,
            "channels": 1,
            "samplerate": 48_000.0,
        }
    ]


def test_audio_input_test_returns_error_instead_of_raising(monkeypatch) -> None:
    class Default:
        device = [0, 1]

    class FakeSoundDevice:
        default = Default()

        @staticmethod
        def query_devices(index=None, kind=None):
            return {
                "name": "Broken Mic",
                "max_input_channels": 1,
                "default_samplerate": 48_000,
            }

        @staticmethod
        def check_input_settings(**kwargs):
            pass

        @staticmethod
        def rec(*args, **kwargs):
            raise RuntimeError("input busy")

    monkeypatch.setattr(audio_devices, "_load_sounddevice", lambda: FakeSoundDevice())

    result = audio_devices.test_audio_input_device(0, duration=0.1)

    assert result["ok"] is False
    assert "input busy" in result["error"]
