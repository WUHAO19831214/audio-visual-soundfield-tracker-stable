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

    index = audio_devices.get_default_input_device_index()

    assert index == 4


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
            "hostapi": "unknown",
            "is_default": False,
            "channels": 1,
            "samplerate": 48_000.0,
            "display_name": "Wireless Mic Rx | index=1 | 48000 Hz | channels=1",
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


def test_preferred_device_chooses_wireless_mic_rx() -> None:
    devices = [
        {
            "index": 3,
            "name": "USB audio CODEC",
            "is_default": True,
        },
        {
            "index": 4,
            "name": "Wireless Mic Rx",
            "is_default": False,
        },
    ]

    device, reason = audio_devices.find_preferred_input_device(devices)

    assert device["index"] == 4
    assert "Wireless Mic Rx" in reason


def test_preferred_device_falls_back_to_usb_audio_codec() -> None:
    devices = [
        {"index": 1, "name": "MacBook Air麦克风", "is_default": False},
        {"index": 3, "name": "USB audio CODEC", "is_default": True},
    ]

    device, reason = audio_devices.find_preferred_input_device(devices)

    assert device["index"] == 3
    assert "USB audio CODEC" in reason


def test_device_enumeration_skips_one_malformed_device(monkeypatch) -> None:
    class Default:
        device = [2, 6]

    class FakeSoundDevice:
        default = Default()

        @staticmethod
        def query_devices(index=None, kind=None):
            if index is not None:
                return {
                    "name": "Wireless Mic Rx",
                    "max_input_channels": 1,
                    "default_samplerate": 48_000,
                }
            return [
                {"name": "Bad", "max_input_channels": "not-a-number"},
                {"name": "Output", "max_input_channels": 0, "default_samplerate": 48_000},
                {
                    "name": "Wireless Mic Rx",
                    "max_input_channels": 1,
                    "default_samplerate": 48_000,
                    "hostapi": 0,
                },
            ]

        @staticmethod
        def query_hostapis(index):
            return {"name": "Core Audio"}

    monkeypatch.setattr(audio_devices, "_load_sounddevice", lambda: FakeSoundDevice())

    devices = audio_devices.list_audio_input_devices(force_refresh=True)

    assert len(devices) == 1
    assert devices[0]["index"] == 2
    assert devices[0]["hostapi"] == "Core Audio"
