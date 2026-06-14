"""Friendly sounddevice input-device discovery."""

from __future__ import annotations


class AudioDeviceError(RuntimeError):
    pass


def _load_sounddevice():
    try:
        import sounddevice as sd

        return sd
    except Exception as exc:
        raise AudioDeviceError(
            "无法加载 sounddevice。请激活 .venv 后执行 pip install sounddevice。"
        ) from exc


def list_audio_input_devices() -> list[dict]:
    """Return all devices with at least one input channel."""
    sd = _load_sounddevice()
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise AudioDeviceError(f"无法枚举音频输入设备：{exc}") from exc

    return [
        {
            "index": index,
            "name": str(device["name"]),
            "max_input_channels": int(device["max_input_channels"]),
            "default_samplerate": float(device["default_samplerate"]),
        }
        for index, device in enumerate(devices)
        if int(device["max_input_channels"]) > 0
    ]


def get_default_input_device() -> dict | None:
    sd = _load_sounddevice()
    try:
        default_value = sd.default.device
        try:
            default_index = default_value[0]
        except (TypeError, IndexError):
            default_index = default_value
        if default_index is None or int(default_index) < 0:
            return None
        device = sd.query_devices(int(default_index), "input")
        return {
            "index": int(default_index),
            "name": str(device["name"]),
            "max_input_channels": int(device["max_input_channels"]),
            "default_samplerate": float(device["default_samplerate"]),
        }
    except Exception as exc:
        raise AudioDeviceError(f"无法读取默认麦克风：{exc}") from exc


def check_audio_input_device(device_index: int | None = None) -> dict:
    sd = _load_sounddevice()
    try:
        device = sd.query_devices(device_index, "input")
        sample_rate = int(float(device["default_samplerate"]) or 48_000)
        sd.check_input_settings(device=device_index, channels=1, samplerate=sample_rate)
        return {
            "index": device_index,
            "name": str(device["name"]),
            "sample_rate": sample_rate,
        }
    except Exception as exc:
        label = "默认麦克风" if device_index is None else f"麦克风 index={device_index}"
        raise AudioDeviceError(f"{label}不可用：{exc}") from exc
