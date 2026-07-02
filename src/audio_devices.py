"""Friendly sounddevice input-device discovery."""

from __future__ import annotations

import math

import numpy as np


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
            "channels": int(device["max_input_channels"]),
            "samplerate": float(device["default_samplerate"]),
        }
        for index, device in enumerate(devices)
        if int(device["max_input_channels"]) > 0
    ]


def get_default_input_device_index() -> tuple[int | None, str]:
    """Return the current default input index, never a sounddevice pair."""
    sd = _load_sounddevice()
    try:
        default_value = sd.default.device
        try:
            default_index = default_value[0]
        except (TypeError, IndexError):
            default_index = default_value
        if default_index is not None and int(default_index) >= 0:
            device = sd.query_devices(int(default_index), "input")
            if int(device["max_input_channels"]) > 0:
                return int(default_index), "sounddevice 默认输入设备"
    except Exception:
        default_index = None

    try:
        for index, device in enumerate(sd.query_devices()):
            if int(device["max_input_channels"]) > 0:
                return int(index), "默认输入无效，已回退到第一个可用输入设备"
    except Exception as exc:
        raise AudioDeviceError(f"无法读取默认麦克风：{exc}") from exc
    return None, "没有可用输入设备"


def get_default_input_device() -> dict | None:
    sd = _load_sounddevice()
    try:
        default_index, _ = get_default_input_device_index()
        if default_index is None:
            return None
        device = sd.query_devices(default_index, "input")
        return {
            "index": int(default_index),
            "name": str(device["name"]),
            "max_input_channels": int(device["max_input_channels"]),
            "default_samplerate": float(device["default_samplerate"]),
            "channels": int(device["max_input_channels"]),
            "samplerate": float(device["default_samplerate"]),
        }
    except Exception as exc:
        raise AudioDeviceError(f"无法读取默认麦克风：{exc}") from exc


def check_audio_input_device(device_index: int | None = None) -> dict:
    sd = _load_sounddevice()
    try:
        resolved_index = device_index
        if resolved_index is None:
            resolved_index, _ = get_default_input_device_index()
        device = sd.query_devices(resolved_index, "input")
        sample_rate = int(float(device["default_samplerate"]) or 48_000)
        channels = min(1, int(device["max_input_channels"]))
        sd.check_input_settings(device=resolved_index, channels=channels, samplerate=sample_rate)
        return {
            "index": resolved_index,
            "name": str(device["name"]),
            "sample_rate": sample_rate,
            "samplerate": sample_rate,
            "channels": channels,
        }
    except Exception as exc:
        label = "默认麦克风" if device_index is None else f"麦克风 index={device_index}"
        raise AudioDeviceError(f"{label}不可用：{exc}") from exc


def _audio_stats(samples: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if values.size == 0:
        rms = 0.0
        peak = 0.0
    else:
        rms = float(np.sqrt(np.mean(values**2)))
        peak = float(np.max(np.abs(values)))
    dbfs = 20.0 * math.log10(max(rms, 1e-12))
    meter = min(1.0, max(0.0, (dbfs + 80.0) / 60.0))
    return rms, dbfs, peak, meter


def test_audio_input_device(
    device_index: int | None,
    duration: float = 1.0,
    sample_rate: int | None = None,
) -> dict:
    """Try recording from one input device and return diagnostics instead of raising."""
    try:
        sd = _load_sounddevice()
        device = check_audio_input_device(device_index)
        rate = int(sample_rate or device["sample_rate"])
        frame_count = max(1, int(rate * duration))
        recording = sd.rec(
            frame_count,
            samplerate=rate,
            channels=1,
            dtype="float32",
            device=device["index"],
        )
        sd.wait()
        rms, dbfs, peak, _ = _audio_stats(recording)
        return {
            "ok": True,
            "device_index": device["index"],
            "device_name": device["name"],
            "sample_rate": rate,
            "channels": device["channels"],
            "rms": rms,
            "db": dbfs,
            "peak": peak,
            "error": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "device_index": device_index,
            "device_name": "",
            "sample_rate": sample_rate or 0,
            "channels": 0,
            "rms": 0.0,
            "db": -120.0,
            "peak": 0.0,
            "error": str(exc),
        }


def measure_audio_input_level(
    device_index: int | None = None, duration_sec: float = 0.5
) -> dict:
    """Record a short block and return a simple input-level summary."""
    sd = _load_sounddevice()
    device = check_audio_input_device(device_index)
    sample_rate = int(device["sample_rate"])
    frame_count = max(1, int(sample_rate * duration_sec))
    try:
        recording = sd.rec(
            frame_count,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device_index,
        )
        sd.wait()
    except Exception as exc:
        label = "默认麦克风" if device_index is None else f"麦克风 index={device_index}"
        raise AudioDeviceError(f"{label}输入电平读取失败：{exc}") from exc

    rms, dbfs, peak, meter = _audio_stats(recording)
    return {
        "name": device["name"],
        "sample_rate": sample_rate,
        "rms": rms,
        "peak": peak,
        "dbfs": dbfs,
        "meter": meter,
    }
