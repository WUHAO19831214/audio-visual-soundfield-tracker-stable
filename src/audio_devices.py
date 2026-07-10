"""Friendly sounddevice input-device discovery."""

from __future__ import annotations

import math
from typing import Any

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


PREFERRED_INPUT_KEYWORDS = [
    "Wireless Mic Rx",
    "Wireless",
    "Mic Rx",
    "USB audio CODEC",
    "MacBook Air麦克风",
]


def _default_input_index_from_sounddevice(sd: Any) -> int | None:
    """Resolve sounddevice's [input, output] default pair to an input index."""
    try:
        default_value = sd.default.device
        try:
            candidate = default_value[0]
        except (TypeError, IndexError):
            candidate = default_value
        if candidate is not None and int(candidate) >= 0:
            device = sd.query_devices(int(candidate), "input")
            if int(device["max_input_channels"]) > 0:
                return int(candidate)
    except Exception:
        pass
    return None


def _hostapi_name(sd: Any, hostapi: Any) -> str | int:
    if hostapi is None:
        return "unknown"
    try:
        info = sd.query_hostapis(int(hostapi))
        return str(info.get("name", hostapi))
    except Exception:
        return int(hostapi) if isinstance(hostapi, (int, float)) else str(hostapi)


def list_audio_input_devices(force_refresh: bool = False) -> list[dict]:
    """Return a fresh, fault-tolerant snapshot of input-capable devices.

    ``sounddevice`` does not expose a reliable public device-cache reset. A forced
    refresh therefore deliberately calls ``query_devices()`` again and never keeps
    a module-level cached result; Streamlit owns any short-lived UI cache.
    """
    del force_refresh
    sd = _load_sounddevice()
    try:
        raw_devices = list(sd.query_devices())
    except Exception as exc:
        raise AudioDeviceError(f"无法枚举音频输入设备：{exc}") from exc

    default_index = _default_input_index_from_sounddevice(sd)
    inputs: list[dict] = []
    for index, raw_device in enumerate(raw_devices):
        try:
            channels = int(raw_device["max_input_channels"])
            if channels <= 0:
                continue
            name = str(raw_device["name"])
            sample_rate = float(raw_device["default_samplerate"])
            hostapi = _hostapi_name(sd, raw_device.get("hostapi"))
        except Exception:
            # One malformed PortAudio record must not hide other usable devices.
            continue
        inputs.append(
            {
                "index": int(index),
                "name": name,
                "hostapi": hostapi,
                "max_input_channels": channels,
                "default_samplerate": sample_rate,
                "is_default": index == default_index,
                "channels": channels,
                "samplerate": sample_rate,
                "display_name": (
                    f"{name} | index={index} | {int(sample_rate)} Hz | "
                    f"channels={channels}"
                ),
            }
        )
    return inputs


def get_default_input_device_index() -> int | None:
    """Return a valid input index, resolving default [input, output] pairs."""
    sd = _load_sounddevice()
    index = _default_input_index_from_sounddevice(sd)
    if index is not None:
        return index
    try:
        devices = list_audio_input_devices(force_refresh=True)
    except AudioDeviceError:
        return None
    return int(devices[0]["index"]) if devices else None


def find_preferred_input_device(
    devices: list[dict], preferred_keywords: list[str] | None = None
) -> tuple[dict | None, str]:
    """Choose a stable preferred input device without relying on its name at capture."""
    if not devices:
        return None, "没有可用输入设备"
    keywords = preferred_keywords or PREFERRED_INPUT_KEYWORDS
    normalized = [(device, str(device["name"]).casefold()) for device in devices]

    for device, name in normalized:
        if name == "wireless mic rx":
            return device, "完全匹配 Wireless Mic Rx"
    for device, name in normalized:
        if "wireless" in name or "mic rx" in name:
            return device, "模糊匹配 Wireless / Mic Rx"
    for device, name in normalized:
        if "usb audio codec" in name:
            return device, "未发现 Wireless Mic Rx，匹配 USB audio CODEC"
    for keyword in keywords:
        needle = keyword.casefold()
        for device, name in normalized:
            if needle in name:
                return device, f"匹配偏好关键字 {keyword}"
    for device in devices:
        if device.get("is_default"):
            return device, "使用系统默认输入设备"
    return devices[0], "使用第一个可用输入设备"


def diagnose_audio_devices() -> dict:
    """Collect a non-fatal PortAudio snapshot for the page and CLI diagnostics."""
    try:
        sd = _load_sounddevice()
        raw_devices = list(sd.query_devices())
        devices = list_audio_input_devices(force_refresh=True)
        recommended, recommendation_reason = find_preferred_input_device(devices)
        names = [str(device["name"]).casefold() for device in devices]
        wireless_found = any("wireless mic rx" in name for name in names)
        usb_codec_found = any("usb audio codec" in name for name in names)
        suggestions: list[str] = []
        if not wireless_found:
            suggestions = [
                "点击“刷新麦克风列表”重新枚举设备。",
                "重新插拔 USB 无线麦克风接收器。",
                "在 macOS 声音设置中选择 Wireless Mic Rx 后重启 Streamlit。",
                "如果仍只看到 USB audio CODEC，可尝试选择它；部分接收器会使用该通用名称。",
            ]
        return {
            "ok": True,
            "sounddevice_version": getattr(sd, "__version__", "unknown"),
            "default_device": repr(sd.default.device),
            "default_input_index": get_default_input_device_index(),
            "raw_device_count": len(raw_devices),
            "input_device_count": len(devices),
            "devices": devices,
            "wireless_mic_rx_found": wireless_found,
            "usb_audio_codec_found": usb_codec_found,
            "recommended_device": recommended,
            "recommendation_reason": recommendation_reason,
            "suggestions": suggestions,
            "error": "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "sounddevice_version": "unknown",
            "default_device": "unknown",
            "default_input_index": None,
            "raw_device_count": 0,
            "input_device_count": 0,
            "devices": [],
            "wireless_mic_rx_found": False,
            "usb_audio_codec_found": False,
            "recommended_device": None,
            "recommendation_reason": "",
            "suggestions": [],
            "error": str(exc),
        }


def get_default_input_device() -> dict | None:
    sd = _load_sounddevice()
    try:
        default_index = get_default_input_device_index()
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
            resolved_index = get_default_input_device_index()
        if resolved_index is None:
            raise RuntimeError("没有可用输入设备")
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
