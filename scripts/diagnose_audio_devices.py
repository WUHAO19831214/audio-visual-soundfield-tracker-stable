#!/usr/bin/env python3
"""Print a no-file-write diagnostic snapshot of sounddevice input devices."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio_devices import diagnose_audio_devices, test_audio_input_device


def main() -> int:
    diagnostic = diagnose_audio_devices()
    print("Python:", platform.python_version())
    print("sounddevice:", diagnostic["sounddevice_version"])
    print("sd.default.device:", diagnostic["default_device"])
    print("Raw device count:", diagnostic["raw_device_count"])
    print("Input device count:", diagnostic["input_device_count"])
    print("Wireless Mic Rx found:", diagnostic["wireless_mic_rx_found"])
    print("USB audio CODEC found:", diagnostic["usb_audio_codec_found"])
    if not diagnostic["ok"]:
        print("Diagnostic error:", diagnostic["error"])
        return 1

    print("\nInput devices:")
    for device in diagnostic["devices"]:
        marker = " (default)" if device["is_default"] else ""
        print(f"- {device['display_name']} | hostapi={device['hostapi']}{marker}")

    recommended = diagnostic["recommended_device"]
    print("\nRecommended:", recommended["display_name"] if recommended else "none")
    print("Reason:", diagnostic["recommendation_reason"])
    if not recommended:
        return 1

    print("\nTesting recommended input for 1 second...")
    result = test_audio_input_device(recommended["index"], duration=1.0)
    if result["ok"]:
        print(
            "Test OK: "
            f"RMS={result['rms']:.6f}, dBFS={result['db']:.1f}, "
            f"peak={result['peak']:.6f}"
        )
        return 0
    print("Test failed:", result["error"])
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
