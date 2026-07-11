#!/usr/bin/env python3
"""Exercise one short PortAudio input stream and always close it."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sounddevice as sd

from src.audio_devices import diagnose_audio_devices, test_audio_input_device


def main() -> int:
    diagnostic = diagnose_audio_devices()
    print("Python:", platform.python_version())
    print("sounddevice:", getattr(sd, "__version__", "unknown"))
    try:
        print("PortAudio:", sd.get_portaudio_version())
    except Exception as exc:
        print("PortAudio version unavailable:", exc)
    print("sd.default.device:", diagnostic["default_device"])
    print("\nInput devices:")
    for device in diagnostic["devices"]:
        print("-", device["display_name"], "| hostapi=", device["hostapi"])

    recommended = diagnostic.get("recommended_device")
    if not diagnostic["ok"] or not recommended:
        print("No recommended input device:", diagnostic.get("error", ""))
        return 1

    print("\nTesting for 2 seconds:", recommended["display_name"])
    result = test_audio_input_device(recommended["index"], duration=2.0)
    if not result["ok"]:
        print("FAILED:", result["error"])
        return 1
    print(
        "OK: "
        f"RMS={result['rms']:.6f}, dBFS={result['db']:.1f}, "
        f"peak={result['peak']:.6f}"
    )
    print("InputStream stopped and closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
