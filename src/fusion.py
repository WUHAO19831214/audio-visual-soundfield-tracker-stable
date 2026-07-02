"""Timestamp alignment and thread-safe synchronized capture state."""

from __future__ import annotations

import bisect
import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping

from .audio_recorder import AudioRecorder
from .csv_logger import write_csv
from .sync_clock import SyncClock


FUSED_FIELDS = [
    "timestamp",
    "matched",
    "time_diff_sec",
    "track_id",
    "center_x",
    "center_y",
    "bbox_width",
    "bbox_height",
    "rms",
    "db",
    "dominant_frequency_hz",
    "spectral_centroid_hz",
    "zero_crossing_rate",
]

VISUAL_FIELDS = [
    "timestamp",
    "track_id",
    "center_x",
    "center_y",
    "bbox_width",
    "bbox_height",
]

AUDIO_FIELDS = [
    "timestamp",
    "rms",
    "db",
    "dominant_frequency_hz",
    "spectral_centroid_hz",
    "zero_crossing_rate",
]


def fuse_records(
    visual_records: Iterable[Mapping],
    audio_records: Iterable[Mapping],
    max_time_diff_sec: float = 0.15,
) -> list[dict]:
    """Match every audio row to the nearest visual row.

    Audio timestamps define the output cadence. Audio rows without a visual
    sample within ``max_time_diff_sec`` remain in the result as unmatched.
    """
    visuals = sorted(
        (dict(row) for row in visual_records), key=lambda row: float(row["timestamp"])
    )
    audio = sorted(
        (dict(row) for row in audio_records), key=lambda row: float(row["timestamp"])
    )
    visual_times = [float(row["timestamp"]) for row in visuals]
    fused: list[dict] = []

    for audio_row in audio:
        timestamp = float(audio_row["timestamp"])
        nearest = None
        time_diff = None

        if visual_times:
            position = bisect.bisect_left(visual_times, timestamp)
            candidate_indices = [
                index
                for index in (position - 1, position)
                if 0 <= index < len(visuals)
            ]
            nearest_index = min(
                candidate_indices,
                key=lambda index: abs(visual_times[index] - timestamp),
            )
            nearest = visuals[nearest_index]
            time_diff = abs(float(nearest["timestamp"]) - timestamp)

        matched = nearest is not None and time_diff is not None and time_diff <= max_time_diff_sec
        row = {
            "timestamp": timestamp,
            "matched": matched,
            "time_diff_sec": round(time_diff, 6) if time_diff is not None else "",
            "track_id": nearest.get("track_id", "") if matched else "",
            "center_x": nearest.get("center_x", "") if matched else "",
            "center_y": nearest.get("center_y", "") if matched else "",
            "bbox_width": nearest.get("bbox_width", "") if matched else "",
            "bbox_height": nearest.get("bbox_height", "") if matched else "",
            "rms": audio_row.get("rms", ""),
            "db": audio_row.get("db", ""),
            "dominant_frequency_hz": audio_row.get("dominant_frequency_hz", ""),
            "spectral_centroid_hz": audio_row.get("spectral_centroid_hz", ""),
            "zero_crossing_rate": audio_row.get("zero_crossing_rate", ""),
        }
        fused.append(row)

    return fused


@dataclass(frozen=True)
class CaptureArtifacts:
    fused_csv: Path
    visual_csv: Path
    audio_csv: Path
    trajectory_plot: Path | None
    intensity_plot: Path | None
    frequency_plot: Path | None
    trajectory_overlay_plot: Path | None = None
    acoustic_db_values_plot: Path | None = None
    acoustic_db_colormap_plot: Path | None = None
    acoustic_frequency_values_plot: Path | None = None
    acoustic_frequency_colormap_plot: Path | None = None
    acoustic_spectral_centroid_colormap_plot: Path | None = None
    acoustic_video: Path | None = None
    summary_json: Path | None = None
    summary: dict | None = None


class SynchronizedCaptureSession:
    """Shared state written by independent audio and video processors."""

    def __init__(
        self,
        output_dir: str | Path = "data/output",
        audio_block_duration_sec: float = 0.1,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.clock = SyncClock()
        self._lock = threading.RLock()
        self._visual_records: list[dict] = []
        self._audio_records: list[dict] = []
        self._latest_visual: dict | None = None
        self._latest_audio: dict | None = None
        self._audio_device_info: dict = {}
        self._audio_error = ""
        self.finalized = False
        self.artifacts: CaptureArtifacts | None = None
        self.audio_recorder = AudioRecorder(
            clock=self.clock,
            block_duration_sec=audio_block_duration_sec,
            on_features=self.add_audio_record,
        )

    def add_visual_track(self, timestamp: float, track: Mapping | None) -> None:
        with self._lock:
            if track is None:
                self._latest_visual = None
                return
            row = {
                "timestamp": float(timestamp),
                "track_id": track.get("track_id", ""),
                "center_x": track.get("center_x", ""),
                "center_y": track.get("center_y", ""),
                "bbox_width": track.get("bbox_width", ""),
                "bbox_height": track.get("bbox_height", ""),
            }
            self._visual_records.append(row)
            self._latest_visual = row

    def add_audio_record(self, row: Mapping) -> None:
        with self._lock:
            normalized = {field: row.get(field, "") for field in AUDIO_FIELDS}
            self._audio_records.append(normalized)
            self._latest_audio = normalized

    def set_audio_device_info(self, info: Mapping) -> None:
        with self._lock:
            self._audio_device_info = dict(info)

    def set_audio_error(self, error: str) -> None:
        with self._lock:
            self._audio_error = str(error)

    def visual_snapshot(self) -> list[dict]:
        with self._lock:
            return [row.copy() for row in self._visual_records]

    def audio_snapshot(self) -> list[dict]:
        with self._lock:
            return [row.copy() for row in self._audio_records]

    def status(self) -> dict:
        with self._lock:
            return {
                "visual": self._latest_visual.copy() if self._latest_visual else None,
                "audio": self._latest_audio.copy() if self._latest_audio else None,
                "visual_samples": len(self._visual_records),
                "audio_samples": len(self._audio_records),
                "elapsed_sec": self.clock.elapsed(),
                "audio_device": self._audio_device_info.copy(),
                "audio_error": self._audio_error,
            }

    def finalize(
        self,
        prefix: str | None = None,
        export_acoustic_trajectory: bool = True,
        acoustic_label_every: int = 10,
    ) -> CaptureArtifacts:
        """Freeze current data, export raw/fused CSV files, and make plots."""
        with self._lock:
            if self.artifacts is not None:
                return self.artifacts
            visual = [row.copy() for row in self._visual_records]
            audio = [row.copy() for row in self._audio_records]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = prefix or datetime.now().strftime("capture_%Y%m%d_%H%M%S")
        visual_path = write_csv(
            self.output_dir / f"{stem}_visual.csv", VISUAL_FIELDS, visual
        )
        audio_path = write_csv(
            self.output_dir / f"{stem}_audio.csv", AUDIO_FIELDS, audio
        )
        fused = fuse_records(visual, audio)
        fused_path = write_csv(
            self.output_dir / f"{stem}_fused.csv",
            FUSED_FIELDS,
            fused,
        )
        with self._lock:
            audio_device_info = self._audio_device_info.copy()
            audio_error = self._audio_error

        from .soundfield_visualizer import (
            plot_frequency_map,
            plot_sound_intensity_map,
            plot_trajectory,
        )

        trajectory = plot_trajectory(fused_path)
        intensity = plot_sound_intensity_map(fused_path)
        frequency = plot_frequency_map(fused_path)
        trajectory_overlay = None
        acoustic_db_values = None
        acoustic_db_colormap = None
        acoustic_frequency_values = None
        acoustic_frequency_colormap = None
        acoustic_spectral_centroid_colormap = None

        if export_acoustic_trajectory:
            from .acoustic_trajectory_visualizer import (
                save_acoustic_trajectory_colormap,
                save_acoustic_trajectory_values,
            )
            from .trajectory_visualizer import (
                build_track_history_from_csv,
                choose_main_track,
                save_trajectory_blank,
            )

            track_history = build_track_history_from_csv(fused_path)
            main_track_id = choose_main_track(track_history)
            trajectory_overlay = save_trajectory_blank(
                track_history,
                self.output_dir / f"{stem}_fused_trajectory_overlay.png",
                image_size=None,
                main_track_id=main_track_id,
                show_all_tracks=True,
            )
            acoustic_db_values = save_acoustic_trajectory_values(
                fused_path,
                self.output_dir / f"{stem}_acoustic_trajectory_db_values.png",
                metric="db",
                label_every=acoustic_label_every,
            )
            acoustic_db_colormap = save_acoustic_trajectory_colormap(
                fused_path,
                self.output_dir / f"{stem}_acoustic_trajectory_db_colormap.png",
                metric="db",
            )
            acoustic_frequency_values = save_acoustic_trajectory_values(
                fused_path,
                self.output_dir / f"{stem}_acoustic_trajectory_frequency_values.png",
                metric="dominant_frequency_hz",
                label_every=acoustic_label_every,
            )
            acoustic_frequency_colormap = save_acoustic_trajectory_colormap(
                fused_path,
                self.output_dir / f"{stem}_acoustic_trajectory_frequency_colormap.png",
                metric="dominant_frequency_hz",
            )
            acoustic_spectral_centroid_colormap = save_acoustic_trajectory_colormap(
                fused_path,
                self.output_dir
                / f"{stem}_acoustic_trajectory_spectral_centroid_colormap.png",
                metric="spectral_centroid_hz",
            )

        summary = {
            "ok": bool(visual and audio and fused),
            "warning": "" if visual and audio and fused else "采集样本不足，结果不可信",
            "video_sample_count": len(visual),
            "audio_sample_count": len(audio),
            "fused_sample_count": len(fused),
            "audio_device_index": audio_device_info.get("index", ""),
            "audio_device_name": audio_device_info.get("name", ""),
            "audio_sample_rate": audio_device_info.get("sample_rate", ""),
            "audio_started_at": audio_device_info.get("started_at", ""),
            "audio_last_chunk_at": audio_device_info.get("last_chunk_at", ""),
            "audio_error": audio_error or audio_device_info.get("last_error", ""),
        }
        summary_path = self.output_dir / f"{stem}_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        artifacts = CaptureArtifacts(
            fused_csv=fused_path,
            visual_csv=visual_path,
            audio_csv=audio_path,
            trajectory_plot=trajectory,
            intensity_plot=intensity,
            frequency_plot=frequency,
            trajectory_overlay_plot=trajectory_overlay,
            acoustic_db_values_plot=acoustic_db_values,
            acoustic_db_colormap_plot=acoustic_db_colormap,
            acoustic_frequency_values_plot=acoustic_frequency_values,
            acoustic_frequency_colormap_plot=acoustic_frequency_colormap,
            acoustic_spectral_centroid_colormap_plot=acoustic_spectral_centroid_colormap,
            summary_json=summary_path,
            summary=summary,
        )
        with self._lock:
            self.artifacts = artifacts
            self.finalized = True
        return artifacts
