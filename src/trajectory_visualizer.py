"""Trajectory drawing helpers for tracked visual targets."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np


Point = dict[str, float | int | str]
TrackHistory = Mapping[int | str, list[Mapping]]


def _track_sort_key(track_id: int | str) -> tuple[int, str]:
    try:
        return (0, f"{int(track_id):08d}")
    except (TypeError, ValueError):
        return (1, str(track_id))


def _track_color(track_id: int | str) -> tuple[int, int, int]:
    seed = abs(hash(str(track_id)))
    return (
        80 + seed % 150,
        80 + (seed // 7) % 150,
        80 + (seed // 17) % 150,
    )


def _as_number(value) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_points(
    points: list[Mapping], current_frame_index: int | None = None
) -> list[Point]:
    normalized: list[Point] = []
    for point in points:
        x = _as_number(point.get("center_x"))
        y = _as_number(point.get("center_y"))
        if x is None or y is None:
            continue
        frame_index = point.get("frame_index")
        if current_frame_index is not None:
            frame_number = _as_number(frame_index)
            if frame_number is not None and frame_number > current_frame_index:
                continue
        normalized.append(
            {
                "center_x": x,
                "center_y": y,
                "frame_index": frame_index if frame_index != "" else len(normalized),
                "timestamp": point.get("timestamp", ""),
            }
        )

    def sort_key(point: Point) -> tuple[float, float]:
        frame_number = _as_number(point.get("frame_index"))
        timestamp = _as_number(point.get("timestamp"))
        return (
            frame_number if frame_number is not None else float("inf"),
            timestamp if timestamp is not None else float("inf"),
        )

    return sorted(normalized, key=sort_key)


def _selected_track_ids(
    track_history: TrackHistory,
    main_track_id: int | str | None,
    show_all_tracks: bool,
) -> list[int | str]:
    if show_all_tracks:
        return sorted(track_history.keys(), key=_track_sort_key)
    if main_track_id is not None and main_track_id in track_history:
        return [main_track_id]
    chosen = choose_main_track(track_history)
    return [] if chosen is None else [chosen]


def choose_main_track(track_history: TrackHistory) -> int | str | None:
    """Pick the track with the most valid trajectory points."""
    best_track_id = None
    best_count = -1
    for track_id in sorted(track_history.keys(), key=_track_sort_key):
        count = len(_normalized_points(list(track_history[track_id])))
        if count > best_count:
            best_track_id = track_id
            best_count = count
    return best_track_id


def draw_tracks_on_frame(
    frame,
    track_history: TrackHistory,
    current_frame_index: int | None = None,
    main_track_id: int | str | None = None,
    show_all_tracks: bool = True,
):
    """Draw track polylines onto a BGR frame and return a copy."""
    canvas = frame.copy()
    if canvas.size == 0:
        return canvas

    selected = _selected_track_ids(track_history, main_track_id, show_all_tracks)
    for track_id in selected:
        points = _normalized_points(list(track_history.get(track_id, [])), current_frame_index)
        if not points:
            continue
        xy = np.array(
            [(round(point["center_x"]), round(point["center_y"])) for point in points],
            dtype=np.int32,
        )
        is_main = main_track_id is not None and str(track_id) == str(main_track_id)
        color = (0, 0, 255) if is_main else _track_color(track_id)
        thickness = 3 if is_main else 2
        if len(xy) >= 2:
            cv2.polylines(canvas, [xy.reshape((-1, 1, 2))], False, color, thickness)
        for point in xy[-20:]:
            cv2.circle(canvas, tuple(point), 4 if is_main else 3, color, -1)
        cv2.circle(canvas, tuple(xy[0]), 7, (0, 180, 0), -1)
        cv2.circle(canvas, tuple(xy[-1]), 7, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            f"track {track_id}",
            (int(xy[-1][0]) + 8, int(xy[-1][1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def _infer_image_size(track_history: TrackHistory) -> tuple[int, int]:
    xs: list[float] = []
    ys: list[float] = []
    for points in track_history.values():
        for point in _normalized_points(list(points)):
            xs.append(float(point["center_x"]))
            ys.append(float(point["center_y"]))
    if not xs or not ys:
        return (1280, 720)
    width = max(320, int(max(xs) + 80))
    height = max(240, int(max(ys) + 80))
    return (width, height)


def _blank_canvas(image_size: tuple[int, int] | tuple[int, int, int] | None):
    if image_size is None:
        width, height = 1280, 720
    elif len(image_size) >= 3:
        height, width = int(image_size[0]), int(image_size[1])
    else:
        width, height = int(image_size[0]), int(image_size[1])
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    return canvas


def save_trajectory_overlay(
    background_frame,
    track_history: TrackHistory,
    output_path: str | Path,
    main_track_id: int | str | None = None,
    show_all_tracks: bool = True,
) -> Path:
    """Save trajectories over a supplied BGR background frame."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = draw_tracks_on_frame(
        background_frame,
        track_history,
        main_track_id=main_track_id,
        show_all_tracks=show_all_tracks,
    )
    cv2.imwrite(str(output), rendered)
    return output


def save_trajectory_blank(
    track_history: TrackHistory,
    output_path: str | Path,
    image_size,
    main_track_id: int | str | None = None,
    show_all_tracks: bool = True,
) -> Path:
    """Save trajectories on a blank image."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    size = image_size or _infer_image_size(track_history)
    rendered = draw_tracks_on_frame(
        _blank_canvas(size),
        track_history,
        main_track_id=main_track_id,
        show_all_tracks=show_all_tracks,
    )
    cv2.imwrite(str(output), rendered)
    return output


def build_track_history_from_csv(csv_path: str | Path) -> dict[int | str, list[Point]]:
    """Build a track history from a trajectory or fused CSV file."""
    history: dict[int | str, list[Point]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            matched = str(row.get("matched", "true")).lower()
            if matched not in ("true", "1", "yes", ""):
                continue
            track_id = row.get("track_id", "")
            x = _as_number(row.get("center_x"))
            y = _as_number(row.get("center_y"))
            if track_id == "" or x is None or y is None:
                continue
            try:
                normalized_id: int | str = int(float(track_id))
            except (TypeError, ValueError):
                normalized_id = track_id
            history.setdefault(normalized_id, []).append(
                {
                    "frame_index": row.get("frame_index", index),
                    "timestamp": row.get("timestamp", ""),
                    "center_x": x,
                    "center_y": y,
                    "bbox_width": row.get("bbox_width", ""),
                    "bbox_height": row.get("bbox_height", ""),
                }
            )
    return history
