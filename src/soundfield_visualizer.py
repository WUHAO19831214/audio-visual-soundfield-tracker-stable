"""Visualizations generated from a fused audio-visual CSV file."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("data/output")


def _load_matched(csv_path: str | Path, value_column: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if "matched" in frame:
        matched = frame["matched"].astype(str).str.lower().isin(("true", "1", "yes"))
        frame = frame[matched]

    numeric_columns = ["center_x", "center_y"]
    if value_column:
        numeric_columns.append(value_column)
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=numeric_columns)


def _output_path(csv_path: str | Path, suffix: str) -> Path:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR / f"{Path(csv_path).stem}_{suffix}.png"


def _style_axes(axis: plt.Axes) -> None:
    axis.set_xlabel("Image X (pixels)")
    axis.set_ylabel("Image Y (pixels)")
    axis.invert_yaxis()
    axis.grid(alpha=0.2)
    axis.set_aspect("equal", adjustable="datalim")


def _tracking_subject(frame: pd.DataFrame) -> str:
    if "tracking_mode" not in frame.columns:
        return "person"
    modes = frame["tracking_mode"].astype(str)
    if modes.eq("tennis_ball_color").any():
        return "tennis"
    if modes.eq("custom_object_template").any():
        return "custom"
    return "person"


def _tracking_subject_csv(csv_path: str | Path) -> str:
    try:
        frame = pd.read_csv(csv_path, usecols=["tracking_mode"])
    except (ValueError, OSError):
        return "person"
    return _tracking_subject(frame)


def plot_trajectory(csv_path: str | Path) -> Path:
    """Plot center_x/center_y movement and save it under data/output/."""
    frame = _load_matched(csv_path)
    subject = _tracking_subject_csv(csv_path)
    output = _output_path(csv_path, "trajectory")
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    if frame.empty:
        axis.text(0.5, 0.5, "No matched trajectory data", ha="center", va="center")
    else:
        timestamps = pd.to_numeric(frame["timestamp"], errors="coerce")
        if timestamps.notna().all():
            elapsed = timestamps.to_numpy() - float(timestamps.iloc[0])
        else:
            elapsed = np.arange(len(frame), dtype=float)
        axis.plot(frame["center_x"], frame["center_y"], color="#1565c0", alpha=0.8)
        rendered = axis.scatter(
            frame["center_x"],
            frame["center_y"],
            c=elapsed,
            cmap="viridis",
            s=22,
        )
        figure.colorbar(rendered, ax=axis, label="Elapsed time (seconds)")
        axis.scatter(frame.iloc[0]["center_x"], frame.iloc[0]["center_y"], c="green", s=80, label="Start")
        axis.scatter(frame.iloc[-1]["center_x"], frame.iloc[-1]["center_y"], c="red", s=80, label="End")
        legend_title = {
            "tennis": "Tennis ball marker track",
            "custom": "Custom object track",
        }.get(subject)
        axis.legend(title=legend_title)
    axis.set_title({
        "tennis": "Tennis Ball Marker Trajectory",
        "custom": "Custom Object Trajectory",
        "person": "Tracked Person Trajectory",
    }[subject])
    _style_axes(axis)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def _plot_spatial_values(
    csv_path: str | Path,
    value_column: str,
    title: str,
    colorbar_label: str,
    suffix: str,
    cmap: str,
) -> Path:
    frame = _load_matched(csv_path, value_column)
    subject = _tracking_subject_csv(csv_path)
    output = _output_path(csv_path, suffix)
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)

    if frame.empty:
        axis.text(0.5, 0.5, "No matched spatial data", ha="center", va="center")
    else:
        x = frame["center_x"].to_numpy()
        y = frame["center_y"].to_numpy()
        values = frame[value_column].to_numpy()
        rendered = None

        points = np.column_stack((x, y))
        centered = points - points.mean(axis=0)
        if len(frame) >= 4 and np.linalg.matrix_rank(centered) >= 2:
            try:
                rendered = axis.tricontourf(x, y, values, levels=16, cmap=cmap)
            except (RuntimeError, ValueError):
                rendered = None
        if rendered is None:
            rendered = axis.scatter(x, y, c=values, cmap=cmap, s=55, alpha=0.9)
        else:
            axis.scatter(x, y, c=values, cmap=cmap, s=12, edgecolors="none")
        figure.colorbar(rendered, ax=axis, label=colorbar_label)

    subject_label = {
        "tennis": "Tennis Ball Marker",
        "custom": "Custom Object",
        "person": "Tracked Person",
    }[subject]
    axis.set_title(f"{subject_label} | {title}")
    _style_axes(axis)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def plot_sound_intensity_map(csv_path: str | Path) -> Path:
    """Plot relative sound intensity in image coordinates."""
    return _plot_spatial_values(
        csv_path,
        value_column="db",
        title="Spatial Relative Sound Intensity",
        colorbar_label="Relative level (dBFS)",
        suffix="sound_intensity",
        cmap="inferno",
    )


def plot_frequency_map(csv_path: str | Path) -> Path:
    """Plot dominant frequency in image coordinates."""
    return _plot_spatial_values(
        csv_path,
        value_column="dominant_frequency_hz",
        title="Spatial Dominant Frequency",
        colorbar_label="Dominant frequency (Hz)",
        suffix="dominant_frequency",
        cmap="turbo",
    )
