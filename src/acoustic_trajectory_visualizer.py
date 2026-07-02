"""Acoustic values plotted along visual trajectories."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd


METRIC_LABELS = {
    "db": ("Relative level", "dBFS", "inferno"),
    "dominant_frequency_hz": ("Dominant frequency", "Hz", "turbo"),
    "spectral_centroid_hz": ("Spectral centroid", "Hz", "viridis"),
}


def _metric_info(metric: str) -> tuple[str, str, str]:
    return METRIC_LABELS.get(metric, (metric, "", "viridis"))


def _load_points(fused_csv_path: str | Path, metric: str) -> pd.DataFrame:
    frame = pd.read_csv(fused_csv_path)
    if "matched" in frame:
        matched = frame["matched"].astype(str).str.lower().isin(("true", "1", "yes"))
        frame = frame[matched]
    columns = ["center_x", "center_y", metric]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        return pd.DataFrame(columns=columns)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=columns).reset_index(drop=True)


def _setup_axis(axis: plt.Axes, background_image_path: str | Path | None, frame: pd.DataFrame) -> None:
    if background_image_path and Path(background_image_path).exists():
        image = mpimg.imread(background_image_path)
        axis.imshow(image)
        axis.set_xlim(0, image.shape[1])
        axis.set_ylim(image.shape[0], 0)
    elif not frame.empty:
        x_margin = max(40.0, (frame["center_x"].max() - frame["center_x"].min()) * 0.12)
        y_margin = max(40.0, (frame["center_y"].max() - frame["center_y"].min()) * 0.12)
        axis.set_xlim(frame["center_x"].min() - x_margin, frame["center_x"].max() + x_margin)
        axis.set_ylim(frame["center_y"].max() + y_margin, frame["center_y"].min() - y_margin)
    axis.set_xlabel("Image X (pixels)")
    axis.set_ylabel("Image Y (pixels)")
    axis.grid(alpha=0.2)


def _save_placeholder(output_path: str | Path, metric: str, message: str) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    title, unit, _ = _metric_info(metric)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.text(0.5, 0.5, message, ha="center", va="center")
    axis.set_title(f"{title} trajectory" + (f" ({unit})" if unit else ""))
    axis.set_axis_off()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def save_acoustic_trajectory_values(
    fused_csv_path: str | Path,
    output_path: str | Path,
    metric: str = "db",
    label_every: int = 10,
    background_image_path: str | Path | None = None,
) -> Path:
    """Save a trajectory plot with numeric acoustic labels."""
    frame = _load_points(fused_csv_path, metric)
    if frame.empty:
        return _save_placeholder(output_path, metric, "No matched acoustic trajectory data")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    title, unit, _ = _metric_info(metric)
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    _setup_axis(axis, background_image_path, frame)
    axis.plot(frame["center_x"], frame["center_y"], color="#1565c0", alpha=0.65)
    axis.scatter(frame["center_x"], frame["center_y"], color="#1565c0", s=22)
    interval = max(1, int(label_every))
    for index, row in frame.iterrows():
        if index % interval != 0 and index not in (0, len(frame) - 1):
            continue
        value = f"{row[metric]:.1f}" if unit else str(row[metric])
        axis.annotate(
            value,
            (row["center_x"], row["center_y"]),
            textcoords="offset points",
            xytext=(5, -8),
            fontsize=8,
            color="#111111",
        )
    axis.set_title(f"{title} values along visual trajectory" + (f" ({unit})" if unit else ""))
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def save_acoustic_trajectory_colormap(
    fused_csv_path: str | Path,
    output_path: str | Path,
    metric: str = "db",
    background_image_path: str | Path | None = None,
) -> Path:
    """Save a trajectory plot using color to encode one acoustic metric."""
    frame = _load_points(fused_csv_path, metric)
    if frame.empty:
        return _save_placeholder(output_path, metric, "No matched acoustic trajectory data")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    title, unit, cmap = _metric_info(metric)
    figure, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    _setup_axis(axis, background_image_path, frame)
    axis.plot(frame["center_x"], frame["center_y"], color="#777777", alpha=0.35)
    rendered = axis.scatter(
        frame["center_x"],
        frame["center_y"],
        c=frame[metric],
        cmap=cmap,
        s=54,
        edgecolors="white",
        linewidths=0.4,
    )
    figure.colorbar(rendered, ax=axis, label=f"{title}" + (f" ({unit})" if unit else ""))
    axis.set_title(f"{title} colormap along visual trajectory")
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
