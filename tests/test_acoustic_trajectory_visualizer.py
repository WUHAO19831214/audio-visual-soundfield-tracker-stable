from pathlib import Path

import pandas as pd

from src.acoustic_trajectory_visualizer import (
    save_acoustic_trajectory_colormap,
    save_acoustic_trajectory_values,
)


def test_acoustic_trajectory_plots_are_saved(tmp_path: Path) -> None:
    csv_path = tmp_path / "capture_fused.csv"
    pd.DataFrame(
        {
            "timestamp": [0.0, 0.1, 0.2, 0.3],
            "matched": [True, True, True, True],
            "center_x": [20, 60, 90, 120],
            "center_y": [30, 50, 70, 90],
            "db": [-45.0, -35.0, -30.0, -25.0],
            "dominant_frequency_hz": [220.0, 330.0, 440.0, 550.0],
            "spectral_centroid_hz": [300.0, 420.0, 500.0, 640.0],
        }
    ).to_csv(csv_path, index=False)

    values = save_acoustic_trajectory_values(
        csv_path, tmp_path / "db_values.png", metric="db", label_every=2
    )
    colormap = save_acoustic_trajectory_colormap(
        csv_path,
        tmp_path / "centroid_colormap.png",
        metric="spectral_centroid_hz",
    )

    assert values.exists() and values.stat().st_size > 0
    assert colormap.exists() and colormap.stat().st_size > 0


def test_acoustic_trajectory_empty_or_unmatched_does_not_crash(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty_fused.csv"
    pd.DataFrame(
        {
            "timestamp": [0.0, 0.1],
            "matched": [False, False],
            "center_x": ["", ""],
            "center_y": ["", ""],
            "db": [-40.0, -39.0],
        }
    ).to_csv(csv_path, index=False)

    values = save_acoustic_trajectory_values(
        csv_path, tmp_path / "empty_values.png", metric="db"
    )
    colormap = save_acoustic_trajectory_colormap(
        csv_path, tmp_path / "empty_colormap.png", metric="db"
    )

    assert values.exists() and values.stat().st_size > 0
    assert colormap.exists() and colormap.stat().st_size > 0
