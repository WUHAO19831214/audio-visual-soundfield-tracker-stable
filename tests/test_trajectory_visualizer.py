from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.trajectory_visualizer import (
    build_track_history_from_csv,
    choose_main_track,
    draw_tracks_on_frame,
    save_trajectory_blank,
    save_trajectory_overlay,
)


def test_track_history_from_csv_skips_unmatched_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "tracks.csv"
    pd.DataFrame(
        {
            "timestamp": [0.0, 0.1, 0.2],
            "matched": [True, False, True],
            "track_id": [1, 1, 2],
            "center_x": [20, 40, 80],
            "center_y": [30, 60, 90],
        }
    ).to_csv(csv_path, index=False)

    history = build_track_history_from_csv(csv_path)

    assert sorted(history) == [1, 2]
    assert len(history[1]) == 1
    assert choose_main_track(history) == 1


def test_trajectory_outputs_are_saved(tmp_path: Path) -> None:
    history = {
        7: [
            {"frame_index": 0, "center_x": 30, "center_y": 40},
            {"frame_index": 1, "center_x": 70, "center_y": 80},
        ]
    }
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    rendered = draw_tracks_on_frame(frame, history, main_track_id=7)
    assert rendered.shape == frame.shape
    assert int(rendered.sum()) > 0

    overlay = save_trajectory_overlay(frame, history, tmp_path / "overlay.png")
    blank = save_trajectory_blank(history, tmp_path / "blank.png", (160, 120))

    assert overlay.exists() and overlay.stat().st_size > 0
    assert blank.exists() and blank.stat().st_size > 0
    assert cv2.imread(str(blank)) is not None
