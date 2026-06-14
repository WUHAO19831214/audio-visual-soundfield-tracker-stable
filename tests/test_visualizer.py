from pathlib import Path

import pandas as pd

import src.soundfield_visualizer as visualizer


def test_all_visualizations_are_saved(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "fused.csv"
    pd.DataFrame(
        {
            "timestamp": [1.0, 1.1, 1.2, 1.3],
            "matched": [True, True, True, True],
            "center_x": [100, 200, 100, 200],
            "center_y": [100, 100, 200, 200],
            "db": [-30, -20, -25, -15],
            "dominant_frequency_hz": [220, 440, 330, 550],
        }
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(visualizer, "DEFAULT_OUTPUT_DIR", tmp_path)

    paths = [
        visualizer.plot_trajectory(csv_path),
        visualizer.plot_sound_intensity_map(csv_path),
        visualizer.plot_frequency_map(csv_path),
    ]

    assert all(path.exists() and path.stat().st_size > 0 for path in paths)

