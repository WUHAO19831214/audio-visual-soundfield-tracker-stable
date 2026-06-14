from src.fusion import FUSED_FIELDS, fuse_records


def audio_row(timestamp: float) -> dict:
    return {
        "timestamp": timestamp,
        "rms": 0.1,
        "db": -20.0,
        "dominant_frequency_hz": 440.0,
        "spectral_centroid_hz": 500.0,
        "zero_crossing_rate": 0.05,
    }


def test_audio_rows_match_nearest_visual_timestamp() -> None:
    visuals = [
        {
            "timestamp": 1.00,
            "track_id": 7,
            "center_x": 100,
            "center_y": 200,
            "bbox_width": 50,
            "bbox_height": 120,
        },
        {
            "timestamp": 1.20,
            "track_id": 7,
            "center_x": 120,
            "center_y": 210,
            "bbox_width": 50,
            "bbox_height": 120,
        },
    ]

    fused = fuse_records(visuals, [audio_row(1.16)])

    assert list(fused[0]) == FUSED_FIELDS
    assert fused[0]["matched"] is True
    assert fused[0]["track_id"] == 7
    assert fused[0]["center_x"] == 120
    assert fused[0]["time_diff_sec"] == 0.04


def test_audio_row_is_unmatched_after_threshold() -> None:
    visuals = [
        {
            "timestamp": 1.0,
            "track_id": 2,
            "center_x": 10,
            "center_y": 20,
            "bbox_width": 30,
            "bbox_height": 40,
        }
    ]

    fused = fuse_records(visuals, [audio_row(1.151)])

    assert fused[0]["matched"] is False
    assert fused[0]["track_id"] == ""
    assert fused[0]["time_diff_sec"] == 0.151

