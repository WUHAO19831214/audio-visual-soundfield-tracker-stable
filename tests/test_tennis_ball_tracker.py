import cv2
import numpy as np
import pytest

from src.tennis_ball_tracker import (
    TennisBallTracker,
    choose_best_candidate,
    estimate_hsv_range_from_roi,
    find_ball_candidates,
    make_tennis_mask,
)


def tennis_frame(center=(80, 60), radius=14) -> np.ndarray:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.circle(frame, center, radius, (0, 255, 255), -1)
    return frame


def test_make_mask_and_find_candidates() -> None:
    frame = tennis_frame()

    mask = make_tennis_mask(frame, (20, 100, 100), (40, 255, 255))
    candidates = find_ball_candidates(mask)

    assert mask.shape == frame.shape[:2]
    assert np.count_nonzero(mask) > 0
    assert len(candidates) == 1
    assert candidates[0]["area"] > 400
    assert candidates[0]["circularity"] > 0.7


def test_choose_best_candidate_uses_previous_center() -> None:
    candidates = [
        {"center_x": 20.0, "center_y": 20.0, "area": 500.0, "circularity": 0.9},
        {"center_x": 100.0, "center_y": 80.0, "area": 900.0, "circularity": 0.9},
    ]

    selected = choose_best_candidate(candidates, previous_center=(18, 19))

    assert selected is candidates[0]


def test_estimate_hsv_range_contains_roi_color() -> None:
    frame = tennis_frame()

    lower, upper = estimate_hsv_range_from_roi(frame, (67, 47, 27, 27))
    center_hsv = cv2.cvtColor(
        np.array([[[0, 255, 255]]], dtype=np.uint8), cv2.COLOR_BGR2HSV
    )[0, 0]

    assert lower[0] <= int(center_hsv[0]) <= upper[0]
    assert lower[1] <= int(center_hsv[1]) <= upper[1]
    assert lower[2] <= int(center_hsv[2]) <= upper[2]


@pytest.mark.parametrize(
    "bbox",
    [(0, 0, 0, 10), (-1, 0, 10, 10), (150, 10, 20, 20)],
)
def test_estimate_hsv_rejects_invalid_roi(bbox) -> None:
    with pytest.raises(ValueError):
        estimate_hsv_range_from_roi(tennis_frame(), bbox)


def test_tracker_detects_marker_and_smooths_center() -> None:
    tracker = TennisBallTracker(
        hsv_lower=(20, 100, 100),
        hsv_upper=(40, 255, 255),
        min_area=100,
        max_area=2000,
        min_circularity=0.6,
        smoothing=0.5,
    )

    first = tracker.update(tennis_frame(center=(60, 60)))
    second = tracker.update(tennis_frame(center=(80, 60)))

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["tracking_mode"] == "tennis_ball_color"
    assert second["class_name"] == "tennis_ball_marker"
    assert second["track_id"] == 1
    assert 68 <= second["center_x"] <= 72
    assert second["marker_radius"] > 10
    assert second["marker_circularity"] > 0.6


def test_tracker_lost_state_is_nonfatal() -> None:
    tracker = TennisBallTracker(
        hsv_lower=(20, 100, 100),
        hsv_upper=(40, 255, 255),
        min_area=100,
        max_area=2000,
    )

    result = tracker.update(np.zeros((120, 160, 3), dtype=np.uint8))

    assert result["ok"] is False
    assert result["tracking_status"] == "lost"
    assert result["lost_frame_count"] == 1
    assert tracker.tracking_success_rate == 0.0
