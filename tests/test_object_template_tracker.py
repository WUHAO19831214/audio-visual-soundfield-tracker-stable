import cv2
import numpy as np
import pytest

import src.object_template_tracker as object_tracker


def test_validate_bbox_accepts_valid_box() -> None:
    frame = np.zeros((100, 160, 3), dtype=np.uint8)

    assert object_tracker.validate_bbox(frame, (10, 20, 40, 30)) == (
        10.0,
        20.0,
        40.0,
        30.0,
    )


@pytest.mark.parametrize(
    "bbox",
    [
        (10, 10, 0, 20),
        (10, 10, 20, 0),
        (-1, 10, 20, 20),
        (150, 10, 20, 20),
        (10, 90, 20, 20),
    ],
)
def test_validate_bbox_rejects_invalid_boxes(bbox) -> None:
    frame = np.zeros((100, 160, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        object_tracker.validate_bbox(frame, bbox)


def test_create_opencv_tracker_is_available_or_friendly() -> None:
    try:
        tracker = object_tracker.create_opencv_tracker("MIL")
    except RuntimeError as exc:
        assert "opencv-contrib-python" in str(exc)
    else:
        assert tracker is not None


def test_initialize_failure_is_non_fatal(monkeypatch) -> None:
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    monkeypatch.setattr(
        object_tracker,
        "create_opencv_tracker",
        lambda tracker_type: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )
    tracker = object_tracker.ObjectTemplateTracker("CSRT")

    assert tracker.initialize(frame, (10, 10, 20, 20)) is False
    assert tracker.initialized is False
    assert "unavailable" in tracker.last_error


def test_update_before_initialize_returns_lost() -> None:
    tracker = object_tracker.ObjectTemplateTracker()
    frame = np.zeros((100, 160, 3), dtype=np.uint8)

    result = tracker.update(frame)

    assert result["ok"] is False
    assert result["tracking_status"] == "lost"
    assert result["tracking_mode"] == "custom_object_template"
    assert result["track_id"] == 1


def test_initialize_passes_integer_bbox_to_opencv(monkeypatch) -> None:
    class FakeTracker:
        received_bbox = None

        def init(self, frame, bbox):
            self.received_bbox = bbox
            return None

    fake = FakeTracker()
    monkeypatch.setattr(
        object_tracker,
        "create_opencv_tracker",
        lambda tracker_type: fake,
    )
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    tracker = object_tracker.ObjectTemplateTracker("MIL")

    assert tracker.initialize(frame, (10.2, 20.4, 40.1, 30.3)) is True
    assert fake.received_bbox == (10, 20, 40, 30)


def test_current_opencv_capability_is_reportable() -> None:
    available = []
    for tracker_type in object_tracker.TRACKER_FALLBACK_ORDER:
        try:
            object_tracker.create_opencv_tracker(tracker_type)
        except RuntimeError:
            continue
        available.append(tracker_type)

    assert isinstance(available, list)
    assert cv2.__version__
