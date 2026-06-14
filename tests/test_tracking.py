import numpy as np

from src.camera_processor import CameraProcessor, PrimaryTrackSelector, Track


def make_track(track_id: int, x: int, hits: int) -> Track:
    return Track(
        track_id=track_id,
        x1=x - 20,
        y1=100,
        x2=x + 20,
        y2=220,
        confidence=0.9,
        first_seen=0.0,
        last_seen=float(hits),
        hits=hits,
    )


def test_single_person_is_selected_automatically() -> None:
    selector = PrimaryTrackSelector()
    track = make_track(3, 100, 1)

    assert selector.select([track], (480, 640, 3)).track_id == 3


def test_persistent_center_track_is_preferred() -> None:
    selector = PrimaryTrackSelector()
    short_edge_track = make_track(1, 20, 2)
    persistent_center_track = make_track(2, 320, 12)

    selected = selector.select(
        [short_edge_track, persistent_center_track], (480, 640, 3)
    )

    assert selected.track_id == 2


class DetectorWithoutTrackingWarning:
    backend_name = "test"
    supports_native_tracking = False

    def detect(self, frame):
        return []


def test_camera_processor_always_has_tracking_warning() -> None:
    processor = CameraProcessor(detector=DetectorWithoutTrackingWarning())

    assert processor.tracking_warning == ""
    processor.process_frame(np.zeros((32, 32, 3), dtype=np.uint8), 1.0)
    assert processor.tracking_warning == ""
