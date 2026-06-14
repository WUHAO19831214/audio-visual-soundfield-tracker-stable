from pathlib import Path

import numpy as np

from src.detector import Detector


class FakeYoloModel:
    names = {0: "person"}

    def predict(self, **kwargs):
        return []

    def track(self, **kwargs):
        return []


def test_detector_uses_yolo_when_local_weight_loads(tmp_path: Path) -> None:
    weight = tmp_path / "yolov8n.pt"
    weight.write_bytes(b"fake")

    detector = Detector(
        model_paths=[weight], yolo_factory=lambda path: FakeYoloModel()
    )

    assert detector.backend_name == "YOLO"
    assert detector.weight_path == weight.resolve()
    assert detector.supports_native_tracking is True


def test_missing_weight_falls_back_without_crashing(tmp_path: Path) -> None:
    detector = Detector(model_paths=[tmp_path / "missing.pt"])

    assert detector.backend_name == "OpenCV HOG"
    assert "未找到" in detector.fallback_reason


def test_failed_yolo_load_falls_back_to_hog(tmp_path: Path) -> None:
    weight = tmp_path / "yolov8n.pt"
    weight.write_bytes(b"fake")

    def fail_to_load(path):
        raise RuntimeError("bad weight")

    detector = Detector(model_paths=[weight], yolo_factory=fail_to_load)
    detections = detector.detect(np.zeros((128, 64, 3), dtype=np.uint8))

    assert detector.backend_name == "OpenCV HOG"
    assert detections == []
    assert "bad weight" in detector.fallback_reason

