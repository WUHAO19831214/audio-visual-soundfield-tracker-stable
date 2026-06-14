"""Shared runtime configuration for detection and local capture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
PRIMARY_YOLO_WEIGHTS = MODEL_DIR / "yolov8n.pt"
SECONDARY_YOLO_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
LOG_DIR = PROJECT_ROOT / "data" / "logs"


@dataclass(frozen=True)
class DetectionConfig:
    confidence_threshold: float = 0.25
    person_only: bool = True
    tracker: str = "bytetrack.yaml"

