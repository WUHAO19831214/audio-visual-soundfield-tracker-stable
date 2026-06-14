#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Create it with: python3.11 -m venv .venv"
  exit 1
fi

source .venv/bin/activate
python -m pip install -U ultralytics
mkdir -p models

python - <<'PY'
from pathlib import Path
import shutil

from ultralytics import YOLO

target = Path("models/yolov8n.pt")
if target.exists():
    print(f"YOLO weights already exist: {target}")
else:
    print("Downloading or preparing yolov8n.pt ...")
    model = YOLO("yolov8n.pt")
    candidates = [Path("yolov8n.pt")]
    ckpt_path = getattr(model, "ckpt_path", None)
    if ckpt_path:
        candidates.append(Path(ckpt_path))
    source = next((path for path in candidates if path.exists()), None)
    if source is None:
        raise SystemExit(
            "Ultralytics loaded the model but its weight file was not found. "
            "Place yolov8n.pt manually at models/yolov8n.pt."
        )
    shutil.copy2(source, target)
    print(f"Copied weights to: {target}")

YOLO(str(target))
print("YOLO setup finished.")
PY
