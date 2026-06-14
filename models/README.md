# Local detector weights

The application first looks for `models/yolov8n.pt` and uses Ultralytics YOLO
with ByteTrack when it is available. OpenCV HOG is only a fallback.

Run `scripts/setup_yolo.sh` or place a local Ultralytics model at
`models/yolov8n.pt`. The application only loads existing local files; downloads
are performed explicitly by the setup script.
