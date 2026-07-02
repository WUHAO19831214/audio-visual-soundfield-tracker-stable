#!/usr/bin/env python3
"""Check whether RF-DETR can be imported without making it a requirement."""

from __future__ import annotations

import importlib.util


def main() -> int:
    candidates = ("rfdetr", "rfdetr_detector")
    found = [name for name in candidates if importlib.util.find_spec(name)]
    if found:
        print("RF-DETR optional package detected:", ", ".join(found))
        print("当前项目仍以 YOLO 为主检测后端；RF-DETR 仅建议用于对比实验。")
        return 0
    print("RF-DETR optional package not installed.")
    print("这是正常状态：requirements.txt 不包含 RF-DETR，当前项目默认使用 YOLO。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
