# 视觉—声音同步采集：稳定基线

本仓库是 `audio-visual-soundfield-tracker` 的**稳定基线版本**，用于复现实验、课堂演示和故障回退。它不是开发仓库的镜像，也不会自动包含开发版中的双摄与三维实验功能。

## 基线身份

| 项目 | 记录 |
| --- | --- |
| 稳定仓库 commit | `6c5c6b7ca23155db58b63794993322fbdeb8f868` |
| 对应开发仓库 commit | `6c5c6b7ca23155db58b63794993322fbdeb8f868` |
| 基线版本标识 | `stable-baseline-2026-07-11`（文档标识，不是 Git tag） |
| 对应 Git tag | 尚未创建；当前以完整 commit SHA 为唯一版本锚点 |
| 基线日期 | 2026-07-11 |
| 最后验证日期 | 2026-07-13 |
| 验证计算机 | MacBook Air (M2, 16 GB) |
| 验证操作系统 | macOS 26.3.1 (Build 25D2128) |
| 自动验证 | `compileall` 通过；`pytest -q` 为 64 passed |

完整的版本边界、设备说明和复现命令见 [STABLE_BASELINE.md](STABLE_BASELINE.md)。

## 已纳入的稳定功能

- 图片与视频 person 计数、标注和轨迹导出；
- 单摄像头实时 person 计数，YOLOv8 + ByteTrack 优先，OpenCV HOG 回退；
- 单摄像头视觉—音频同步采集；
- 浏览器 WebRTC 摄像头和 OpenCV 本机摄像头两条采集路径；
- `sounddevice` 麦克风枚举、输入测试与诊断；
- person、指定物体模板、HSV 网球标记三种追踪方式；
- RMS、相对 dBFS、主频、频谱质心和过零率；
- visual/audio/fused CSV、二维轨迹图和图像平面声学分布图。

## 明确不包含

本稳定基线不包含开发版在后续提交中加入的：

- 双摄像头实验模式和双摄启动预热/重试；
- 双摄虚拟三维轨迹；
- 交互式 Plotly 三维视角及三维声学叠加；
- 棋盘格双目标定、极线校正和米制三角测量；
- `virtual_*`、`world_*_m` 等双摄 CSV 字段。

如需要这些功能，请使用开发仓库并明确记录所用 commit；不要把开发版 README 复制到本仓库。

## 安装与启动

推荐 Python 3.11–3.12，Apple Silicon 使用 ARM64 Python：

```bash
git clone https://github.com/WUHAO19831214/audio-visual-soundfield-tracker-stable.git
cd audio-visual-soundfield-tracker-stable
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

打开 `http://localhost:8501`。首次使用需允许终端/浏览器访问摄像头与麦克风。YOLO 权重放在 `models/yolov8n.pt`；没有权重时允许回退到 HOG。

## 最小复现检查

```bash
source .venv/bin/activate
python -m compileall app.py src tests
pytest -q
streamlit run app.py
curl -I http://localhost:8501
```

HTTP 200 仅说明服务可访问；摄像头、麦克风和具体实验仍须在目标设备上单独验证。

## 数据与物理含义

- `center_x/center_y` 是图像像素坐标，不是实际米制坐标。
- `rms` 是归一化 PCM 的均方根幅值；`db` 是 dBFS，不是 dB SPL。
- 融合以音频时间戳为基准，寻找 0.15 秒内最近的有效视觉帧。
- 连续声场色块来自实测匹配点之间的三角剖分插值；散点才是实测位置。
- 摄像头必须固定；若麦克风不随测点移动，图只表示位置与固定麦克风读数的关联。

## 输出

同步采集会在 `data/output/` 生成 `capture_*_visual.csv`、`*_audio.csv`、`*_fused.csv`、summary JSON 及二维轨迹/声学 PNG。原始采集数据、视频、音频、图片和模型权重不应提交到 Git。

## 维护原则

只有经过明确验证的开发仓库 commit/tag 才能升级为新稳定基线。升级时必须同步更新 `STABLE_BASELINE.md` 中的来源版本、验证日期、设备、测试结果、样例数据说明和开发版差异。
