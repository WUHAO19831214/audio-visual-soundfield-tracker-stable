# 视觉—声音同步采集与空间声场分析系统

这是一个完全本地运行的 Python/Streamlit 实验原型。当前版本保留图片计数、视频计数、摄像头实时计数和轨迹 CSV，并新增第 3 阶段的“视觉—音频同步采集”：追踪主 person 的图像位置，同时以 0.1 秒为周期提取麦克风特征，按统一时间戳融合并生成空间分布图。

## 功能

- 图片 person 计数与标注。
- 视频 person 计数、主目标轨迹记录、标注视频和 CSV 导出。
- 摄像头实时 person 计数与持续 `track_id`。
- 摄像头和麦克风同步采集。
- 实时显示主 `track_id`、`center_x/center_y`、相对 dB 和主频。
- 停止后导出视觉、音频和融合 CSV。
- 生成运动轨迹图、空间声强图、空间主频图。
- 优先使用本地 YOLOv8 权重，视频和实时模式使用 ByteTrack；只有 YOLO 不可用时才回退到 OpenCV HOG。
- 两种实时采集通路：浏览器 WebRTC，以及 OpenCV 摄像头 + sounddevice 麦克风。

## 安装与启动

推荐 Python 3.11 至 3.12。Apple Silicon macOS 应使用原生 ARM64 Python，以便直接安装 PyAV 二进制包。

```bash
cd /Users/wuhao/LocalProjects/Codex/macmini/audio-visual-soundfield-tracker
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

浏览器打开终端给出的本地地址，通常是 `http://localhost:8501`。第一次访问摄像头或麦克风时，macOS 和浏览器可能分别要求授权。所有结果保存在 `data/output/`，不接入云端或数据库。

## 提高识别质量：启用 YOLO

项目优先查找 `models/yolov8n.pt`，其次查找项目根目录下的 `yolov8n.pt`。找不到权重、未安装 `ultralytics` 或模型加载失败时才回退到 OpenCV HOG。

```bash
cd /Users/wuhao/LocalProjects/Codex/macmini/audio-visual-soundfield-tracker
source .venv/bin/activate
pip install -U ultralytics
mkdir -p models
```

将已有的 `yolov8n.pt` 放到：

```text
models/yolov8n.pt
```

也可以显式运行准备脚本。脚本会安装 Ultralytics、下载缺失权重并复制到 `models/`；应用本身不会静默下载模型：

```bash
./scripts/setup_yolo.sh
```

重新启动：

```bash
streamlit run app.py
```

侧边栏应显示：

```text
检测后端：YOLO
YOLO 权重路径：.../models/yolov8n.pt
Tracker：bytetrack.yaml
```

若仍显示 `OpenCV HOG`，侧边栏会同时给出缺少权重或加载失败原因。

## 摄像头与麦克风选择

### 两种采集方式

页面按用途提供两条通路：

- **浏览器摄像头 / 麦克风**：推荐用于 iPhone Continuity Camera，也支持 Mac 内置、USB 和虚拟摄像头。
- **OpenCV 摄像头 + sounddevice 麦克风**：推荐用于固定设备的本地稳定实验。

### OpenCV 本机摄像头

“摄像头实时计数”和“视觉—音频同步采集”都可以选择 OpenCV 本机摄像头。页面提供：

- 摄像头 index 0–5 下拉框；
- 扫描本机摄像头按钮；
- 读取单帧测试按钮；
- 开始和停止按钮。

macOS 上优先使用 `cv2.CAP_AVFOUNDATION`，失败后再尝试 OpenCV 默认后端。通常 Mac 内置摄像头是 index 0；iPhone Continuity Camera、USB 摄像头或虚拟摄像头可能是 1、2 或更高，具体以扫描和单帧预览为准。

同步采集页面会列出 sounddevice 可见的输入设备。“默认麦克风”不传设备编号，由 macOS 默认输入决定；也可以明确选择外接麦克风。麦克风失败时页面会显示原因，但摄像头检测不会因此停止。

### 浏览器摄像头与设备选择

第一次启动使用默认约束，不传递任何空 deviceId：

```text
video: { width: { ideal: 1280 }, height: { ideal: 720 } }
audio: true
```

浏览器授权后，WebRTC 组件通过 `enumerateDevices()` 显示 `Select Device`。只有浏览器返回了有效设备 ID 后才会使用该选择，不会构造 `deviceId: {exact: ""}`。页面还提供“重置浏览器设备选择”，用于清除已经断开或过期的设备 ID，然后重新从默认摄像头启动。

浏览器设备列表由浏览器权限控制，Python 端无法直接读取。若列表为空：

1. 使用最新版 Chrome 或 Safari；
2. 确认页面地址为 `localhost`；
3. 点击地址栏的摄像头权限并选择允许；
4. 先允许默认摄像头成功启动，再刷新页面并打开 `Select Device`。

## 使用 iPhone Continuity Camera

1. Mac 和 iPhone 登录同一个 Apple ID。
2. 打开两台设备的 Wi-Fi、蓝牙和接力。
3. 将 iPhone 靠近 Mac；必要时锁定 iPhone 屏幕。
4. 先在 FaceTime、Photo Booth 或其他应用中确认 iPhone 摄像头可用。
5. 在本项目中优先选择“浏览器摄像头 / 麦克风”模式。
6. 页面打开后先允许浏览器摄像头权限，让默认摄像头成功启动。
7. 停止采集后点击 WebRTC 组件的 `Select Device`；如果列表中出现 iPhone 名称，选择该设备再启动。
8. 如果设备列表仍没有 iPhone，检查浏览器地址栏权限，刷新页面，并确认 Continuity Camera 没有被其他应用占用。
9. OpenCV index 扫描不到 iPhone 是正常情况，不代表 Continuity Camera 不可用；此时继续使用浏览器模式。

### 测试默认摄像头和 OpenCV 兜底

1. 在 macOS“系统设置 → 隐私与安全性”中允许终端或 Codex 使用摄像头和麦克风。
2. 测试默认摄像头：选择 index 0，点击“测试摄像头并读取一帧”。
3. 点击“扫描本机摄像头 0–5”，依次测试其他编号以寻找 USB 或虚拟摄像头。
4. 如果 index 0 不可用，页面会显示错误，可改选其他 index，不会终止应用。

## 同步采集实验流程

1. 固定摄像头，确保实验区域完整入画，尽量避免明显背光。
2. 连接并选择麦克风。为了稳定，应关闭系统自动增益、降噪和回声消除（若设备允许）。
3. 在侧栏选择“视觉—音频同步采集”。iPhone 优先选择浏览器模式；固定外接设备可选择 OpenCV + sounddevice。
4. 浏览器模式先授权再通过 `Select Device` 选择设备；OpenCV 模式则选择 index 和麦克风并运行采集前检查。
5. 一个人手持或佩戴麦克风在画面内移动。只有一个 person 时自动选中；多人时优先选择持续时间较长且靠近画面中心的 `track_id`。绿色框和 `MAIN` 标签表示当前主目标。
6. 页面实时查看 `track_id`、位置、相对 dB 和主频。建议至少采集 10 秒，并缓慢覆盖需要分析的区域。
7. 点击“停止并生成结果”。系统自动导出融合 CSV，并生成轨迹、声强和主频三张图。
8. 从页面下载结果，或直接查看 `data/output/`。

## 时间同步规则

- `src/sync_clock.py` 用一个共享 `SyncClock` 为视觉帧和音频块生成时间戳。
- 导出的 `timestamp` 是 Unix 秒；内部用 `time.perf_counter()` 保持会话内单调递增。
- 视觉处理尽可能跟随输入帧率；实际帧率取决于检测后端和电脑性能。
- 音频默认每 `0.1` 秒提取一次特征，即约 10 fps。
- `src/fusion.py` 以每条音频记录为基准，使用二分查找寻找最近视觉帧。
- 时间差不超过 `0.15` 秒时 `matched=True`；超过阈值或当时没有主目标时记为 `matched=False`，位置字段留空。
- 未匹配音频仍保留在融合 CSV 中，便于检查丢帧和检测中断，而绘图只使用匹配成功的数据。

## 融合 CSV 字段

| 字段 | 含义 |
| --- | --- |
| `timestamp` | 音频块时间戳，Unix 秒 |
| `matched` | 是否匹配到 0.15 秒以内的视觉帧 |
| `time_diff_sec` | 最近视觉帧与音频块的绝对时间差 |
| `track_id` | 主追踪对象 ID |
| `center_x`, `center_y` | 检测框中心的图像像素坐标 |
| `bbox_width`, `bbox_height` | 检测框像素宽高 |
| `rms` | 音频均方根幅值 |
| `db` | 相对数字满刻度声强，dBFS |
| `dominant_frequency_hz` | FFT 最大非直流分量的频率 |
| `spectral_centroid_hz` | 频谱质心 |
| `zero_crossing_rate` | 过零率 |

## 输出文件

每次同步采集停止后会生成：

- `capture_*_visual.csv`：主目标视觉轨迹原始记录。
- `capture_*_audio.csv`：0.1 秒音频特征原始记录。
- `capture_*_fused.csv`：按音频时间戳融合后的数据。
- `capture_*_fused_trajectory.png`：位置轨迹。
- `capture_*_fused_sound_intensity.png`：空间相对声强分布。
- `capture_*_fused_dominant_frequency.png`：空间主频分布。

视频模式另行生成标注 MP4 和主目标轨迹 CSV。

## 实验边界与注意事项

1. 第一版记录的是摄像头图像平面坐标，不是真实空间坐标。坐标原点在图像左上角，`x` 向右、`y` 向下；绘图时会翻转 y 轴以保持直观。
2. 若需要真实空间坐标，后续需要加入标定板或 ArUco marker，完成透视校正和像素到实际长度的转换。
3. 当前 `db` 是根据数字音频幅值计算的相对声强（dBFS），不是经过声级计校准的绝对声压级 dB SPL。浏览器或操作系统的自动增益也可能改变结果。
4. 若要获得更准确的声场，应使用频响稳定的外接麦克风，固定输入增益，并使用声级计或校准器进行声学校准。
5. 摄像头应固定不动；镜头移动会让图像坐标失去统一空间参考。
6. 当前系统适合课堂实验原型、相对分布分析和教学演示，不适合作为法定计量或工程验收设备。

## 测试

```bash
source .venv/bin/activate
python -m pip install pytest
pytest -q
```

自动测试覆盖 YOLO/HOG 后端选择、设备枚举失败处理、Streamlit 同步采集页面、0.1 秒音频分块、0.15 秒融合阈值、主目标选择和三类图像导出。真实摄像头、Continuity Camera 与麦克风权限仍需在本机人工验收。

## 代码结构

```text
app.py                         Streamlit 页面和 WebRTC 处理器
src/browser_capture.py         按需加载的 WebRTC/PyAV 处理器
src/sync_clock.py              视觉/音频共享时钟
src/config.py                  模型、输出目录和默认检测配置
src/detector.py                YOLO 优先检测、ByteTrack 与 HOG 回退
src/camera_processor.py        track_id 历史、主目标选择和画面标注
src/camera_devices.py          OpenCV/AVFoundation 摄像头枚举
src/audio_devices.py           sounddevice 麦克风枚举
src/local_capture.py           本机摄像头和麦克风后台采集线程
src/audio_recorder.py          音频缓冲和 0.1 秒分块
src/audio_features.py          RMS、dB、主频、频谱质心、过零率
src/fusion.py                  最近时间戳匹配和会话导出
src/soundfield_visualizer.py   轨迹、声强、主频空间图
src/csv_logger.py              CSV 写入工具
tests/                         可重复的自动测试
```
