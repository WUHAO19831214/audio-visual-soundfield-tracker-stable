"""Streamlit interface for local visual counting and soundfield capture."""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.audio_devices import (
    AudioDeviceError,
    check_audio_input_device,
    get_default_input_device,
    list_audio_input_devices,
)
from src.camera_devices import list_available_cameras, read_camera_preview
from src.camera_processor import CameraProcessor
from src.config import LOG_DIR, OUTPUT_DIR as CONFIG_OUTPUT_DIR
from src.csv_logger import write_csv
from src.detector import Detector
from src.fusion import SynchronizedCaptureSession
from src.local_capture import (
    LocalCameraWorker,
    LocalFusionWorker,
    check_writable_directory,
)


OUTPUT_DIR = Path("data/output")
UPLOAD_DIR = Path("data/uploads")
MODES = ["图片计数", "视频计数", "摄像头实时计数", "视觉—音频同步采集"]
TRACK_FIELDS = [
    "timestamp",
    "track_id",
    "center_x",
    "center_y",
    "bbox_width",
    "bbox_height",
]


@st.cache_resource
def get_detector(confidence_threshold: float, person_only: bool) -> Detector:
    return Detector(
        confidence_threshold=confidence_threshold,
        person_only=person_only,
    )


def render_download(path: Path, label: str, mime: str) -> None:
    if path.exists():
        st.download_button(
            label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            key=f"download_{path.name}_{label}",
        )


def _browser_component_key(base: str, reset_generation: int) -> str:
    session_key = "browser_component_session_id"
    if session_key not in st.session_state:
        st.session_state[session_key] = uuid.uuid4().hex[:10]
    return f"{base}-v2-{st.session_state[session_key]}-{reset_generation}"


def image_counting_mode(detector: Detector) -> None:
    st.subheader("图片计数")
    uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is None:
        st.info("上传一张包含人物的图片开始检测。")
        return

    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        st.error("无法读取该图片。")
        return

    processor = CameraProcessor(detector=detector)
    annotated, detections = processor.detect_image(image)
    st.metric("检测到的 person 数量", len(detections))
    st.caption(f"检测后端：{processor.backend_name}")
    if detector.fallback_reason:
        st.warning(detector.fallback_reason)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)


def _process_uploaded_video(
    uploaded, confidence_threshold: float, person_only: bool
) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_path = UPLOAD_DIR / f"{timestamp}_{Path(uploaded.name).name}"
    input_path.write_bytes(uploaded.getvalue())
    output_path = OUTPUT_DIR / f"video_{timestamp}_tracked.mp4"
    csv_path = OUTPUT_DIR / f"video_{timestamp}_trajectory.csv"

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError("无法打开上传的视频。")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("无法创建输出视频，请检查本机 OpenCV 编码支持。")

    processor = CameraProcessor(
        Detector(
            confidence_threshold=confidence_threshold,
            person_only=person_only,
        )
    )
    trajectory: list[dict] = []
    unique_ids: set[int] = set()
    max_persons = 0
    frame_index = 0
    progress = st.progress(0.0, text="正在处理视频...")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_timestamp = frame_index / fps
            annotated, tracks, primary = processor.process_frame(frame, frame_timestamp)
            writer.write(annotated)
            unique_ids.update(int(track["track_id"]) for track in tracks)
            max_persons = max(max_persons, len(tracks))
            if primary is not None:
                trajectory.append(
                    {
                        "timestamp": frame_timestamp,
                        "track_id": primary["track_id"],
                        "center_x": primary["center_x"],
                        "center_y": primary["center_y"],
                        "bbox_width": primary["bbox_width"],
                        "bbox_height": primary["bbox_height"],
                    }
                )
            frame_index += 1
            if frame_index % 5 == 0:
                progress.progress(
                    min(frame_index / frame_count, 1.0), text="正在处理视频..."
                )
    finally:
        capture.release()
        writer.release()
        progress.empty()

    write_csv(csv_path, TRACK_FIELDS, trajectory)
    return {
        "video": output_path,
        "csv": csv_path,
        "unique_count": len(unique_ids),
        "max_persons": max_persons,
        "backend": processor.backend_name,
        "tracking_warning": getattr(processor, "tracking_warning", ""),
    }


def video_counting_mode(confidence_threshold: float, person_only: bool) -> None:
    st.subheader("视频计数与轨迹记录")
    uploaded = st.file_uploader("上传视频", type=["mp4", "mov", "avi", "mkv"])
    if uploaded is not None and st.button("开始处理视频", type="primary"):
        try:
            st.session_state.video_result = _process_uploaded_video(
                uploaded, confidence_threshold, person_only
            )
        except Exception as exc:
            st.error(f"视频处理失败：{exc}")

    result = st.session_state.get("video_result")
    if not result:
        return
    first, second = st.columns(2)
    first.metric("累计出现的 track_id", result["unique_count"])
    second.metric("单帧最大人数", result["max_persons"])
    st.caption(f"检测后端：{result['backend']}")
    if result.get("tracking_warning"):
        st.warning(result["tracking_warning"])
    st.video(str(result["video"]))
    render_download(result["video"], "下载标注视频", "video/mp4")
    render_download(result["csv"], "下载轨迹 CSV", "text/csv")


def _camera_index_controls(prefix: str) -> int:
    scan_key = f"{prefix}_camera_scan"
    if st.button("扫描本机摄像头 0–5", key=f"{prefix}_scan_button"):
        with st.spinner("正在逐个测试摄像头编号..."):
            st.session_state[scan_key] = list_available_cameras(max_index=5)

    devices = st.session_state.get(scan_key, [])
    names = {item["index"]: item["name"] for item in devices}
    camera_index = st.selectbox(
        "摄像头 index",
        options=list(range(6)),
        format_func=lambda index: names.get(index, f"Camera {index}（未扫描/未确认）"),
        key=f"{prefix}_camera_index",
    )
    if devices:
        st.caption("扫描到：" + "、".join(item["name"] for item in devices))
    if st.button("测试摄像头并读取一帧", key=f"{prefix}_test_button"):
        try:
            frame = read_camera_preview(int(camera_index))
            st.session_state[f"{prefix}_preview"] = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )
            st.success(f"Camera {camera_index} 可用")
        except Exception as exc:
            st.error(f"摄像头测试失败：{exc}")
    preview = st.session_state.get(f"{prefix}_preview")
    if preview is not None:
        st.image(preview, caption=f"Camera {camera_index} 单帧预览", width=480)
    return int(camera_index)


@st.fragment(run_every=0.2)
def _render_local_camera_worker(worker_key: str) -> None:
    worker = st.session_state.get(worker_key)
    if worker is None:
        return
    frame = worker.latest_frame()
    status = worker.status()
    if frame is not None:
        st.image(frame, channels="RGB", use_container_width=True)
    left, right = st.columns(2)
    left.metric("当前人数", status.get("count", 0))
    primary = status.get("primary")
    right.metric("主 track_id", primary["track_id"] if primary else "--")
    st.caption(f"检测后端：{status['backend']}")
    if status.get("tracking_warning"):
        st.warning(status["tracking_warning"])
    if status.get("error"):
        st.error(status["error"])


def _browser_camera_counting(
    confidence_threshold: float, person_only: bool
) -> None:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    from src.browser_capture import LiveCountingProcessor

    st.info(
        "如果要使用 iPhone Continuity Camera，请先允许浏览器使用默认摄像头，"
        "停止采集后点击 WebRTC 组件中的 Select Device，再选择名称包含 iPhone 的摄像头。"
    )
    with st.expander("浏览器看不到摄像头或 iPhone？"):
        st.markdown(
            "- 使用最新版 Chrome 或 Safari，并确认地址是 `localhost`。\n"
            "- 点击地址栏摄像头权限，先允许默认摄像头，再刷新页面。\n"
            "- 确认 Mac 与 iPhone 使用同一 Apple ID，Wi-Fi、蓝牙和接力已开启。\n"
            "- 浏览器设备列表由浏览器权限控制，Python 无法直接读取。"
        )
    reset_key = "camera_browser_device_reset"
    if st.button("重置浏览器设备选择", key="reset_browser_camera"):
        st.session_state[reset_key] = st.session_state.get(reset_key, 0) + 1
        st.info("已清除本页面使用的旧设备选择。请重新点击 Start 并授权。")
    component_generation = st.session_state.get(reset_key, 0)
    try:
        context = webrtc_streamer(
            key=_browser_component_key("camera-counting", component_generation),
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: LiveCountingProcessor(
                confidence_threshold, person_only
            ),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                },
                "audio": False,
            },
            async_processing=True,
        )
    except Exception as exc:
        st.error(f"浏览器摄像头组件启动失败：{exc}")
        st.info("请切换到“OpenCV 本机摄像头（推荐）”继续使用。")
        return
    count_slot = st.empty()
    track_slot = st.empty()
    while context.state.playing:
        processor = context.video_processor
        if processor is not None:
            status = processor.status()
            count_slot.metric("当前人数", status["count"])
            primary = status["primary"]
            track_slot.info(
                f"当前主 track_id：{primary['track_id']}"
                if primary
                else "当前未检测到 person"
            )
        time.sleep(0.2)


def camera_counting_mode(confidence_threshold: float, person_only: bool) -> None:
    st.subheader("摄像头实时计数")
    source = st.radio(
        "采集方式",
        [
            "浏览器摄像头（推荐用于 iPhone Continuity Camera）",
            "OpenCV 本机摄像头（推荐用于本地稳定实验）",
        ],
        horizontal=True,
        key="counting_capture_source",
    )
    if source.startswith("浏览器"):
        _browser_camera_counting(confidence_threshold, person_only)
        return

    camera_index = _camera_index_controls("counting")
    worker_key = "local_counting_worker"
    worker = st.session_state.get(worker_key)
    start_col, stop_col = st.columns(2)
    if start_col.button("开始实时检测", type="primary", disabled=bool(worker and worker.running)):
        try:
            if worker is not None:
                worker.stop()
            worker = LocalCameraWorker(
                camera_index,
                confidence_threshold=confidence_threshold,
                person_only=person_only,
            )
            worker.start()
            st.session_state[worker_key] = worker
        except Exception as exc:
            st.error(f"无法开始本机摄像头检测：{exc}")
    if stop_col.button("停止实时检测", disabled=not bool(worker and worker.running)):
        worker.stop()
    _render_local_camera_worker(worker_key)


def _display_capture_artifacts(artifacts) -> None:
    st.success("采集已停止，融合 CSV 和可视化文件已生成。")
    st.write(f"融合数据：`{artifacts.fused_csv}`")
    render_download(artifacts.fused_csv, "下载融合 CSV", "text/csv")

    plots = [
        (artifacts.trajectory_plot, "位置—时间轨迹"),
        (artifacts.intensity_plot, "空间声强分布"),
        (artifacts.frequency_plot, "空间主频分布"),
    ]
    for path, title in plots:
        if path and path.exists():
            st.markdown(f"**{title}**")
            st.image(str(path), use_container_width=True)

    with st.expander("原始数据文件"):
        render_download(artifacts.visual_csv, "下载视觉轨迹 CSV", "text/csv")
        render_download(artifacts.audio_csv, "下载音频特征 CSV", "text/csv")


def _browser_fusion_capture(
    confidence_threshold: float, person_only: bool
) -> None:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    from src.browser_capture import FusionAudioProcessor, FusionVideoProcessor

    st.info(
        "如果要使用 iPhone Continuity Camera，请先允许浏览器访问默认摄像头和麦克风，"
        "停止后点击 Select Device，在浏览器设备列表中选择 iPhone。"
    )
    with st.expander("浏览器设备列表排查"):
        st.markdown(
            "- 使用最新版 Chrome 或 Safari，并通过 `localhost` 打开本项目。\n"
            "- 在地址栏权限中允许摄像头和麦克风，然后刷新页面。\n"
            "- 先让默认设备成功启动，浏览器才会显示完整设备名称。\n"
            "- 确认同一 Apple ID、Wi-Fi、蓝牙和接力均已开启。\n"
            "- 浏览器设备列表由浏览器权限控制，需授权后由 WebRTC 组件显示。"
        )
    reset_key = "fusion_browser_device_reset"
    if st.button("重置浏览器音视频设备选择", key="reset_browser_fusion"):
        st.session_state[reset_key] = st.session_state.get(reset_key, 0) + 1
        st.info("已清除本页面使用的旧设备选择。请重新点击 Start 并授权。")
    component_generation = st.session_state.get(reset_key, 0)

    session = st.session_state.get("fusion_session")
    if session is None or session.finalized:
        session = SynchronizedCaptureSession(output_dir=OUTPUT_DIR)
        st.session_state.fusion_session = session

    try:
        context = webrtc_streamer(
            key=_browser_component_key(
                "audio-visual-fusion", component_generation
            ),
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: FusionVideoProcessor(
                session, confidence_threshold, person_only
            ),
            audio_processor_factory=lambda: FusionAudioProcessor(session),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280},
                    "height": {"ideal": 720},
                },
                "audio": True,
            },
            sendback_audio=False,
            async_processing=True,
        )
    except Exception as exc:
        st.error(f"浏览器音视频组件启动失败：{exc}")
        st.info("请切换到“OpenCV 摄像头 + sounddevice 麦克风（推荐）”。")
        return

    columns = st.columns(4)
    track_slot = columns[0].empty()
    position_slot = columns[1].empty()
    db_slot = columns[2].empty()
    frequency_slot = columns[3].empty()
    sample_slot = st.empty()

    was_playing = context.state.playing
    if was_playing:
        st.session_state.fusion_capture_active = True
    while context.state.playing:
        status = session.status()
        visual = status["visual"]
        audio = status["audio"]
        track_slot.metric("主 track_id", visual["track_id"] if visual else "--")
        position_slot.metric(
            "位置 (x, y)",
            f"{visual['center_x']:.0f}, {visual['center_y']:.0f}" if visual else "--",
        )
        db_slot.metric("相对声强", f"{audio['db']:.1f} dB" if audio else "--")
        frequency_slot.metric(
            "主频", f"{audio['dominant_frequency_hz']:.1f} Hz" if audio else "--"
        )
        sample_slot.caption(
            f"视觉样本 {status['visual_samples']} 条｜音频样本 {status['audio_samples']} 条｜"
            f"已采集 {status['elapsed_sec']:.1f} 秒"
        )
        time.sleep(0.1)

    should_finalize = (
        not context.state.playing
        and not session.finalized
        and (was_playing or st.session_state.get("fusion_capture_active", False))
    )
    if should_finalize:
        with st.spinner("正在对齐时间戳并生成图像..."):
            artifacts = session.finalize()
            st.session_state.last_fusion_artifacts = artifacts
            st.session_state.fusion_capture_active = False

    artifacts = st.session_state.get("last_fusion_artifacts")
    if artifacts is not None:
        _display_capture_artifacts(artifacts)


def _audio_device_selector() -> int | None:
    try:
        default_device = get_default_input_device()
        devices = list_audio_input_devices()
    except AudioDeviceError as exc:
        st.warning(str(exc))
        return None

    options: list[int | None] = [None] + [item["index"] for item in devices]
    device_names = {item["index"]: item["name"] for item in devices}
    default_text = (
        f"默认麦克风（{default_device['name']}）"
        if default_device
        else "默认麦克风"
    )
    selected = st.selectbox(
        "麦克风输入设备",
        options=options,
        format_func=lambda value: default_text
        if value is None
        else f"{value}: {device_names.get(value, '未知设备')}",
        key="local_audio_device",
    )
    return selected


def _run_local_preflight(camera_index: int, audio_device_index: int | None) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    try:
        frame = read_camera_preview(camera_index)
        checks.append(
            ("摄像头", True, f"Camera {camera_index}: {frame.shape[1]}x{frame.shape[0]}")
        )
    except Exception as exc:
        checks.append(("摄像头", False, str(exc)))

    try:
        audio = check_audio_input_device(audio_device_index)
        checks.append(("麦克风", True, f"{audio['name']} @ {audio['sample_rate']} Hz"))
    except AudioDeviceError as exc:
        checks.append(("麦克风", False, str(exc)))

    detector = Detector()
    checks.append(
        (
            "检测器",
            detector.backend_name == "YOLO",
            str(detector.weight_path) if detector.weight_path else detector.fallback_reason,
        )
    )
    for label, path in (("输出目录", CONFIG_OUTPUT_DIR), ("日志目录", LOG_DIR)):
        ok, detail = check_writable_directory(path)
        checks.append((label, ok, detail))
    return checks


def _show_preflight(checks: list[tuple[str, bool, str]]) -> None:
    for label, ok, detail in checks:
        if ok:
            st.success(f"{label}：{detail}")
        else:
            st.warning(f"{label}：{detail}")


@st.fragment(run_every=0.2)
def _render_local_fusion_worker(worker_key: str) -> None:
    worker = st.session_state.get(worker_key)
    if worker is None:
        return
    frame = worker.latest_frame()
    status = worker.status()
    if frame is not None:
        st.image(frame, channels="RGB", use_container_width=True)
    columns = st.columns(4)
    visual = status.get("visual")
    audio = status.get("audio")
    columns[0].metric("主 track_id", visual["track_id"] if visual else "--")
    columns[1].metric(
        "位置 (x, y)",
        f"{visual['center_x']:.0f}, {visual['center_y']:.0f}" if visual else "--",
    )
    columns[2].metric("相对声强", f"{audio['db']:.1f} dB" if audio else "--")
    columns[3].metric(
        "主频", f"{audio['dominant_frequency_hz']:.1f} Hz" if audio else "--"
    )
    st.caption(
        f"检测后端：{status['backend']}｜视觉样本 {status['visual_samples']} 条｜"
        f"音频样本 {status['audio_samples']} 条｜已采集 {status['elapsed_sec']:.1f} 秒"
    )
    if status.get("audio_error"):
        st.warning(
            "麦克风未正常采集，但摄像头仍在运行。错误：" + status["audio_error"]
        )
    if status.get("tracking_warning"):
        st.warning(status["tracking_warning"])
    if status.get("error"):
        st.error(status["error"])


def _local_fusion_capture(
    confidence_threshold: float, person_only: bool
) -> None:
    st.caption(
        "推荐本地实验使用此通路：OpenCV 直接读取摄像头，sounddevice 直接读取麦克风，"
        "不依赖浏览器 SELECT DEVICE。"
    )
    camera_index = _camera_index_controls("fusion")
    audio_device_index = _audio_device_selector()
    if st.button("运行采集前检查"):
        with st.spinner("正在检查摄像头、麦克风、检测器和输出目录..."):
            checks = _run_local_preflight(camera_index, audio_device_index)
            st.session_state.local_preflight_checks = checks
    checks = st.session_state.get("local_preflight_checks")
    if checks:
        with st.expander("采集前检查结果", expanded=True):
            _show_preflight(checks)

    worker_key = "local_fusion_worker"
    worker = st.session_state.get(worker_key)
    start_col, stop_col = st.columns(2)
    if start_col.button(
        "开始同步采集", type="primary", disabled=bool(worker and worker.running)
    ):
        checks = _run_local_preflight(camera_index, audio_device_index)
        st.session_state.local_preflight_checks = checks
        camera_ok = next(ok for label, ok, _ in checks if label == "摄像头")
        output_ok = all(
            ok for label, ok, _ in checks if label in ("输出目录", "日志目录")
        )
        if not camera_ok or not output_ok:
            st.error("摄像头或输出目录检查失败，未开始采集。请查看检查结果。")
        else:
            try:
                if worker is not None:
                    worker.stop()
                worker = LocalFusionWorker(
                    camera_index=camera_index,
                    audio_device_index=audio_device_index,
                    output_dir=OUTPUT_DIR,
                    confidence_threshold=confidence_threshold,
                    person_only=person_only,
                )
                worker.start()
                st.session_state[worker_key] = worker
            except Exception as exc:
                st.error(f"无法开始同步采集：{exc}")

    can_finalize = bool(worker and not worker.session.finalized)
    if stop_col.button("停止并生成结果", disabled=not can_finalize):
        try:
            with st.spinner("正在停止设备、融合时间戳并生成图像..."):
                artifacts = worker.stop_and_finalize()
                st.session_state.last_fusion_artifacts = artifacts
        except Exception as exc:
            st.error(f"停止或导出失败：{exc}")

    _render_local_fusion_worker(worker_key)
    artifacts = st.session_state.get("last_fusion_artifacts")
    if artifacts is not None:
        _display_capture_artifacts(artifacts)


def fusion_capture_mode(confidence_threshold: float, person_only: bool) -> None:
    st.subheader("视觉—音频同步采集")
    source = st.radio(
        "采集方式",
        [
            "浏览器摄像头 / 麦克风（推荐用于 iPhone Continuity Camera）",
            "OpenCV 摄像头 + sounddevice 麦克风（推荐用于本地稳定实验）",
        ],
        horizontal=True,
        key="fusion_capture_source",
    )
    if source.startswith("浏览器"):
        _browser_fusion_capture(confidence_threshold, person_only)
    else:
        _local_fusion_capture(confidence_threshold, person_only)


def _stop_inactive_local_workers(mode: str) -> None:
    counting_worker = st.session_state.get("local_counting_worker")
    if mode != "摄像头实时计数" and counting_worker and counting_worker.running:
        counting_worker.stop()

    fusion_worker = st.session_state.get("local_fusion_worker")
    if (
        mode != "视觉—音频同步采集"
        and fusion_worker
        and not fusion_worker.session.finalized
    ):
        st.session_state.last_fusion_artifacts = fusion_worker.stop_and_finalize()


def _render_device_diagnostics(detector: Detector) -> None:
    with st.sidebar.expander("设备诊断"):
        st.write(f"检测后端：**{detector.backend_name}**")
        st.caption(
            "YOLO 权重路径："
            + (str(detector.weight_path) if detector.weight_path else "未加载")
        )

        if st.button("扫描 OpenCV 摄像头 0–5", key="diagnostic_camera_scan"):
            with st.spinner("正在扫描摄像头编号..."):
                st.session_state.diagnostic_cameras = list_available_cameras(
                    max_index=5
                )
        cameras = st.session_state.get("diagnostic_cameras")
        if cameras is None:
            st.caption("OpenCV 摄像头：尚未扫描")
        elif cameras:
            st.caption(
                "OpenCV 可用 index："
                + "、".join(str(item["index"]) for item in cameras)
            )
        else:
            st.warning("OpenCV 未扫描到摄像头；这不代表浏览器无法使用 iPhone。")

        try:
            audio_devices = list_audio_input_devices()
            if audio_devices:
                st.caption("sounddevice 输入设备：")
                for item in audio_devices:
                    st.text(f"{item['index']}: {item['name']}")
            else:
                st.caption("sounddevice 未发现输入设备")
        except AudioDeviceError as exc:
            st.warning(str(exc))

        st.caption(
            "浏览器设备列表由浏览器权限控制，需先授权，再由 WebRTC 组件的 "
            "Select Device 显示。"
        )
        st.info(
            "推荐：iPhone Continuity Camera 使用浏览器模式；固定本地实验使用 "
            "OpenCV + sounddevice。"
        )


def main() -> None:
    st.set_page_config(page_title="视觉—声音空间分析", layout="wide")
    st.title("视觉—声音同步采集与空间声场分析系统")
    st.caption("本地实验原型｜图像平面坐标｜相对声强")

    mode = st.sidebar.radio("功能模式", MODES)
    _stop_inactive_local_workers(mode)
    st.sidebar.subheader("检测设置")
    confidence_threshold = st.sidebar.slider(
        "conf 阈值", min_value=0.1, max_value=0.9, value=0.25, step=0.05
    )
    person_only = st.sidebar.checkbox("只检测 person", value=True)
    detector = get_detector(confidence_threshold, person_only)
    st.sidebar.metric("检测后端", detector.backend_name)
    st.sidebar.caption(
        "YOLO 权重路径："
        + (str(detector.weight_path) if detector.weight_path else "未加载")
    )
    st.sidebar.caption("Tracker：bytetrack.yaml")
    if detector.backend_name != "YOLO":
        st.sidebar.warning(
            f"当前已回退到 OpenCV HOG。{detector.fallback_reason}"
        )
    _render_device_diagnostics(detector)

    if mode == "图片计数":
        image_counting_mode(detector)
    elif mode == "视频计数":
        video_counting_mode(confidence_threshold, person_only)
    elif mode == "摄像头实时计数":
        camera_counting_mode(confidence_threshold, person_only)
    else:
        fusion_capture_mode(confidence_threshold, person_only)


if __name__ == "__main__":
    main()
